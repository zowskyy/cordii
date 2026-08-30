import json
from pathlib import Path

from core.context import Context
from core.messages import Message
from core.plugin import Plugin
from core.reality import RealityProjector
from core.registry import PluginRegistry
from plugins.agent.loop import AgentLoop, TOKEN_BUDGET
from plugins.core.event_logger import EventLogger
from plugins.tools.file import FileTools


class FakeModel(Plugin):
    name = "ollama_model"

    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools):
        self.calls += 1
        return self.responses.pop(0)


def build_agent(tmp_path, responses, config=None):
    from core.messages import Message

    ctx = Context(config=config or {})
    reg = PluginRegistry(ctx)
    reg.register(EventLogger(tmp_path / "test.db"))
    reg.register(FakeModel(responses))
    reg.register(FileTools(tmp_path))
    reg.register(AgentLoop())
    reg.start_all()
    return ctx, reg


def tc(name, args):
    return {"function": {"name": name, "arguments": args}}


def test_agent_single_tool_call_roundtrip(tmp_path):
    from core.messages import Message

    tool_call = tc("write_file", {"path": "a.txt", "content": "hello"})
    ctx, reg = build_agent(tmp_path, [Message("assistant", "", tool_calls=[tool_call]), Message("assistant", "done")])
    assert ctx.plugins["agent_loop"].run("create a file") == "done"
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "hello"
    assert ctx.messages[2].tool_calls == [tool_call]
    assert ctx.messages[3].role == "tool"
    reg.stop_all()


def test_agent_multiple_tool_calls(tmp_path):
    from core.messages import Message

    calls = [tc("write_file", {"path": "a.txt", "content": "a"}), tc("write_file", {"path": "b.txt", "content": "b"})]
    ctx, reg = build_agent(tmp_path, [Message("assistant", "", tool_calls=calls), Message("assistant", "both done")])
    result = ctx.plugins["agent_loop"].run("make two files")
    # 3x: loop now continues until model emits done (not auto-done), so result is model-provided "both done"
    assert "done" in result.lower()
    assert (tmp_path / "a.txt").read_text(encoding="utf-8") == "a"
    assert (tmp_path / "b.txt").read_text(encoding="utf-8") == "b"
    reg.stop_all()


def test_agent_handles_tool_error(tmp_path):
    from core.messages import Message

    bad_call = tc("read_file", {"path": "../secret.txt"})
    ctx, reg = build_agent(tmp_path, [Message("assistant", "", tool_calls=[bad_call]), Message("assistant", "I recovered")])
    assert ctx.plugins["agent_loop"].run("read secret") == "I recovered"
    tool_msgs = [m for m in ctx.messages if m.role == "tool"]
    payload = json.loads(tool_msgs[0].content)
    assert "error" in payload
    reg.stop_all()


def test_agent_maintains_history_across_rounds(tmp_path):
    from core.messages import Message

    ctx, reg = build_agent(tmp_path, [Message("assistant", "first")])
    assert ctx.plugins["agent_loop"].run("hello") == "first"
    assert [m.role for m in ctx.messages] == ["user", "system", "assistant"]
    reg.stop_all()


def test_agent_emits_events(tmp_path):
    from core.messages import Message

    tool_call = tc("write_file", {"path": "a.txt", "content": "hello"})
    ctx, reg = build_agent(tmp_path, [Message("assistant", "", tool_calls=[tool_call]), Message("assistant", "done")])
    ctx.plugins["agent_loop"].run("create a file")
    types = [e.type for e in ctx.plugins["event_log"].get_session_events(ctx.plugins["continuity"].session_id)]
    assert {"session.start", "user.message", "tool.result"}.issubset(types)
    reg.stop_all()


class CapturingModel(Plugin):
    name = "ollama_model"
    dependencies = ()

    def __init__(self):
        super().__init__()
        self.seen: list = []

    def chat(self, messages, tools):
        self.seen.append(list(messages))
        return Message("assistant", "done")


def test_agent_lite_uses_envelope(tmp_path):
    ctx = Context(config={"profile": "lite", "workspace": str(tmp_path)})
    reg = PluginRegistry(ctx)
    reg.register(EventLogger(tmp_path / "test.db"))
    model = CapturingModel()
    reg.register(model)
    reg.register(FileTools(tmp_path))
    reg.register(AgentLoop(max_rounds=3))
    reg.start_all()
    try:
        result = ctx.plugins["agent_loop"].run("hello")
        assert result == "done"
        loop = ctx.plugins["agent_loop"]
        fed = model.seen[-1]
        # Invariant 1: lite feeds the compiled envelope, never the mutable cache.
        assert fed is not ctx.messages
        assert [m.role for m in fed] == ["system", "user"]
        assert fed[0].role == "system"
        assert fed[0].content == loop._system_prompt
        assert fed[1].role == "user"
        assert "hello" in fed[1].content
        # Invariant 3: compilation is deterministic over (log, manifest, assets).
        event_log = ctx.plugins["event_log"]
        sid = loop._get_session_id()
        a = RealityProjector(event_log).compile_request(sid, loop._manifest, loop._system_prompt, loop._tool_schemas, TOKEN_BUDGET)
        b = RealityProjector(event_log).compile_request(sid, loop._manifest, loop._system_prompt, loop._tool_schemas, TOKEN_BUDGET)
        assert a.full_request_hash == b.full_request_hash
        assert a.request_prefix_hash == b.request_prefix_hash
    finally:
        reg.stop_all()


def test_tool_result_truncated_to_calibrated_limit(tmp_path):
    """4k window protection: a single tool result must never exceed its
    calibrated byte cap (1.5b preset: 8192 bytes) — one result cannot swallow
    the window regardless of file size on disk. Truncation is marked so the
    model knows the file continues."""
    from core.context import MODEL_PRESETS, DEFAULT_PRESET_KEY

    ctx, reg = build_agent(tmp_path, [Message("assistant", "done")])
    try:
        limit = MODEL_PRESETS[DEFAULT_PRESET_KEY]["max_tool_result_bytes"]
        loop = ctx.plugins["agent_loop"]
        loop._record_tool_result("read_file", {"path": "big.txt"}, "x" * (limit + 5000), True)
        content = ctx.messages[-1].content
        assert len(content.encode("utf-8")) <= limit + 100  # + marker overhead
        assert content[:limit] == "x" * limit
        assert "truncated" in content
    finally:
        reg.stop_all()


def test_tool_result_limit_follows_calibration_override(tmp_path):
    """Calibration separation: the tool-result cap comes from
    Context.config["calibration"] (merged over the preset), not a loop literal."""
    ctx, reg = build_agent(
        tmp_path, [Message("assistant", "done")],
        config={"calibration": {"max_tool_result_bytes": 100}},
    )
    try:
        loop = ctx.plugins["agent_loop"]
        assert loop._max_result_bytes == 100
        loop._record_tool_result("read_file", {"path": "b.txt"}, "y" * 500, True)
        content = ctx.messages[-1].content
        assert len(content.encode("utf-8")) <= 100 + 100
        assert "truncated" in content
    finally:
        reg.stop_all()


def test_multi_domain_llm_fallback_gated_by_profile_and_flag():
    """P0 zero-token guarantee: the multi-domain LLM fallback for unresolved
    fragments must run ONLY in the full profile AND with --enable-semantic-router.
    Otherwise the multi-domain path is abandoned (query falls through to the
    deterministic routers + agent loop) and _call_llm_directly is never called."""
    from plugins.agent.multi_domain_router import DomainResult, MultiDomainResult
    from plugins.agent.query_splitter import Fragment

    class _StubMultiDomain:
        def route_multi(self, text, ctx):
            return MultiDomainResult(
                results=[
                    DomainResult(fragment=Fragment(text="derivative", domain="math"), domain="math", response="2*x"),
                    DomainResult(fragment=Fragment(text="pizza", domain="general"), domain="general", response=None),
                ],
                has_unresolved=True,
            )

    def make_loop(config):
        loop = AgentLoop()
        loop.register(Context(config=config))
        loop._multi_domain_router = _StubMultiDomain()
        loop._aggregator = object()
        routed = []
        loop._call_llm_directly = lambda text: (routed.append(text) or "llm-answer")
        return loop, routed

    query = "What is the derivative of x squared? And how is pizza made?"

    # lite (default profile): no routing LLM, multi-domain abandoned
    loop, routed = make_loop({"profile": "lite"})
    loop._try_multi_domain(query)
    assert routed == []
    assert loop._multi_domain_results == []

    # full without flag: still no routing LLM
    loop, routed = make_loop({"profile": "full"})
    loop._try_multi_domain(query)
    assert routed == []
    assert loop._multi_domain_results == []

    # full + explicit flag: fallback allowed
    loop, routed = make_loop({"profile": "full", "semantic_router_enabled": True})
    loop._try_multi_domain(query)
    assert routed == ["pizza"]
    assert len(loop._multi_domain_results) == 2
