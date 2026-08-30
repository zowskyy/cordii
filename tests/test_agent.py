import json
from pathlib import Path

from core.context import Context
from core.plugin import Plugin
from core.registry import PluginRegistry
from plugins.agent.loop import AgentLoop
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


def build_agent(tmp_path, responses):
    from core.messages import Message

    ctx = Context()
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
    assert ctx.plugins["agent_loop"].run("make two files") == "done"
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
