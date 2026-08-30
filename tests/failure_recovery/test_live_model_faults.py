from __future__ import annotations

import socket
import tempfile
from pathlib import Path

import pytest

from core.context import Context
from core.messages import Message
from core.registry import PluginRegistry
from plugins.agent.loop import AgentLoop
from plugins.core.event_logger import EventLogger
from plugins.model.ollama import OllamaModel
from plugins.tools.file import FileTools


def _ollama_available():
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=1):
            return True
    except OSError:
        return False


REAL_MODEL = "qwen2.5-coder:1.5b"
OLLAMA_URL = "http://127.0.0.1:11434"


def _live_agent(tmp_path, rounds=3):
    workspace = Path(tmp_path) / "workspace"
    workspace.mkdir()
    ctx = Context(config={"workspace": str(workspace), "model": REAL_MODEL, "ollama_url": OLLAMA_URL})
    reg = PluginRegistry(ctx)
    reg.register(EventLogger(Path(tmp_path) / "test.db"))
    reg.register(OllamaModel(model=REAL_MODEL, base_url=OLLAMA_URL))
    reg.register(FileTools(workspace))
    reg.register(AgentLoop(max_rounds=rounds))
    reg.start_all()
    return ctx, reg


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_live_model_clean_baseline(tmp_path):
    ctx, reg = _live_agent(tmp_path)
    try:
        r = ctx.plugins["agent_loop"].run("Say hello")
        assert isinstance(r, str) and len(r) > 0
    finally:
        reg.stop_all()


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_live_model_tool_call_roundtrip(tmp_path):
    ctx, reg = _live_agent(tmp_path)
    try:
        r = ctx.plugins["agent_loop"].run("Create a file test.txt with content hello world")
        assert isinstance(r, str) and len(r) > 0
        f = Path(tmp_path) / "workspace" / "test.txt"
        if f.exists():
            assert "hello" in f.read_text(encoding="utf-8").lower()
    finally:
        reg.stop_all()


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
def test_live_model_event_logging(tmp_path):
    ctx, reg = _live_agent(tmp_path)
    try:
        ctx.plugins["agent_loop"].run("Say hello")
        events = ctx.plugins["event_log"].get_session_events(ctx.plugins["continuity"].session_id)
        types = [e.type for e in events]
        assert {"session.start", "user.message", "assistant.message"}.issubset(types)
    finally:
        reg.stop_all()
