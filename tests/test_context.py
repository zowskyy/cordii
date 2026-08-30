import threading

from core.context import Context
from core.errors import CancelledError


def test_message_roundtrip_with_tool_calls():
    ctx = Context()
    calls = [{"type": "function", "function": {"name": "read_file", "arguments": {"path": "a.txt"}}}]
    msg = ctx.append_message("assistant", "reading", tool_calls=calls)
    assert msg.to_dict()["tool_calls"] == calls


def test_context_append_preserves_tool_calls():
    ctx = Context()
    calls = [{"function": {"name": "read_file", "arguments": {}}}]
    ctx.append_message("assistant", "", tool_calls=calls)
    assert ctx.messages[0].tool_calls == calls


def test_context_clear_removes_all_messages():
    ctx = Context()
    ctx.append_message("user", "one")
    ctx.append_message("assistant", "two")
    ctx.clear_messages()
    assert ctx.messages == []


def test_cancel_event_scoped_to_run():
    ctx = Context()
    ctx.cancel()
    try:
        ctx.check_cancelled()
    except CancelledError:
        pass
    else:
        raise AssertionError("cancel should be set")

    ctx.reset_cancel()
    ctx.check_cancelled()
