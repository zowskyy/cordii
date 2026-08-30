from __future__ import annotations

import json
import tempfile
from pathlib import Path

from core.event_log import EventLog
from core.events import Event
from core.reality import RealityProjector, CurrentReality


def test_snapshot_save_and_load():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        with EventLog(db) as log:
            projector = RealityProjector(log, snapshot_threshold=2)
            for i in range(4):
                log.append(Event(type="user.message", session_id="s1", payload={"content": f"m{i}"}))

            reality = projector.get_reality("s1")
            assert len(reality.messages) == 4

            snapshot = log.load_snapshot("s1")
            assert snapshot is not None
            version, state = snapshot
            assert version >= 2
            assert len(state["messages"]) == 4


def test_snapshot_compresses_state():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        with EventLog(db) as log:
            log.save_snapshot("s1", 1, {"messages": ["a" * 1000]}, compress=True)
            snapshot = log.load_snapshot("s1")
            assert snapshot is not None


def test_snapshot_idempotent_on_replay():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "test.db"
        with EventLog(db) as log:
            projector = RealityProjector(log, snapshot_threshold=2)
            for i in range(4):
                log.append(Event(type="user.message", session_id="s1", payload={"content": f"m{i}"}))

            first = projector.get_reality("s1")
            second = projector.get_reality("s1")
            assert len(first.messages) == len(second.messages) == 4


def test_reality_apply_event_tracks_files():
    reality = CurrentReality()
    event = Event(
        type="tool.result",
        session_id="s1",
        payload={
            "tool_name": "file_write",
            "arguments": {"path": "a.txt"},
            "success": True,
        },
    )
    reality.apply_event(event)
    assert "a.txt" in reality.files_written
    assert "file_write" in reality.tools_used


def test_reality_summary_shape():
    reality = CurrentReality()
    reality.apply_event(Event(type="user.message", session_id="s1", payload={"content": "hi"}))
    summary = reality.summary()
    assert summary["message_count"] == 1
    assert summary["error_count"] == 0
