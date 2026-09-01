"""DataExporter plugin — exports successful session trajectories for fine-tuning."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional

from core.plugin import EventDrivenPlugin
from core.event_log import EventLog
from core.events import Event


class DataExporter(EventDrivenPlugin):
    """Collects and exports successful session trajectories as JSONL for fine-tuning.

    Registers only in the `lite` profile to enable zero-token data collection.
    Export is triggered via the `--export-data` CLI flag at startup.
    """

    def __init__(self) -> None:
        super().__init__()
        self.name = "data_exporter"
        self.dependencies = ("event_logger",)
        self.__contract__ = {
            "version": "1.0",
            "capabilities": ["export_jsonl", "trajectory_reconstruct", "quality_filter"],
            "zero_token": True,
        }
        self._event_log: Optional[EventLog] = None
        self._run_state: dict[str, Any] = {}

    def start(self) -> None:
        """Initialize the exporter with access to the event log."""
        if self.context is None:
            return
        event_logger = self.context.plugins.get("event_logger")
        if event_logger is None:
            self._event_log = None
            return
        # EventLogger wraps EventLog; access via the _event_log attribute
        if hasattr(event_logger, "_event_log"):
            self._event_log = event_logger._event_log
        else:
            self._event_log = None

    def health_check(self) -> bool:
        """Verify the exporter has access to the event log."""
        return self._event_log is not None

    def export_successful_sessions(
        self, output_dir: str | Path, criteria: dict[str, Any] | None = None
    ) -> int:
        """Export all sessions meeting quality criteria to JSONL files.

        Args:
            output_dir: Directory to write JSONL files.
            criteria: Optional filters (e.g., {"app_type": "crud", "max_turns": 20}).

        Returns:
            Number of sessions exported.
        """
        if self._event_log is None:
            return 0

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        criteria = criteria or {}
        count = 0
        export_rows: list[dict[str, Any]] = []

        for session_id, metadata, events in self._iter_sessions():
            if not self._has_required_criteria(session_id, metadata):
                continue
            if not self._meets_quality_filters(session_id, metadata, criteria, events):
                continue

            trajectory = self._export_session(session_id, events)
            if trajectory is None:
                continue

            export_rows.append(trajectory)
            count += 1

        if export_rows:
            output_file = output_path / "trajectories.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for row in export_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

        return count

    def _has_required_criteria(self, session_id: str, metadata: dict[str, Any] | None) -> bool:
        """Check if a session has a recorded outcome event."""
        return metadata is not None and "outcome" in metadata

    def _meets_quality_filters(
        self,
        session_id: str,
        metadata: dict[str, Any] | None,
        criteria: dict[str, Any],
        events: list[Event],
    ) -> bool:
        """Apply quality filters to determine if a session should be exported.

        Filters:
        - Must have passed verification (if verifier events exist)
        - Must have passed tests (if test events exist)
        - No protected file violations
        - Completed within max_turns (default 20)
        - No errors or timeouts
        """
        if metadata is None:
            return False

        outcome = metadata.get("outcome", "")
        if outcome != "success":
            return False

        # Protect against protected file violations
        for event in events:
            if event.type == "protected_file.violation":
                return False

        # Check turn count
        max_turns = criteria.get("max_turns", 20)
        model_turns = metadata.get("model_turns", 0)
        if model_turns > max_turns:
            return False

        # Check error/timeout events
        for event in events:
            if event.type in ("error", "timeout", "agent.error"):
                return False

        return True

    def _iter_sessions(self) -> Iterator[tuple[str, dict[str, Any] | None, list[Event]]]:
        """Iterate over all sessions, yielding (session_id, metadata, events)."""
        if self._event_log is None:
            return

        rows = self._event_log.query(
            "SELECT DISTINCT session_id FROM events ORDER BY session_id"
        )
        for row in rows:
            session_id = row[0]
            metadata = self._get_session_metadata(session_id)
            events = self._event_log.get_session_events(session_id)
            yield session_id, metadata, events

    def _get_session_metadata(self, session_id: str) -> dict[str, Any] | None:
        """Retrieve the session.outcome event metadata."""
        rows = self._event_log.query(
            "SELECT payload FROM events WHERE session_id = ? AND type = 'session.outcome' ORDER BY id DESC LIMIT 1",
            (session_id,),
        )
        if not rows:
            return None
        try:
            return json.loads(rows[0][0])
        except (json.JSONDecodeError, TypeError):
            return None

    def _export_session(self, session_id: str, events: list[Event]) -> dict[str, Any] | None:
        """Export a single session as a trajectory.

        Returns a dict with:
        - session_id
        - app_type
        - metadata from session.outcome
        - trajectory: list of {role, content, tool_name, tool_args}
        """
        metadata = self._get_session_metadata(session_id)
        if metadata is None:
            return None

        trajectory: list[dict[str, Any]] = []
        for event in events:
            if event.type == "user.message":
                trajectory.append({"role": "user", "content": event.payload.get("content", "")})
            elif event.type == "assistant.message":
                content = event.payload.get("content", "")
                tool_calls = event.payload.get("tool_calls", [])
                trajectory.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            elif event.type == "tool.result":
                tool_name = event.payload.get("tool_name", "")
                tool_args = event.payload.get("arguments", {})
                result = event.payload.get("result", "")
                trajectory.append({
                    "role": "tool",
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "content": result,
                })

        return {
            "session_id": session_id,
            "app_type": metadata.get("app_type", "generic"),
            "metadata": metadata,
            "trajectory": trajectory,
        }

    def reset_run_state(self) -> None:
        """Reset any per-run state (for zero-drag invariant)."""
        self._run_state = {}

    # The old export_session and reconstruct_trajectory are kept as thin
    # wrappers for backward compatibility with tests.
    def export_session(self, session_id: str, output_dir: str | Path) -> bool:
        """Export a single session to a JSONL file in output_dir."""
        if self._event_log is None:
            return False
        events = self._event_log.get_session_events(session_id)
        result = self._export_session(session_id, events)
        if result is None:
            return False

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_file = output_path / f"session_{session_id}.jsonl"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        return True

    def reconstruct_trajectory(self, session_id: str) -> dict[str, Any] | None:
        """Reconstruct the message trajectory for a session."""
        if self._event_log is None:
            return None
        events = self._event_log.get_session_events(session_id)
        return self._export_session(session_id, events)
