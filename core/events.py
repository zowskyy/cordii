from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# Event type constants
SESSION_START = "session.start"
SESSION_END = "session.end"
USER_MESSAGE = "user.message"
ASSISTANT_MESSAGE = "assistant.message"
TOOL_INVOKED = "tool.invoked"
TOOL_RESULT = "tool.result"
TOOL_ERROR = "tool.error"
TASK_START = "task.start"
TASK_END = "task.end"
GEN_START = "gen.start"
GEN_COMPLETE = "gen.complete"
GEN_SENT = "gen.sent"
STEP_TRACE = "step.trace"


@dataclass
class Event:
    type: str
    session_id: str
    payload: dict[str, Any]
    id: int | None = None
    timestamp: str = field(default_factory=lambda: _utc_now())
    task_id: str | None = None
    parent_event_id: int | None = None
    operation_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "type": self.type,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "parent_event_id": self.parent_event_id,
            "operation_id": self.operation_id,
            "payload": self.payload,
        }

    @classmethod
    def from_row(cls, row: tuple) -> Event:
        return cls(
            id=row[0],
            timestamp=row[1],
            type=row[2],
            session_id=row[3],
            task_id=row[4],
            parent_event_id=row[5],
            operation_id=row[6],
            payload=json.loads(row[7]) if row[7] else {},
        )


@dataclass
class TraceStep:
    step_id: str
    session_id: str
    tool_name: str
    input: dict[str, Any]
    output: str = ""
    duration_ms: float = 0.0
    error_type: str | None = None
    token_cost: int = 0
    context_size_before: int = 0
    context_size_after: int = 0
    dependency_ids: list[str] = field(default_factory=list)
    parallelizable: bool = False
    governance_check_passed: bool = True
    timestamp: str = field(default_factory=lambda: _utc_now())


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
