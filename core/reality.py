from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .events import Event, USER_MESSAGE, ASSISTANT_MESSAGE, TOOL_RESULT, TOOL_ERROR, TASK_START


@dataclass
class CurrentReality:
    messages: list[dict[str, Any]] = field(default_factory=list)
    files_read: set[str] = field(default_factory=set)
    files_written: set[str] = field(default_factory=set)
    files_listed: set[str] = field(default_factory=set)
    tools_used: list[str] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    task_ids: set[str] = field(default_factory=set)
    last_event_id: int = 0
    last_timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "files_read": list(self.files_read), "files_written": list(self.files_written), "files_listed": list(self.files_listed), "task_ids": list(self.task_ids)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CurrentReality:
        return cls(
            messages=data.get("messages", []),
            files_read=set(data.get("files_read", [])),
            files_written=set(data.get("files_written", [])),
            files_listed=set(data.get("files_listed", [])),
            tools_used=data.get("tools_used", []),
            errors=data.get("errors", []),
            task_ids=set(data.get("task_ids", [])),
            last_event_id=data.get("last_event_id", 0),
            last_timestamp=data.get("last_timestamp"),
        )

    def apply_event(self, event: Event) -> None:
        if event.id is not None:
            self.last_event_id = max(self.last_event_id, event.id)
        self.last_timestamp = event.timestamp
        if event.task_id:
            self.task_ids.add(event.task_id)
        payload = event.payload or {}
        if event.type == USER_MESSAGE:
            self.messages.append({"role": "user", "content": payload.get("content", "")})
        elif event.type == ASSISTANT_MESSAGE:
            self.messages.append({"role": "assistant", "content": payload.get("content", ""), "tool_calls": payload.get("tool_calls")})
        elif event.type == TOOL_RESULT:
            tool_name = payload.get("tool_name", "")
            self.tools_used.append(tool_name)
            self.messages.append({"role": "tool", "content": payload.get("content", ""), "tool_name": tool_name})
            path = payload.get("arguments", {}).get("path", "")
            if tool_name == "file_read" and path:
                self.files_read.add(path)
            elif tool_name == "file_write" and path:
                self.files_written.add(path)
            elif tool_name == "file_list":
                self.files_listed.add(path or ".")
            if payload.get("success") is False:
                self.errors.append({"tool": tool_name, "error": payload.get("error", "Unknown error"), "timestamp": event.timestamp})
        elif event.type == TOOL_ERROR:
            self.errors.append({"tool": payload.get("tool_name", "unknown"), "error": payload.get("error", "Unknown error"), "timestamp": event.timestamp})
        elif event.type == TASK_START and event.task_id:
            self.task_ids.add(event.task_id)

    def summary(self) -> dict[str, Any]:
        return {"message_count": len(self.messages), "files_read": sorted(self.files_read), "files_written": sorted(self.files_written), "files_listed": sorted(self.files_listed), "tools_used": self.tools_used, "error_count": len(self.errors), "tasks": sorted(self.task_ids), "last_activity": self.last_timestamp}


class RealityProjector:
    def __init__(self, event_log: Any, snapshot_threshold: int = 50) -> None:
        self._event_log = event_log
        self._snapshot_threshold = snapshot_threshold
        self._cache: dict[str, CurrentReality] = {}

    def get_reality(self, session_id: str) -> CurrentReality:
        if session_id in self._cache:
            return self._cache[session_id]
        snapshot = self._event_log.load_snapshot(session_id)
        if snapshot:
            version, state_dict = snapshot
            reality = CurrentReality.from_dict(state_dict)
            events = self._event_log.get_events_after(session_id, version)
        else:
            reality = CurrentReality()
            events = self._event_log.get_session_events(session_id)
        for event in events:
            reality.apply_event(event)
        if len(events) >= self._snapshot_threshold:
            self._event_log.save_snapshot(session_id, reality.last_event_id, reality.to_dict())
        self._cache[session_id] = reality
        return reality

    def invalidate_cache(self, session_id: str | None = None) -> None:
        if session_id:
            self._cache.pop(session_id, None)
        else:
            self._cache.clear()
