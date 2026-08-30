from __future__ import annotations

import re
from typing import Any

from .events import Event


class Summarizer:
    def __init__(self, model: Any = None) -> None:
        self._model = model

    def summarize_events(self, events: list[Event], max_length: int = 500) -> str:
        if not events:
            return ""
        if self._model is None:
            return self._heuristic_summary(events, max_length)
        text = self._format_events_for_summary(events)
        prompt = (
            "Summarize these agent events concisely, focusing on:\n"
            "- What the user asked\n"
            "- What tools were used\n"
            "- Key results or errors\n"
            "- Current state\n\n"
            f"Events:\n{text}\n\n"
            f"Summary (max {max_length} chars):"
        )
        return self._model.generate(prompt)

    def _heuristic_summary(self, events: list[Event], max_length: int) -> str:
        parts = []
        for event in events[-10:]:
            payload = event.payload or {}
            if event.type == "user.message":
                parts.append(f"User: {payload.get('content', '')[:100]}")
            elif event.type == "tool.result":
                tool = payload.get("tool_name", "unknown")
                if payload.get("success"):
                    parts.append(f"Tool {tool} succeeded")
                else:
                    parts.append(f"Tool {tool} failed: {payload.get('error', '')[:50]}")
        summary = "; ".join(parts)
        return summary[:max_length]

    def _format_events_for_summary(self, events: list[Event]) -> str:
        lines = []
        for event in events[-20:]:
            payload = event.payload or {}
            if event.type == "user.message":
                lines.append(f"[{event.timestamp}] User: {payload.get('content', '')}")
            elif event.type == "assistant.message":
                lines.append(f"[{event.timestamp}] Assistant: {payload.get('content', '')[:200]}")
            elif event.type == "tool.result":
                tool = payload.get("tool_name", "unknown")
                status = "ok" if payload.get("success") else f"error: {payload.get('error', '')[:100]}"
                lines.append(f"[{event.timestamp}] Tool {tool}: {status}")
        return "\n".join(lines)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def fold_messages(messages: list[Any], max_messages: int = 40, task_state: dict[str, Any] | None = None) -> list[Any]:
        # P1 FIX: Unified pruner preserves assistant+tool_calls (critical for 1.5B coherence)
        # Previous version dropped assistant reasoning which breaks small model's few-shot memory
        if len(messages) <= max_messages:
            return messages
        task_state = task_state or {}
        user_goal = task_state.get("goal", "")
        files_touched = task_state.get("files_touched", [])
        tools_used = task_state.get("tools_used", [])
        unresolved = task_state.get("unresolved_subtasks", [])
        ledger_parts = [f"Task: {user_goal}"]
        if files_touched:
            ledger_parts.append(f"Files: {', '.join(files_touched)}")
        if tools_used:
            ledger_parts.append(f"Tools used: {', '.join(tools_used)}")
        if unresolved:
            ledger_parts.append(f"Remaining: {', '.join(unresolved)}")
        ledger = " | ".join(ledger_parts)
        # Preserve: user + system + assistant (especially with tool_calls) + tool
        keep_users = [m for m in messages if getattr(m, "role", None) == "user"]
        keep_systems = [m for m in messages if getattr(m, "role", None) == "system"]
        keep_assistants = [m for m in messages if getattr(m, "role", None) == "assistant"]
        kept_tools = [m for m in messages if getattr(m, "role", None) == "tool"]
        # Always keep last 2 messages (usually assistant tool_call + tool result) plus all users and assistants with tool_calls
        essential_assistants = [m for m in keep_assistants if getattr(m, "tool_calls", None)]
        # Budget: keep ledger + users + essential_assistants + recent tools, trim oldest systems first
        remaining = max_messages - 1 - len(keep_users) - len(essential_assistants)
        if remaining < 0:
            essential_assistants = essential_assistants[remaining:]
            remaining = 0
        # Keep recent tools within budget
        kept_tools = kept_tools[-remaining:] if remaining > 0 else []
        # Keep recent systems that are not old ledger-like (prefer recent)
        keep_systems = keep_systems[-(max_messages - len(keep_users) - len(essential_assistants) - len(kept_tools) - 1):]
        from core.messages import Message
        summary_msg = Message(role="system", content=f"[Context folded] {ledger}")
        # Reassemble in original order
        candidates = set(id(m) for m in [summary_msg] + keep_users + keep_systems + essential_assistants + kept_tools)
        # Preserve original ordering but ensure summary first, and keep last assistants that had no tool_calls if space
        ordered = [summary_msg]
        for m in messages:
            if id(m) in candidates and m not in ordered:
                ordered.append(m)
        # If we still have space, add most recent non-essential assistants
        if len(ordered) < max_messages:
            for m in reversed(keep_assistants):
                if m not in ordered and len(ordered) < max_messages:
                    ordered.insert(1, m)
        ordered.sort(key=lambda m: messages.index(m) if m in messages else -1)
        if ordered[0] != summary_msg:
            ordered = [summary_msg] + [m for m in ordered if m != summary_msg]
        return ordered[:max_messages]
