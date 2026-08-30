from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.messages import Message
from core.plugin import EventDrivenPlugin


@dataclass
class AgentIdentity:
    name: str = "Cordelite Agent"
    role: str = "tool-using assistant"
    purpose: str = "complete tasks by calling tools accurately"


@dataclass
class TaskGold:
    goal: str = ""
    expected_outcome: str = ""
    verification_criteria: str = ""
    success_signal: str = "done"


@dataclass
class WorldSnapshot:
    workspace: str = ""
    files_present: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    tools_available: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    unresolved_subtasks: list[str] = field(default_factory=list)
    last_action: str = ""
    last_result: str = ""
    recent_actions: list[str] = field(default_factory=list)


@dataclass
class TemporalAwareness:
    step: int = 0
    max_steps: int = 12
    start_time: float = field(default_factory=time.time)
    elapsed_s: float = 0.0
    budget_remaining: int = 12


@dataclass
class FactEntry:
    content: str
    timestamp: float
    ttl: float = 3600.0

    @property
    def expired(self) -> bool:
        return time.time() - self.timestamp > self.ttl


class FactRegistry:
    def __init__(self, max_entries: int = 50, default_ttl: float = 3600.0) -> None:
        self._facts: dict[str, FactEntry] = {}
        self.max_entries = max_entries
        self.default_ttl = default_ttl

    def add(self, content: str, ttl: Optional[float] = None) -> None:
        key = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        self._facts[key] = FactEntry(
            content=content,
            timestamp=time.time(),
            ttl=ttl or self.default_ttl,
        )
        self._prune()

    def get_recent(self, n: int = 8) -> list[str]:
        self._prune()
        entries = sorted(self._facts.values(), key=lambda f: f.timestamp, reverse=True)
        return [f.content for f in entries[:n]]

    def clear(self) -> None:
        self._facts.clear()

    def _prune(self) -> None:
        now = time.time()
        self._facts = {
            k: v for k, v in self._facts.items()
            if not v.expired
        }
        if len(self._facts) > self.max_entries:
            sorted_items = sorted(self._facts.items(), key=lambda kv: kv[1].timestamp)
            keep = dict(sorted_items[-self.max_entries:])
            self._facts = keep


class WorldState(EventDrivenPlugin):
    name = "world_state"

    def __init__(
        self,
        identity: Optional[AgentIdentity] = None,
        gold: Optional[TaskGold] = None,
        max_rounds: int = 12,
    ) -> None:
        super().__init__()
        self.identity = identity or AgentIdentity()
        self.gold = gold or TaskGold()
        self.world = WorldSnapshot()
        self.temporal = TemporalAwareness(max_steps=max_rounds)
        self.fact_registry = FactRegistry()
        self._task_goal: str = ""
        self._session_id: str = "default"
        self._last_tool_success: bool = True

    def start(self) -> None:
        self.temporal.start_time = time.time()
        self.temporal.step = 0
        self.temporal.budget_remaining = self.temporal.max_steps
        self.world = WorldSnapshot()
        self.fact_registry.clear()

    def on_turn_start(self, event: Any) -> None:
        payload = event.payload if hasattr(event, "payload") else {}
        user_text = payload.get("user_text", "")
        tools_available = payload.get("tools_available", [])
        self._task_goal = user_text
        self.gold.goal = user_text
        self.world.tools_available = tools_available
        self.world.last_action = "Task received"
        self.world.last_result = "Awaiting first action"

        if self.context is not None:
            self.context.prompt_injections.append(self.snapshot_message())

    def on_tool_result(self, event: Any) -> None:
        payload = event.payload if hasattr(event, "payload") else {}
        tool = payload.get("tool", "")
        args = payload.get("arguments", {})
        result = payload.get("result", "")
        success = payload.get("success", True)
        self._last_tool_success = success
        self.temporal.step += 1
        self.temporal.elapsed_s = round(time.time() - self.temporal.start_time, 2)
        self.temporal.budget_remaining = max(0, self.temporal.max_steps - self.temporal.step)

        action_desc = f"[{self.temporal.step}] {tool}({args}) -> {result[:80]}"
        self.world.recent_actions.append(action_desc)
        if len(self.world.recent_actions) > 20:
            self.world.recent_actions = self.world.recent_actions[-20:]

        self.world.last_action = tool
        self.world.last_result = result
        if tool not in self.world.tools_used:
            self.world.tools_used.append(tool)

        fact = f"{tool} call returned: {result[:120]}"
        self.fact_registry.add(fact)

    def on_turn_end(self, event: Any) -> None:
        payload = event.payload if hasattr(event, "payload") else {}
        final_result = payload.get("final_result", "")
        self.fact_registry.add(f"Task completed: {final_result[:120]}")

    def snapshot_message(self) -> Message:
        parts = []
        parts.append("=" * 50)
        parts.append("WORLD STATE")
        parts.append("=" * 50)
        parts.append("")

        parts.append("[IDENTITY]")
        parts.append(f"  Name: {self.identity.name}")
        parts.append(f"  Role: {self.identity.role}")
        parts.append(f"  Purpose: {self.identity.purpose}")
        parts.append("")

        parts.append("[GOLD / TASK]")
        parts.append(f"  Goal: {self.gold.goal}")
        parts.append(f"  Expected outcome: {self.gold.expected_outcome}")
        parts.append(f"  Verification: {self.gold.verification_criteria}")
        parts.append(f"  Success signal: {self.gold.success_signal}")
        parts.append("")

        parts.append("[TEMPORAL]")
        parts.append(f"  Step: {self.temporal.step} / {self.temporal.max_steps}")
        parts.append(f"  Budget remaining: {self.temporal.budget_remaining}")
        parts.append(f"  Elapsed: {self.temporal.elapsed_s}s")
        parts.append("")

        parts.append("[WORLD]")
        parts.append(f"  Workspace: {self.world.workspace}")
        parts.append(f"  Files present: {', '.join(self.world.files_present) if self.world.files_present else '(none yet)'}")
        parts.append(f"  Files touched: {', '.join(self.world.files_touched) if self.world.files_touched else '(none yet)'}")
        parts.append(f"  Tools available: {', '.join(self.world.tools_available) if self.world.tools_available else '(none)'}")
        parts.append(f"  Tools used: {', '.join(self.world.tools_used) if self.world.tools_used else '(none yet)'}")
        parts.append(f"  Unresolved subtasks: {', '.join(self.world.unresolved_subtasks) if self.world.unresolved_subtasks else '(none)'}")
        parts.append("")

        if self.world.recent_actions:
            parts.append("[RECENT ACTIONS]")
            for action in self.world.recent_actions[-8:]:
                parts.append(f"  {action}")
            parts.append("")

        facts = self.fact_registry.get_recent(8)
        if facts:
            parts.append("[MEMORY / FACTS]")
            for fact in facts:
                parts.append(f"  - {fact}")
            parts.append("")

        parts.append("[INSTRUCTION]")
        parts.append("  You are always operating within this world state.")
        parts.append("  Use tools to progress toward the goal.")
        parts.append(f"  When you have fully achieved the goal, respond with: {self.gold.success_signal}")
        parts.append("  Do not call tools that have already succeeded for the same arguments.")

        content = "\n".join(parts)
        return Message(role="system", content=content)
