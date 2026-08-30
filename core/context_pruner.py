from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.messages import Message


@dataclass
class PrunedContext:
    messages: list[Message]
    removed_count: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    strategy: str


class ContextPruner:
    """Reduce effective KV cache pressure by pruning low-importance context.
    P1 FIX: Unified budget 3000 for 4096 ctx (leaves 1k headroom for 1.5B).
    Previous dual-pruner (Summarizer + Pruner) deleted assistant reasoning.
    Now single pruner is authoritative; Summarizer.fold_messages delegates here.
    """

    def __init__(self, max_messages: int = 40, token_budget: int = 3000):
        self.max_messages = max_messages
        self.token_budget = token_budget

    def prune(self, messages: list[Message], task_state: dict[str, Any] | None = None) -> PrunedContext:
        if len(messages) <= self.max_messages:
            return PrunedContext(
                messages=list(messages),
                removed_count=0,
                estimated_tokens_before=sum(len(str(m)) for m in messages),
                estimated_tokens_after=sum(len(str(m)) for m in messages),
                strategy="none",
            )

        scored = []
        for i, msg in enumerate(messages):
            score = self._score(msg, i, len(messages), task_state)
            scored.append((score, i, msg))

        scored.sort(key=lambda x: x[0])
        keep_count = self.max_messages
        keep_indices = {i for _, i, _ in scored[-keep_count:]}
        keep_indices.add(len(messages) - 1)
        keep_indices.add(len(messages) - 2)

        pruned = [m for i, m in enumerate(messages) if i in keep_indices]
        pruned.sort(key=lambda m: messages.index(m))

        return PrunedContext(
            messages=pruned,
            removed_count=len(messages) - len(pruned),
            estimated_tokens_before=sum(len(str(m)) for m in messages),
            estimated_tokens_after=sum(len(str(m)) for m in pruned),
            strategy="importance",
        )

    def _score(self, msg: Message, index: int, total: int, task_state: dict[str, Any] | None) -> float:
        role = msg.role or ""
        if role == "user":
            return 2.0
        if role == "assistant" and msg.tool_calls:
            return 1.5
        if role == "tool":
            try:
                payload = json.loads(msg.content)
                if payload.get("success"):
                    return 1.0
            except (json.JSONDecodeError, AttributeError):
                pass
            return 0.5
        if role == "system":
            if msg.content and "failed" in msg.content.lower():
                return 0.3
            return 0.8
        recency = index / max(total, 1)
        return 0.1 + recency
