from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.messages import Message
from core.summarizer import Summarizer


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
        before_tokens = Summarizer.estimate_tokens(str(messages))

        if len(messages) <= self.max_messages:
            pruned = list(messages)
            strategy = "none"
        else:
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
            strategy = "importance"

        # Token pass — the 4k window is the real constraint: even a FEW
        # messages can exceed the model window when tool results are large.
        # Drop lowest-score messages (protecting the leading system prompt and
        # the two most recent) until within the calibrated token budget.
        # assistant+tool_calls score highest (1.5), so single-pruner
        # preservation holds: they only drop if the budget is unreachable.
        removed = len(messages) - len(pruned)
        token_removed = 0
        while Summarizer.estimate_tokens(str(pruned)) > self.token_budget and len(pruned) > 3:
            candidates = [m for i, m in enumerate(pruned) if i > 0 and i < len(pruned) - 2]
            if not candidates:
                break
            worst = min(candidates, key=lambda m: self._score(m, pruned.index(m), len(pruned), task_state))
            pruned.remove(worst)
            removed += 1
            token_removed += 1
        if token_removed > 0:
            strategy = "token" if strategy == "none" else strategy + "+token"

        return PrunedContext(
            messages=pruned,
            removed_count=removed,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=Summarizer.estimate_tokens(str(pruned)),
            strategy=strategy,
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
