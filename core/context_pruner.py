from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from core.compaction import HybridPruningStrategy, PruningStrategy
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

    def __init__(self, max_messages: int = 40, token_budget: int = 3000, strategy: PruningStrategy | None = None):
        self.max_messages = max_messages
        self.token_budget = token_budget
        self._strategy = strategy or HybridPruningStrategy()

    def prune(self, messages: list[Message], task_state: dict[str, Any] | None = None) -> PrunedContext:
        before_tokens = Summarizer.estimate_tokens(str(messages))

        pruned = self._strategy.prune(
            messages=messages,
            max_messages=self.max_messages,
            token_budget=self.token_budget,
            task_state=task_state,
        )

        removed = len(messages) - len(pruned)
        strategy = getattr(self._strategy, "name", type(self._strategy).__name__) if removed > 0 else "none"

        return PrunedContext(
            messages=pruned,
            removed_count=removed,
            estimated_tokens_before=before_tokens,
            estimated_tokens_after=Summarizer.estimate_tokens(str(pruned)),
            strategy=strategy,
        )
