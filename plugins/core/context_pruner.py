from __future__ import annotations

from core.context_pruner import ContextPruner
from core.plugin import Plugin


class ContextPrunerPlugin(Plugin):
    name = "context_pruner"
    dependencies = ()

    def __init__(self, max_messages: int = 40, token_budget: int = 4000) -> None:
        super().__init__()
        self.max_messages = max_messages
        self.token_budget = token_budget
        self._pruner = ContextPruner(max_messages=max_messages, token_budget=token_budget)

    def start(self) -> None:
        pass

    def prune(self, messages, task_state=None):
        return self._pruner.prune(messages, task_state)
