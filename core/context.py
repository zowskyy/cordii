from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .errors import CancelledError
from .messages import Message

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model calibration table — single source of truth for per-model numbers.
#
# "As above, so below" split: invariant layers (AgentLoop, ContextPruner,
# RealityProjector) must NOT hardcode model-specific numbers. They read their
# calibration from Context.config["calibration"] (see calibration_from_context).
# This table is where per-model values live; it is seeded from the capacity
# model (scripts/capacity_calculator.py) and re-measured per model via
# `capacity_calculator.py --verify` and the --live benchmark pool.
#
# Unknown models fall back to DEFAULT_PRESET_KEY: the smallest budget is the
# safe direction — it folds more often but can never exceed the model window.
# ---------------------------------------------------------------------------
MODEL_PRESETS: Dict[str, Dict[str, Any]] = {
    # max_tool_result_bytes: hard cap on a single tool result entering the context
    # (~56% of the model window in bytes, at ~3.5 bytes/token) — one result must
    # never be able to swallow the window, regardless of file size on disk.
    "1.5b": {"label": "qwen2.5-coder:1.5b (4k, flaky)", "max_tokens": 4096, "pruner_budget": 3000, "safety": 0.85, "max_messages": 40, "rounds_per_file": 1.3, "max_tool_result_bytes": 8192},
    "7b": {"label": "qwen2.5-coder:7b (8k, stable)", "max_tokens": 8192, "pruner_budget": 6500, "safety": 0.88, "max_messages": 60, "rounds_per_file": 1.05, "max_tool_result_bytes": 16384},
    "14b": {"label": "qwen2.5-coder:14b (16k)", "max_tokens": 16384, "pruner_budget": 14000, "safety": 0.90, "max_messages": 80, "rounds_per_file": 1.02, "max_tool_result_bytes": 32768},
}
DEFAULT_PRESET_KEY = "1.5b"

_MODEL_SIZE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*b$", re.IGNORECASE)


def preset_key_for_model(model_name: str) -> str:
    """Map an Ollama model name (e.g. 'qwen2.5-coder:1.5b') to a preset key.

    Exact key match on the tag first, then trailing size suffix
    ('1.5b', '7b', ...); unknown models fall back to DEFAULT_PRESET_KEY.
    """
    if not model_name:
        return DEFAULT_PRESET_KEY
    candidate = str(model_name).split(":")[-1].strip().lower()
    if candidate in MODEL_PRESETS:
        return candidate
    m = _MODEL_SIZE_RE.search(candidate)
    if m:
        key = m.group(1) + "b"
        if key in MODEL_PRESETS:
            return key
    return DEFAULT_PRESET_KEY


def resolve_calibration(model_name: str = "", explicit: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a calibration dict for a model: preset values + explicit overrides.

    Returns a copy (never the table itself) with a 'preset' key naming the
    preset that was used.
    """
    key = preset_key_for_model(model_name)
    cal: Dict[str, Any] = dict(MODEL_PRESETS[key])
    cal["preset"] = key
    if explicit:
        for k, v in explicit.items():
            if v is not None:
                cal[k] = v
    return cal


def calibration_from_context(context: Optional["Context"]) -> Dict[str, Any]:
    """Read the active calibration for a Context.

    Precedence: config["calibration"] (explicit overrides) > config["model"]
    name resolution > DEFAULT_PRESET_KEY. This is the ONLY place invariant
    layers should source per-model numbers from.
    """
    if context is None:
        return resolve_calibration()
    cfg = context.config or {}
    explicit = cfg.get("calibration")
    if isinstance(explicit, dict):
        return resolve_calibration(str(cfg.get("model", "")), explicit)
    return resolve_calibration(str(cfg.get("model", "")))


@dataclass
class Event:
    type: str
    payload: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self) -> None:
        self._listeners: Dict[str, List[Callable[[Event], None]]] = {}
        self._lock = threading.Lock()

    def on(self, event_type: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            self._listeners.setdefault(event_type, []).append(handler)

    def off(self, event_type: str, handler: Callable[[Event], None]) -> None:
        with self._lock:
            handlers = self._listeners.get(event_type, [])
            try:
                handlers.remove(handler)
            except ValueError:
                pass

    def emit(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        event = Event(type=event_type, payload=payload or {})
        with self._lock:
            handlers = list(self._listeners.get(event_type, []))
        for handler in handlers:
            try:
                handler(event)
            except Exception as exc:
                logger.exception("EventBus handler for %r failed: %s", event_type, exc)


@dataclass
class Context:
    config: Dict[str, Any] = field(default_factory=dict)
    messages: List[Message] = field(default_factory=list)
    plugins: Dict[str, Any] = field(default_factory=dict)
    _cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    events: EventBus = field(default_factory=EventBus)
    prompt_injections: List[Message] = field(default_factory=list)

    def append_message(
        self,
        role: str,
        content: str = "",
        *,
        name: Optional[str] = None,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
    ) -> Message:
        message = Message(
            role=role,
            content=content,
            name=name,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_calls=tool_calls,
        )
        self.messages.append(message)
        return message

    def clear_messages(self) -> None:
        self.messages.clear()

    def config_get(self, key: str, default: Any = None) -> Any:
        if key in self.config:
            return self.config[key]
        return default

    def cancel(self) -> None:
        self._cancel_event.set()

    def reset_cancel(self) -> None:
        self._cancel_event = threading.Event()

    def check_cancelled(self) -> None:
        if self._cancel_event.is_set():
            raise CancelledError("Agent run cancelled.")
