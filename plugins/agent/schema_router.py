from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.plugin import Plugin


@dataclass
class LogicalTool:
    """A compact logical tool visible to the model."""
    name: str
    description: str
    args_schema: Dict[str, Any] = field(default_factory=dict)


class SchemaRouter(Plugin):
    """
    Compact-schema router for small models (qwen2.5-coder:1.5b).

    When enabled and compact_schema is True:
    - The model sees a single ``call_tool(tool, args)`` function instead of
      verbose per-tool JSON schemas, saving hundreds of prompt tokens per round.
    - Logical calls (``read``, ``write``, ``list``, ``done``) are expanded
      into real tool calls before execution.
    - Tool results are compressed (truncated) before being added to context.

    When disabled:
    - ``get_model_tools()`` returns ``[]`` (the loop falls through to the
      normal registry schemas).
    - ``expand_call()`` returns ``None`` (no expansion).
    - ``compress_result()`` returns the input unchanged.

    Deterministic and zero-token by design: no model calls, no network.
    """

    name = "schema_router"
    dependencies: tuple[str, ...] = ()
    __contract__: dict[str, Any] = {
        "requires": ("schema_router_enabled", "compact_schema"),
        "provides": ("compact_schema", "expand_call", "compress_result", "get_model_tools"),
        "deterministic": True,
        "zero_token": True,
    }

    # ---- Logical tools the model can see in compact mode ----

    LOGICAL_TOOLS: List[LogicalTool] = [
        LogicalTool(
            name="read",
            description="Read a file from the workspace.",
            args_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        LogicalTool(
            name="write",
            description="Write content to a file.",
            args_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                    "content": {"type": "string", "description": "File content"},
                },
                "required": ["path", "content"],
                "additionalProperties": False,
            },
        ),
        LogicalTool(
            name="list",
            description="List files in a directory.",
            args_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative directory path; default '.'"},
                },
                "required": [],
                "additionalProperties": False,
            },
        ),
        LogicalTool(
            name="delete",
            description="Delete a file from the workspace.",
            args_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path"},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        ),
        LogicalTool(
            name="done",
            description="Signal that the task is complete; respond with text only.",
            args_schema={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        ),
    ]

    # ---- Expansion rules: logical_name -> (real_tool, arg_map, default_args) ----
    # arg_map: logical_arg_name -> real_arg_name
    # default_args: real arg_name -> default value (used when logical args omit it)

    EXPANSIONS: Dict[str, Dict[str, Any]] = {
        "read":   {"real_tool": "read_file",      "arg_map": {"path": "path"}, "defaults": {}},
        "write":  {"real_tool": "write_file",     "arg_map": {"path": "path", "content": "content"}, "defaults": {}},
        "list":   {"real_tool": "list_directory", "arg_map": {"path": "path"}, "defaults": {"path": "."}},
        "delete": {"real_tool": "delete_file",    "arg_map": {"path": "path"}, "defaults": {}},
        "done":   {"real_tool": None,             "arg_map": {}, "defaults": {}},
    }

    # Size threshold for result compression
    _PREVIEW_LIMIT = 200

    def __init__(self, context: Optional[Any] = None) -> None:
        super().__init__()
        self.context = context
        self.enabled = False
        self.compact_mode = False

    def register(self, context: Any) -> None:
        super().register(context)
        cfg = context.config if context is not None else {}
        self.enabled = bool(cfg.get("schema_router_enabled", False))
        self.compact_mode = bool(cfg.get("compact_schema", False))

    def start(self) -> None:
        # Config is read in register() so context.config is already available.
        # No-op start; the router is a pure transformation layer.
        pass

    def health_check(self) -> dict[str, Any]:
        """Verify the router has all required capability methods."""
        return {
            "healthy": self.enabled,
            "enabled": self.enabled,
            "compact_mode": self.compact_mode,
            "has_get_model_tools": hasattr(self, "get_model_tools"),
            "has_expand_call": hasattr(self, "expand_call"),
            "has_compress_result": hasattr(self, "compress_result"),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_model_tools(self) -> List[Dict[str, Any]]:
        """
        Return the tool schemas the model should see.

        If compact mode is enabled, returns a single ``call_tool`` function
        schema. Otherwise returns ``[]`` (loop falls through to registry schemas).
        """
        if not (self.enabled and self.compact_mode):
            return []

        logical_names = [t.name for t in self.LOGICAL_TOOLS]
        description = (
            "call_tool(tool, args). tool ∈ {"
            + ", ".join(logical_names)
            + "}. Format: {\"tool\": \"<name>\", \"args\": {...}}"
        )

        return [
            {
                "type": "function",
                "function": {
                    "name": "call_tool",
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool": {
                                "type": "string",
                                "enum": logical_names,
                                "description": "Logical tool name",
                            },
                            "args": {
                                "type": "object",
                                "description": "Tool arguments as a JSON object",
                            },
                        },
                        "required": ["tool", "args"],
                        "additionalProperties": False,
                    },
                },
            }
        ]

    def expand_call(
        self, call: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Expand a compact-schema tool call (call_tool) into a real tool call.

        Accepts a full OpenAI-format tool call dict as the model emits it:

            {"id": "call_xxx", "type": "function",
             "function": {"name": "call_tool",
                          "arguments": {"tool": "read", "args": {"path": "a.txt"}}}}

        Arguments may come as a dict or a JSON string (both accepted by the
        model depending on the client).

        Returns the expanded call in the same outer format, with the real tool
        name and arguments::

            {"id": "call_xxx", "type": "function",
             "function": {"name": "read_file",
                          "arguments": {"path": "a.txt"}}}

        Returns ``None`` for ``done`` calls (sentinel, no real tool to execute,
        caller should drop the call) and for any non-``call_tool`` call when the
        router is disabled.

        Non-``call_tool`` calls (real tool names) pass through unchanged so
        this method is safe to call on mixed lists of compact and verbose
        calls.
        """
        if not self.enabled:
            return None

        fn = call.get("function", {})
        if not isinstance(fn, dict):
            return None

        name = fn.get("name", "")

        # Passthrough: only expand compact-schema calls
        if name != "call_tool":
            return call

        # Parse arguments (may be a dict or JSON string)
        raw_args = fn.get("arguments")
        if raw_args is None:
            return None
        if isinstance(raw_args, str):
            try:
                raw_args = json.loads(raw_args)
            except (json.JSONDecodeError, TypeError):
                return None
        if not isinstance(raw_args, dict):
            return None

        logical_name = raw_args.get("tool")
        logical_args = raw_args.get("args", {})

        rule = self.EXPANSIONS.get(logical_name)
        if rule is None:
            return None

        real_tool = rule["real_tool"]
        if real_tool is None:
            # "done" — signals completion, no tool execution
            return None

        arg_map = rule["arg_map"]
        defaults = rule["defaults"]

        real_args: Dict[str, Any] = {}
        for logical_arg, real_arg in arg_map.items():
            if logical_arg in logical_args:
                real_args[real_arg] = logical_args[logical_arg]

        # Apply defaults for missing args
        for real_arg, default_val in defaults.items():
            if real_arg not in real_args:
                real_args[real_arg] = default_val

        return {
            "id": call.get("id") or f"call_{logical_name}",
            "type": "function",
            "function": {
                "name": real_tool,
                "arguments": real_args,
            },
        }

    def compress_result(self, result: Any) -> Any:
        """
        Compress a tool result for the model's context window.

        Tool results arrive as strings (raw content from read_file, or JSON
        from list_directory / error dicts). This method:

        - Parses JSON strings and compresses dict fields (truncate ``content``,
          shorten ``error`` to first line).
        - For non-JSON long strings (e.g., large file reads), truncates to a
          preview.
        - Returns the input unchanged when compact_mode is off.
        """
        if not self.compact_mode:
            return result

        # Non-dict results (plain strings) — try JSON, then plain truncation.
        if not isinstance(result, dict):
            if isinstance(result, str):
                # Try parsing as JSON first
                try:
                    parsed = json.loads(result)
                except (json.JSONDecodeError, TypeError):
                    parsed = None

                if isinstance(parsed, dict):
                    compressed = self._compress_dict(parsed)
                    return json.dumps(compressed, ensure_ascii=False)
                elif isinstance(parsed, list):
                    # list_directory results are already compact (just filenames)
                    return result

                # Not JSON — treat as raw text (e.g., read_file output)
                if len(result) > self._PREVIEW_LIMIT:
                    return result[: self._PREVIEW_LIMIT] + "..."
                return result
            return result

        # Already a dict — compress fields directly
        return self._compress_dict(result)

    def _compress_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compress a dict result: truncate content, shorten errors."""
        compressed: Dict[str, Any] = {}
        for key, value in data.items():
            if key == "content" and isinstance(value, str) and len(value) > self._PREVIEW_LIMIT:
                compressed["content_preview"] = value[: self._PREVIEW_LIMIT] + "..."
            elif key == "error" and isinstance(value, str):
                compressed["error"] = value.splitlines()[0] if value else value
            else:
                compressed[key] = value
        if "status" not in compressed:
            compressed["status"] = data.get("status", "unknown")
        return compressed

    def get_logical_tools_description(self) -> str:
        """Short textual description of logical tools (for system prompt if needed)."""
        lines = ["Available tools:"]
        for t in self.LOGICAL_TOOLS:
            args = sorted(t.args_schema.get("properties", {}).keys())
            lines.append(f"- {t.name}({', '.join(args)})")
        return "\n".join(lines)
