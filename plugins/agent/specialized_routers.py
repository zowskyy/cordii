from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Optional


class ToolResultVerifier:
    @staticmethod
    def verify(tool_name: str, arguments: dict[str, Any], result: str) -> bool:
        try:
            data = json.loads(result)
            if data.get("blocked"):
                return False
            if data.get("success") is False:
                return False
            if tool_name == "read_file" and "WRONG" in result:
                return False
        except (json.JSONDecodeError, AttributeError):
            pass
        return True


class RepairMessageBuilder:
    @staticmethod
    def build(tool_name: str, failure_type: Any, healing: dict[str, Any]) -> str:
        action = healing.get("action", "abstain")
        if action == "replan":
            return json.dumps({"role": "system", "content": f"Tool '{tool_name}' failed ({failure_type.value}). Reconsider the plan and try a different approach."}, ensure_ascii=False)
        if action == "cross_check":
            return json.dumps({"role": "system", "content": f"Tool '{tool_name}' output needs verification. Check against an alternative source or re-read the file."}, ensure_ascii=False)
        if action == "escalate":
            return json.dumps({"role": "system", "content": f"Cannot recover from '{tool_name}' failure ({failure_type.value}). Escalating to user."}, ensure_ascii=False)
        return json.dumps({"role": "system", "content": f"Tool '{tool_name}' is blocked after repeated failures ({failure_type.value}). Try a different approach or tool."}, ensure_ascii=False)

    @staticmethod
    def global_replan(failed_tools: list[str]) -> str:
        summary = f"Multiple tools failed: {', '.join(failed_tools)}. Reconsider the remaining plan and try a different approach."
        return json.dumps({"role": "system", "content": summary}, ensure_ascii=False)


class SpecializedRouters:
    def __init__(
        self,
        tool_handlers: dict[str, Callable[[dict[str, Any]], str]],
        context: Any,
        record_tool_result: Callable[[str, dict, str, bool], None],
        resolve_path: Callable[[str], Optional[Path]],
    ) -> None:
        self._tool_handlers = tool_handlers
        self._context = context
        self._record_tool_result = record_tool_result
        self._resolve_path = resolve_path

    def try_zero_thought(self, user_text: str) -> Optional[str]:
        text = user_text.strip()

        read_match = re.match(r"^read\s+([^\s]+)$", text, re.IGNORECASE)
        if read_match:
            path = read_match.group(1)
            handler = self._tool_handlers.get("read_file")
            if handler is None:
                return None
            target = self._resolve_path(path)
            if target is None or not target.exists():
                return None
            try:
                result = handler("read_file", {"path": path})
                self._record_tool_result("read_file", {"path": path}, str(result), True)
                return str(result)
            except Exception as exc:
                error_result = json.dumps({"error": str(exc), "tool": "read_file", "arguments": {"path": path}})
                self._record_tool_result("read_file", {"path": path}, error_result, False)
                return error_result

        write_match = re.match(r"^write\s+(.+?)\s+to\s+([^\s]+)$", text, re.IGNORECASE)
        if write_match:
            content = write_match.group(1)
            path = write_match.group(2)
            handler = self._tool_handlers.get("write_file")
            if handler is None:
                return None
            try:
                result = handler("write_file", {"path": path, "content": content})
                self._record_tool_result("write_file", {"path": path, "content": content}, str(result), True)
                return "done"
            except Exception as exc:
                error_result = json.dumps({"error": str(exc), "tool": "write_file", "arguments": {"path": path, "content": content}})
                self._record_tool_result("write_file", {"path": path, "content": content}, error_result, False)
                return error_result

        if re.match(r"^list\s+files?$", text, re.IGNORECASE):
            handler = self._tool_handlers.get("list_directory")
            if handler is None:
                return None
            try:
                result = handler("list_directory", {"path": "."})
                self._record_tool_result("list_directory", {"path": "."}, str(result), True)
                return str(result)
            except Exception as exc:
                error_result = json.dumps({"error": str(exc), "tool": "list_directory", "arguments": {"path": "."}})
                self._record_tool_result("list_directory", {"path": "."}, error_result, False)
                return error_result

        return None
