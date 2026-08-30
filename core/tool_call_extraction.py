from __future__ import annotations

import hashlib
import json
import re
from typing import Any


def _stable_id(text: str) -> str:
    return f"{hashlib.md5(text.encode()).hexdigest()[:8]}"


def extract_tool_calls_from_text(text: str, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    tool_names = [t.get("function", {}).get("name", "") for t in tools if isinstance(t, dict)]
    if not tool_names:
        return []

    calls: list[dict[str, Any]] = []

    cleaned = text.strip()
    cleaned = re.sub(r'```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'```', '', cleaned).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "tool_calls" in parsed:
            for tc in parsed["tool_calls"]:
                if isinstance(tc, dict) and tc.get("function", {}).get("name") in tool_names:
                    calls.append(tc)
            if calls:
                return calls[:5]
    except (json.JSONDecodeError, ValueError):
        pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(cleaned):
        start = cleaned.find('{', idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(cleaned, start)
            if isinstance(obj, dict) and "tool_calls" in obj:
                for tc in obj["tool_calls"]:
                    if isinstance(tc, dict) and tc.get("function", {}).get("name") in tool_names:
                        calls.append(tc)
                if calls:
                    return calls[:5]
            idx = end
        except (json.JSONDecodeError, ValueError):
            idx = start + 1

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            name = parsed.get("name", "")
            if name in tool_names:
                calls.append({
                    "id": f"call_{_stable_id(cleaned)}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": parsed.get("arguments", {}),
                    }
                })
                return calls[:5]
    except (json.JSONDecodeError, ValueError):
        pass

    for name in tool_names:
        if not name:
            continue
        patterns = [
            re.compile(rf'{re.escape(name)}\s*\(\s*(\{{.*?\}})\s*\)', re.DOTALL | re.IGNORECASE),
            re.compile(rf'{re.escape(name)}\s*\(\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^)]*)\s*\)', re.IGNORECASE),
            re.compile(rf'"{re.escape(name)}"\s*\(\s*(\{{.*?\}})\s*\)', re.DOTALL | re.IGNORECASE),
            re.compile(r'"' + re.escape(name) + r'"\s*:\s*(\{.*?\})', re.DOTALL | re.IGNORECASE),
        ]
        for pattern in patterns:
            for match in pattern.finditer(text):
                args_str = match.group(1).strip()
                args = _parse_args_str(args_str)
                calls.append({
                    "id": f"call_{_stable_id(match.group(0))}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": args,
                    }
                })

    return calls[:5]


def _parse_args_str(args_str: str) -> dict[str, Any]:
    args_str = args_str.strip()
    if not args_str:
        return {}
    if args_str.startswith("{"):
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            pass
    if args_str.startswith('"') and args_str.endswith('"'):
        try:
            return json.loads(args_str)
        except json.JSONDecodeError:
            return {"input": args_str.strip('"')}
    return {"input": args_str}
