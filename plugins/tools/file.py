from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.errors import ToolError, WorkspaceError
from core.plugin import Plugin


class FileTools(Plugin):
    name = "file_tools"

    # 3x deterministic templates — reuse via TEMPLATE: prefix, LLM sends short token, FileTools expands deterministically (pre-existing system)
    TEMPLATES: dict[str, str] = {
        "todo:index.html": '<!DOCTYPE html><html><head><link rel="stylesheet" href="style.css"></head><body><div id="app"><h1>Todo</h1><input id="input" placeholder="todo"><button onclick="add()">Add</button><ul id="list"></ul></div><script src="app.js"></script></body></html>',
        "todo:app.js": 'let todos=[];function add(){let v=document.getElementById("input").value.trim();if(!v)return;todos.push(v);render();document.getElementById("input").value=""}function render(){let l=document.getElementById("list");l.innerHTML=todos.map((t,i)=>`<li>${t} <button onclick="todos.splice(${i},1);render()">x</button></li>`).join("")}',
        "todo:style.css": 'body{font-family:sans-serif;background:#f5f5f5}#app{max-width:500px;margin:40px auto;background:white;padding:20px;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.1)}#input{width:70%;padding:8px}button{padding:8px 12px;margin-left:4px}',
        "todo:crud.js": 'const store={items:[],add(t){this.items.push({id:Date.now(),text:t})},remove(id){this.items=this.items.filter(i=>i.id!==id)},list(){return this.items}};if(typeof module!=="undefined")module.exports=store;',
        "landing:index.html": '<!DOCTYPE html><html><head><link rel="stylesheet" href="style.css"></head><body><header><h1>Landing</h1><p>Welcome</p><button>Get Started</button></header><script src="app.js"></script></body></html>',
        "landing:app.js": 'document.querySelector("button").addEventListener("click",()=>alert("started"))',
        "landing:style.css": 'header{text-align:center;padding:60px;background:#667eea;color:white}button{padding:12px 24px;border:none;border-radius:4px;background:white;color:#667eea}',
    }

    def __init__(self, workspace: str | Path, *, max_read_bytes: int = 2 * 1024 * 1024, max_write_bytes: int = 2 * 1024 * 1024) -> None:
        super().__init__()
        self.workspace = Path(workspace).expanduser().resolve()
        self.max_read_bytes = max_read_bytes
        self.max_write_bytes = max_write_bytes

    def start(self) -> None:
        self.workspace.mkdir(parents=True, exist_ok=True)

    def _resolve(self, user_path: str) -> Path:
        if not isinstance(user_path, str) or not user_path.strip():
            raise WorkspaceError("Path must be a non-empty string.")
        candidate = Path(user_path.strip().replace("\\", "/"))
        if candidate.is_absolute():
            raise WorkspaceError("Absolute paths are not allowed.")
        try:
            resolved = (self.workspace / candidate).resolve()
            resolved.relative_to(self.workspace)
        except ValueError as exc:
            raise WorkspaceError(f"Path escapes workspace: {user_path!r}") from exc
        return resolved

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        if not target.is_file():
            raise ToolError(f"File does not exist: {path}")
        size = target.stat().st_size
        if size > self.max_read_bytes:
            raise ToolError(f"File is too large ({size} bytes); limit is {self.max_read_bytes}.")
        try:
            return target.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ToolError(f"File is not valid UTF-8: {path}") from exc
        except OSError as exc:
            raise ToolError(f"Could not read {path}: {exc}") from exc

    def write_file(self, path: str, content: str) -> str:
        if not isinstance(content, str):
            raise ToolError("content must be a string.")
        # 3x: TEMPLATE: expansion — LLM sends short token (e.g., TEMPLATE:todo:index.html), FileTools expands deterministically
        if content.strip().startswith("TEMPLATE:"):
            key = content.strip()[9:].strip()  # after "TEMPLATE:"
            # support "todo:index.html" or "todo/app.js" -> normalize
            key = key.replace("/", ":").replace("\\", ":")
            expanded = self.TEMPLATES.get(key)
            if expanded is None:
                # fallback: try path-based lookup
                # e.g., write_file path=index.html with content TEMPLATE:todo -> map to todo:index.html
                alt = f"todo:{Path(path).name}"
                expanded = self.TEMPLATES.get(alt)
            if expanded is not None:
                content = expanded
            else:
                raise ToolError(f"Unknown template: {key!r}. Available: {', '.join(self.TEMPLATES)}")
        if len(content.encode("utf-8")) > self.max_write_bytes:
            raise ToolError("Content exceeds write size limit.")
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            target.write_text(content, encoding="utf-8", newline="")
        except OSError as exc:
            raise ToolError(f"Could not write {path}: {exc}") from exc
        return f"Wrote {len(content.encode('utf-8'))} bytes to {path}"

    def list_directory(self, path: str = ".") -> list[str]:
        target = self._resolve(path)
        if not target.is_dir():
            raise ToolError(f"Directory does not exist: {path}")
        try:
            return sorted(item.name for item in target.iterdir())
        except OSError as exc:
            raise ToolError(f"Could not list {path}: {exc}") from exc

    def read_json(self, path: str) -> Any:
        try:
            return json.loads(self.read_file(path))
        except json.JSONDecodeError as exc:
            raise ToolError(f"Invalid JSON in {path}: {exc}") from exc

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {"type": "function", "function": {"name": "read_file", "description": "Read a UTF-8 text file inside the workspace.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}}}}},
            {"type": "function", "function": {"name": "write_file", "description": "Write UTF-8 text to a workspace-relative file.", "parameters": {"type": "object", "required": ["path", "content"], "properties": {"path": {"type": "string", "description": "Workspace-relative file path."}, "content": {"type": "string", "description": "Complete UTF-8 file contents."}}}}},
            {"type": "function", "function": {"name": "list_directory", "description": "List entries in a workspace-relative directory.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Workspace-relative directory path; defaults to '.'."}}}}},
            {"type": "function", "function": {"name": "read_json", "description": "Read and parse a JSON file inside the workspace.", "parameters": {"type": "object", "required": ["path"], "properties": {"path": {"type": "string", "description": "Workspace-relative JSON file path."}}}}},
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if not isinstance(arguments, dict):
            raise ToolError("Tool arguments must be a JSON object.")
        if name == "read_file":
            return self.read_file(str(arguments["path"]))
        if name == "write_file":
            return self.write_file(str(arguments["path"]), str(arguments["content"]))
        if name == "list_directory":
            return json.dumps(self.list_directory(str(arguments.get("path", "."))))
        if name == "read_json":
            return json.dumps(self.read_json(str(arguments["path"])), ensure_ascii=False, indent=2)
        raise ToolError(f"Unknown file tool: {name}")
