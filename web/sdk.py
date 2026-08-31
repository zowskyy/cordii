from __future__ import annotations

import asyncio
import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from main import build_application


@dataclass
class AgentSession:
    id: str
    workspace: Path
    model: str
    profile: str
    enable_semantic_router: bool
    db_path: Path
    ctx: Any = None
    reg: Any = None
    _event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    _listener_task: Optional[asyncio.Task] = field(default=None, repr=False)

    async def start(self) -> None:
        self.ctx, self.reg = build_application(
            self.workspace,
            self.model,
            "http://127.0.0.1:11434",
            self.db_path,
            profile=self.profile,
            enable_semantic_router=self.enable_semantic_router,
        )
        if not hasattr(self.ctx, "event_queue"):
            self.ctx.event_queue = self._event_queue
        self._listen()

    def _listen(self) -> None:
        def _forward(event: Any) -> None:
            try:
                self._event_queue.put_nowait(event)
            except asyncio.QueueFull:
                pass

        self._handler = _forward
        self.ctx.events.on("*", self._handler)

    async def send_message(self, text: str) -> AsyncIterator[dict]:
        if self.ctx is None:
            raise RuntimeError("Session not started")
        await self.ctx.plugins["agent_loop"].run(text)
        while True:
            event = await self._event_queue.get()
            yield {
                "type": event.type,
                "data": event.payload if hasattr(event, "payload") else {},
            }

    def stop(self) -> None:
        if self.ctx is not None and hasattr(self, "_handler"):
            self.ctx.events.off("*", self._handler)
        if self._listener_task and not self._listener_task.done():
            self._listener_task.cancel()
        if self.reg is not None:
            self.reg.stop_all()


_sessions: Dict[str, AgentSession] = {}


def _session_id() -> str:
    return "sess_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def create_session(
    workspace: str = "workspace",
    model: str = "qwen2.5-coder:1.5b",
    profile: str = "lite",
    enable_semantic_router: bool = False,
    db_path: Optional[str] = None,
) -> AgentSession:
    session_id = _session_id()
    workspace_path = Path(workspace).resolve()
    db = Path(db_path) if db_path else Path("continuity/continuity.db").resolve()
    session = AgentSession(
        id=session_id,
        workspace=workspace_path,
        model=model,
        profile=profile,
        enable_semantic_router=enable_semantic_router,
        db_path=db,
    )
    _sessions[session_id] = session
    return session


def get_session(session_id: str) -> AgentSession:
    if session_id not in _sessions:
        raise KeyError(f"Unknown session: {session_id}")
    return _sessions[session_id]


def list_sessions() -> list[str]:
    return list(_sessions.keys())


def remove_session(session_id: str) -> None:
    session = _sessions.pop(session_id, None)
    if session is not None:
        session.stop()
