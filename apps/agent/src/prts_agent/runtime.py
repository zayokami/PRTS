"""把 ``prts.runtime.RuntimeBridge`` 协议跑通到 Agent 内部能力。

skill / task 内的 ``prts.client.notify(...)`` / ``prts.workspace.read(...)``
最终走到这里。``notify`` 通过一个 contextvar 维护"当前会话的 SSE 事件队列"
—— Agent loop 启动时把队列 set 进去,工具调用结束 reset。这样多个并发会话
互不干扰。
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from .llm import ChatMessage, LlmClient
from .memory import SqliteStore
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

_active_notify_queue: ContextVar[asyncio.Queue[dict[str, Any]] | None] = ContextVar(
    "prts_notify_queue", default=None
)


def push_notify(message: str, *, kind: str = "info", payload: dict[str, Any] | None = None) -> bool:
    """把一条 notify 事件塞进当前会话的队列。无活跃会话时丢弃并返回 False。"""
    q = _active_notify_queue.get()
    if q is None:
        return False
    q.put_nowait({"message": message, "kind": kind, "payload": payload or {}})
    return True


def bind_notify_queue(queue: asyncio.Queue[dict[str, Any]]) -> Token[asyncio.Queue[dict[str, Any]] | None]:
    return _active_notify_queue.set(queue)


def unbind_notify_queue(token: Token[asyncio.Queue[dict[str, Any]] | None]) -> None:
    _active_notify_queue.reset(token)


def _safe_workspace_path(workspace_dir: Path, rel: str) -> Path:
    """阻止 ``..`` 越界 / 绝对路径访问 workspace 之外的位置。"""
    target = (workspace_dir / rel).resolve()
    workspace_resolved = workspace_dir.resolve()
    if workspace_resolved not in target.parents and target != workspace_resolved:
        raise PermissionError(f"path 越界: {rel}")
    return target


class AgentRuntimeBridge:
    """实现 ``prts.runtime.RuntimeBridge`` 协议,注入到 SDK。"""

    def __init__(
        self,
        workspace_dir: Path,
        store: SqliteStore,
        tools: ToolRegistry,
        llm_client: LlmClient,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._store = store
        self._tools = tools
        self._llm = llm_client

    async def notify(
        self,
        message: str,
        *,
        kind: str = "info",
        payload: dict[str, Any] | None = None,
    ) -> None:
        delivered = push_notify(message, kind=kind, payload=payload)
        if not delivered:
            logger.warning("notify dropped (no active session): %s", message[:120])

    async def invoke_skill(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._tools.invoke(name, arguments)

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._tools.invoke(name, arguments)

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        msgs: list[ChatMessage] = [ChatMessage(role=m["role"], content=m["content"]) for m in messages]  # type: ignore[typeddict-item]
        return await self._llm.chat(msgs)

    async def read_workspace(self, path: str) -> str:
        target = _safe_workspace_path(self._workspace_dir, path)
        return target.read_text(encoding="utf-8")

    async def write_workspace(self, path: str, content: str) -> None:
        target = _safe_workspace_path(self._workspace_dir, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    async def list_workspace(self, prefix: str = "") -> list[str]:
        base = self._workspace_dir
        out: list[str] = []
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            if rel.startswith(prefix):
                out.append(rel)
        return sorted(out)

    async def history(self, session_id: str | None, limit: int) -> list[dict[str, Any]]:
        if not session_id:
            from prts.context import current as _current_ctx

            session_id = _current_ctx().session_id
        rows = await self._store.history(session_id, limit=limit)
        return [
            {
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at,
                "meta": r.meta,
            }
            for r in rows
        ]
