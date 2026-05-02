"""把 ``prts.runtime.RuntimeBridge`` 协议跑通到 Agent 内部能力。

skill / task 内的 ``prts.client.notify(...)`` / ``prts.workspace.read(...)``
最终走到这里。``notify`` 通过一个 contextvar 维护"当前会话的 SSE 事件队列"
—— Agent loop 启动时把队列 set 进去,工具调用结束 reset。这样多个并发会话
互不干扰。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from contextvars import ContextVar, Token
from pathlib import Path
from typing import Any

from .llm import ChatMessage, LlmClient
from .llm.embedding import EmbeddingClient
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
    """阻止 ``..`` 越界 / 绝对路径访问 workspace 之外的位置。

    用 ``Path.relative_to`` 做归属判断:它在 Windows 上是大小写不敏感的、在
    POSIX 上是大小写敏感的,符合各自文件系统语义。如果 rel 解析后跑出
    workspace 树外,relative_to 抛 ValueError,翻成 PermissionError。
    """
    if not rel:
        raise PermissionError("path 不能为空")
    # 先拒绝绝对路径 —— Path("/foo") / "/etc/passwd" 在 POSIX 会丢掉左半边,
    # 直接用绝对 rel 拼出去就绕过了 workspace_dir 的限制。
    if Path(rel).is_absolute() or rel.startswith(("/", "\\")):
        raise PermissionError(f"绝对路径被拒: {rel}")

    workspace_resolved = workspace_dir.resolve()
    target = (workspace_dir / rel).resolve()
    try:
        target.relative_to(workspace_resolved)
    except ValueError as exc:
        raise PermissionError(f"path 越界 (相对于 {workspace_resolved}): {rel}") from exc
    return target


class AgentRuntimeBridge:
    """实现 ``prts.runtime.RuntimeBridge`` 协议,注入到 SDK。"""

    def __init__(
        self,
        workspace_dir: Path,
        store: SqliteStore,
        tools: ToolRegistry,
        llm_client: LlmClient,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._store = store
        self._tools = tools
        self._llm = llm_client
        self._embedding = embedding_client

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
        # 透传 OpenAI 风格的 tool 字段。``prts.llm.chat`` 的常规用法是单轮翻译 /
        # 总结,但用户也可能塞完整 history 进来做"基于历史的二次推理" —— 那时
        # 必须保留 ``tool_calls`` / ``tool_call_id`` / ``name``,否则下游 LLM
        # 会拿到孤立的 "tool" 消息(没有对应的 assistant.tool_calls),很多
        # provider 会直接 400 报错。
        msgs: list[ChatMessage] = []
        for m in messages:
            entry: ChatMessage = {
                "role": m["role"],
                "content": m.get("content", ""),
            }
            for opt_key in ("tool_calls", "tool_call_id", "name"):
                if opt_key in m:
                    entry[opt_key] = m[opt_key]  # type: ignore[literal-required]
            msgs.append(entry)
        # tools 仅在用户明确传时才透传:默认不带,符合 ``prts.llm.chat`` 的
        # "纯文本辅助调用" 语义。其他未识别 kwargs 静默忽略,避免 LLM 客户端
        # 因无关参数报错。
        tools = kwargs.get("tools")
        return await self._llm.chat(msgs, tools=tools)

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
            rel_parts = p.relative_to(base).parts
            # 跳过 Python 缓存:这些是 import skill 时附带产生的,不属于
            # workspace 内容。同样跳过任意以 _ 开头的目录(_examples/ 等)。
            if any(part == "__pycache__" or part.endswith(".pyc") for part in rel_parts):
                continue
            rel = p.relative_to(base).as_posix()
            if rel.startswith(prefix):
                out.append(rel)
        return sorted(out)

    async def history(self, session_id: str | None, limit: int) -> list[dict[str, Any]]:
        if not session_id:
            from prts.context import current as _current_ctx

            try:
                session_id = _current_ctx().session_id
            except RuntimeError as exc:
                # 没有活跃 PRTS context 时直接给空列表,而不是把 RuntimeError
                # 冒泡到用户脚本 —— 用户脚本调 prts.memory.history() 拿空就好。
                logger.warning("history() called without context: %s", exc)
                return []
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

    async def remember(self, text: str, payload: dict[str, Any] | None = None) -> None:
        if self._embedding is None:
            logger.debug("remember skipped: no embedding client")
            return
        try:
            vec = await self._embedding.embed(text)
            merged = {**(payload or {}), "text": text}
            await self._tools.invoke(
                "prts-vector__upsert",
                {
                    "id": (payload.get("id") if payload else None)
                    or f"mem-{hashlib.sha256(text.encode()).hexdigest()[:16]}",
                    "vector": vec,
                    "payload": merged,
                },
            )
        except Exception:
            logger.exception("remember failed")

    async def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self._embedding is None:
            return []
        try:
            vec = await self._embedding.embed(query)
            raw = await self._tools.invoke(
                "prts-vector__search",
                {"query_vector": vec, "top_k": top_k},
            )
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict) and raw.get("ok"):
                return [
                    {
                        "id": r["id"],
                        "distance": r["distance"],
                        "payload": r.get("payload"),
                    }
                    for r in raw.get("results", [])
                ]
            return []
        except Exception:
            logger.exception("search_memory failed")
            return []
