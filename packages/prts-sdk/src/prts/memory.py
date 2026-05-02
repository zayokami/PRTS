"""prts.memory —— 会话历史 / 向量检索。

P3 接通 ``history()``;``search()`` / ``remember()`` 等向量 API 等 P7 sqlite-vec。
"""

from __future__ import annotations

from typing import Any

from . import runtime as _runtime


async def history(session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """返回 ``[{role, content, created_at, meta}]`` 列表,按时间升序。"""
    return await _runtime.get_runtime().history(session_id, limit)


async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    return await _runtime.get_runtime().search_memory(query, top_k)


async def remember(text: str, payload: dict[str, Any] | None = None) -> None:
    await _runtime.get_runtime().remember(text, payload)
