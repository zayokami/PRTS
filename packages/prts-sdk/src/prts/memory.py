"""prts.memory —— 会话历史 / 向量检索。P0 占位。"""

from __future__ import annotations

from typing import Any


async def search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    raise NotImplementedError("P7 阶段接通 sqlite-vec")


async def remember(text: str, payload: dict[str, Any] | None = None) -> str:
    raise NotImplementedError("P7 阶段接通 sqlite-vec")


async def history(session_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    raise NotImplementedError("P2 阶段实现")
