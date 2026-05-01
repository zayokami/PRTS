"""prts.llm —— 直接调 LLM(返回 OpenAI 协议响应)。P0 占位。"""

from __future__ import annotations

from typing import Any


async def chat(messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    """直接调底层 LLM 客户端,绕过 Agent loop。"""
    raise NotImplementedError("P1/P3 阶段接通")
