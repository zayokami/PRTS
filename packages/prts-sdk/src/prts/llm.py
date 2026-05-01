"""prts.llm —— 脚本里直接调底层 LLM。

不进入 Agent loop,不写 history,适合做"内部翻译 / 总结 / 分类"这类
不希望出现在用户对话里的辅助调用。
"""

from __future__ import annotations

from typing import Any

from . import runtime as _runtime


async def chat(messages: list[dict[str, Any]], **kwargs: Any) -> str:
    """完整文本回复(非流式)。"""
    return await _runtime.get_runtime().chat(messages, **kwargs)
