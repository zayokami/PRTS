"""prts.client —— 脚本反向控制 Agent(notify / chat / 等)。P0 占位。"""

from __future__ import annotations

from typing import Any


async def notify(message: str, **kwargs: Any) -> None:
    """主动给当前用户推消息(经 Gateway 走当前会话渠道)。"""
    raise NotImplementedError("P3 阶段实现")


async def chat(message: str, **kwargs: Any) -> str:
    """主动开一轮对话,LLM 流式返回完整字符串。"""
    raise NotImplementedError("P3 阶段实现")


async def skill(name: str, **kwargs: Any) -> Any:
    """同进程调用别的 @skill。"""
    raise NotImplementedError("P3 阶段实现")
