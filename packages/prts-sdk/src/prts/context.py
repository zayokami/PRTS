"""prts.context —— 当前调用上下文(session / user / history)。

基于 contextvars,跨协程安全。P0 占位,P3 由 Agent 在每次调用时注入。
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CallContext:
    session_id: str
    user_id: str | None = None
    channel: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


_current: ContextVar[CallContext | None] = ContextVar("prts_call_context", default=None)


def current() -> CallContext:
    ctx = _current.get()
    if ctx is None:
        raise RuntimeError(
            "prts.context.current() 需要在 PRTS Agent 运行时调用。"
            "脚本若直接运行不会有 context。"
        )
    return ctx


async def tool(name: str, **kwargs: Any) -> Any:
    """跨工具调用:在 skill 内部触发其他 MCP 工具或 skill。P3 实现。"""
    raise NotImplementedError("P3 阶段实现")
