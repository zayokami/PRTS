"""prts.context —— 当前调用上下文(session / user / history)。

基于 contextvars,跨协程安全。Agent 在每次 LLM 调用前 ``set()`` 一份,
跑完(包含工具调用嵌套)后 ``reset()``。
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Any

from . import runtime as _runtime


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
            "脚本若直接 `python xxx.py` 不会有 context。"
        )
    return ctx


def set(ctx: CallContext) -> Token[CallContext | None]:  # noqa: A001
    """Agent 内部使用:绑定一份新的上下文,返回 token 用于 reset。"""
    return _current.set(ctx)


def reset(token: Token[CallContext | None]) -> None:
    _current.reset(token)


async def tool(name: str, **kwargs: Any) -> Any:
    """跨工具调用:在 skill 内部触发其他 MCP 工具或 skill。

    P3:同进程 ToolRegistry。P4:可路由到外部 MCP server。
    """
    return await _runtime.get_runtime().invoke_tool(name, kwargs)
