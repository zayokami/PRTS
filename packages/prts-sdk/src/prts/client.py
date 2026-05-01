"""prts.client —— 脚本反向控制 Agent。

skill / task 内部用:推消息 / 调其他 skill / 主动开一轮对话。
所有方法都通过 :mod:`prts.runtime` 桥接到 Agent 内部实现。
"""

from __future__ import annotations

from typing import Any

from . import runtime as _runtime


async def notify(message: str, *, kind: str = "info", payload: dict[str, Any] | None = None) -> None:
    """主动给当前用户推消息,经 SSE → Gateway → 当前会话渠道。"""
    await _runtime.get_runtime().notify(message, kind=kind, payload=payload)


async def chat(message: str, **kwargs: Any) -> str:
    """主动开一轮 LLM 对话(无工具),返回完整字符串。"""
    return await _runtime.get_runtime().chat(
        [{"role": "user", "content": message}], **kwargs
    )


async def skill(name: str, **kwargs: Any) -> Any:
    """同进程调用别的 ``@skill`` 注册函数。"""
    return await _runtime.get_runtime().invoke_skill(name, kwargs)
