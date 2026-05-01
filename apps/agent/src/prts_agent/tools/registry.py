"""ToolRegistry —— Agent 内部维护的工具登记表。

P3:仅装从 ``workspace/skills/*.py`` 通过 ``@prts.skill`` 注册的函数。
P4:外部 MCP server 暴露的工具也加到这里,LLM 看到的接口面统一。

存的不是 ``SkillRegistration`` 本身,而是更通用的 ``ToolDefinition``,
不依赖 prts-sdk —— 后续接 MCP 时只需另一个适配层把 mcp tool 包成
``ToolDefinition`` 注册进来即可。
"""

from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    name: str
    description: str | None
    input_schema: dict[str, Any]
    invoker: Callable[[dict[str, Any]], Awaitable[Any]]
    source: str = "skill"  # P4 起会有 "mcp" / "builtin"
    extra: dict[str, Any] = field(default_factory=dict)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            logger.warning("tool %s already registered, overwriting", tool.name)
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def all(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def clear(self) -> None:
        self._tools.clear()

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")
        return await tool.invoker(arguments)

    # ---------- LLM 协议适配器 ---------- #

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """转成 OpenAI Chat Completions 的 ``tools`` 数组。"""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    def to_anthropic_tools(self) -> list[dict[str, Any]]:
        """转成 Anthropic Messages API 的 ``tools`` 数组。"""
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]


def make_skill_invoker(func: Callable[..., Any]) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """把同步 / 异步 skill 函数包成统一的 ``async (args) -> result``。"""
    if inspect.iscoroutinefunction(func):

        async def _async(arguments: dict[str, Any]) -> Any:
            return await func(**arguments)

        return _async

    async def _sync(arguments: dict[str, Any]) -> Any:
        return func(**arguments)

    return _sync
