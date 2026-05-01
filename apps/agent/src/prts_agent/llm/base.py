"""LLM 客户端抽象接口。

P3 起支持 tool calling。``stream_chat`` 返回一个 ``StreamEvent`` 流:

- ``TextEvent``: 增量文本(给前端流式展示)
- ``ToolCallEvent``: 一次完整工具调用请求(在 stream 末尾发出)
- ``EndEvent``: LLM 终止;附带 ``stop_reason``("stop"/"tool_use"/...)

为什么不在 stream 中同时增量推送 tool args:OpenAI / Anthropic 协议都允许
tool args 跨多个 chunk 拼接,但参数本身不是给用户看的,聚合后一次性给上层
更省事;而 text 必须流式以维持回复观感。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict, Union


class ChatMessage(TypedDict, total=False):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Any  # str 或 list (Anthropic 风格 content blocks)
    tool_calls: list[dict[str, Any]]  # OpenAI 风格,assistant 携带的工具调用
    tool_call_id: str  # OpenAI 风格,tool 消息回填
    name: str  # OpenAI 风格,tool 消息工具名


@dataclass
class TextEvent:
    type: Literal["text"]
    delta: str


@dataclass
class ToolCallEvent:
    type: Literal["tool_call"]
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class EndEvent:
    type: Literal["end"]
    stop_reason: str  # "stop" | "tool_use" | "length" | "error" | ...
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


StreamEvent = Union[TextEvent, ToolCallEvent, EndEvent]


class LlmClient(ABC):
    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式 chat。``tools`` 为 None 时退化为无工具普通对话。"""
        raise NotImplementedError

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """便利方法:聚合 ``stream_chat`` 的所有文本,忽略工具调用。"""
        buf: list[str] = []
        async for evt in self.stream_chat(messages, tools=tools):
            if isinstance(evt, TextEvent):
                buf.append(evt.delta)
        return "".join(buf)
