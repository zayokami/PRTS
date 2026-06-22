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
from typing import TYPE_CHECKING, Any, Literal, TypedDict, Union

if TYPE_CHECKING:
    from .model_compat import ModelCompatConfig


class ChatMessage(TypedDict, total=False):
    role: str  # "system" | "user" | "assistant" | "tool"
    content: Any  # str 或 list (Anthropic 风格 content blocks)
    tool_calls: list[dict[str, Any]]  # OpenAI 风格,assistant 携带的工具调用
    tool_call_id: str  # OpenAI 风格,tool 消息回填
    name: str  # OpenAI 风格,tool 消息工具名
    reasoning_content: str  # DeepSeek V4 推理内容,多轮对话必须传回


@dataclass
class TokenUsage:
    """LLM 响应中的 token 消耗统计。

    P8: 用于校准本地 heuristic token 计数,让上下文预算更精准。
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


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
class UsageEvent:
    """流式响应中携带的 token 消耗事件。

    不同 provider 在 stream 的不同位置提供 usage:
    - OpenAI: 部分 provider 在最后一个 chunk 或 choices[0].usage
    - Anthropic: 在 message_stop 事件
    """

    type: Literal["usage"]
    usage: TokenUsage


@dataclass
class EndEvent:
    type: Literal["end"]
    stop_reason: str  # "stop" | "tool_use" | "length" | "error" | ...
    raw_assistant_message: dict[str, Any] = field(default_factory=dict)


StreamEvent = Union[TextEvent, ToolCallEvent, UsageEvent, EndEvent]


class LlmClient(ABC):
    """LLM 客户端抽象接口。

    子类必须暴露 ``model`` (只读) 供上层做 token 预算管理。
    ``compat`` 属性返回模型的兼容性配置,驱动 provider 特定行为。
    """

    def __init__(self) -> None:
        self._last_usage: TokenUsage | None = None
        self._last_reasoning_content: str = ""
        self._compat: ModelCompatConfig | None = None

    @property
    @abstractmethod
    def model(self) -> str:
        """当前使用的模型标识字符串,如 ``"gpt-4o-mini"`` 或 ``"claude-sonnet-4-6"`` 。"""
        raise NotImplementedError

    @property
    def compat(self) -> ModelCompatConfig:
        """模型的兼容性配置。首次访问时按 model 名查找 registry。"""
        if self._compat is None:
            from .model_compat import get_compat

            self._compat = get_compat(self.model)
        return self._compat

    @property
    @abstractmethod
    def model(self) -> str:
        """当前使用的模型标识字符串,如 ``"gpt-4o-mini"`` 或 ``"claude-sonnet-4-6"`` 。"""
        raise NotImplementedError

    @property
    def context_limit(self) -> int:
        """该模型的上下文窗口大小 (tokens)。"""
        from .tokenizer import get_context_limit

        return get_context_limit(self.model)

    @property
    def last_usage(self) -> TokenUsage | None:
        """上一次 ``stream_chat`` 的实际 token 消耗。

        仅在流正常结束后有效;如果流中途取消或未发 UsageEvent,可能为 None。
        """
        return self._last_usage

    @property
    def last_reasoning_content(self) -> str:
        """上一次 ``stream_chat`` 的 reasoning_content (DeepSeek V4 等推理模型)。

        多轮对话中必须传回给 API,否则会得到 400 错误。
        """
        return self._last_reasoning_content

    @abstractmethod
    def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
        abort_signal: Any = None,
    ) -> AsyncIterator[StreamEvent]:
        """流式 chat。``tools`` 为 None 时退化为无工具普通对话。

        ``abort_signal`` (asyncio.Event) 被 set 时,stream 在下一个 chunk 边界退出。
        """
        raise NotImplementedError

    async def chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """便利方法:聚合 ``stream_chat`` 的所有文本,忽略工具调用和 usage 事件。"""
        buf: list[str] = []
        async for evt in self.stream_chat(messages, tools=tools):
            if isinstance(evt, TextEvent):
                buf.append(evt.delta)
        return "".join(buf)
