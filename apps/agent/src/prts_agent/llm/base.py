"""LLM 客户端抽象接口。

P1 阶段只需要流式文本输出;tool calling / vision / 多模态等留给后续阶段。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import TypedDict


class ChatMessage(TypedDict):
    role: str  # "system" | "user" | "assistant"
    content: str


class LlmClient(ABC):
    @abstractmethod
    def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        """逐 token 异步产出文本块。"""
        raise NotImplementedError
