"""LLM 客户端工厂。

通过 ``LLM_PROVIDER`` 环境变量切换:
- ``openai`` (默认): 走 ``AsyncOpenAI(base_url=...)``,兼容 OpenAI/DeepSeek/Ollama/Anthropic-compat
- ``anthropic``: 走 native ``anthropic`` SDK,保留 prompt caching / strict tool schema
"""

from __future__ import annotations

import os

from .anthropic_client import AnthropicLlmClient
from .base import (
    ChatMessage,
    EndEvent,
    LlmClient,
    StreamEvent,
    TextEvent,
    ToolCallEvent,
)
from .openai_client import OpenAILlmClient


def build_llm_client() -> LlmClient:
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    if provider == "anthropic":
        return AnthropicLlmClient(
            api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"),
        )
    return OpenAILlmClient(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("LLM_API_KEY", ""),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )


__all__ = [
    "ChatMessage",
    "EndEvent",
    "LlmClient",
    "StreamEvent",
    "TextEvent",
    "ToolCallEvent",
    "build_llm_client",
]
