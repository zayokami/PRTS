"""OpenAI 协议兼容的流式客户端。

适配 OpenAI 官方、DeepSeek、Ollama (/v1)、Anthropic OpenAI 兼容层等。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from .base import ChatMessage, LlmClient


class OpenAILlmClient(LlmClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        )
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content
