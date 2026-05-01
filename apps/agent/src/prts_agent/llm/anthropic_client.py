"""Native Anthropic 客户端。

走原生 SDK 保留 prompt caching / extended thinking;system 消息抽出来传给 system 参数。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from .base import ChatMessage, LlmClient


class AnthropicLlmClient(LlmClient):
    def __init__(self, api_key: str, model: str) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    async def stream_chat(self, messages: list[ChatMessage]) -> AsyncIterator[str]:
        system = "\n".join(m["content"] for m in messages if m["role"] == "system") or None
        non_system = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] != "system"
        ]
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": non_system,
            "max_tokens": 4096,
        }
        if system is not None:
            kwargs["system"] = system

        async with self._client.messages.stream(**kwargs) as stream:  # type: ignore[arg-type]
            async for text in stream.text_stream:
                yield text
