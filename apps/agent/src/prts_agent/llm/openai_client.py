"""OpenAI 协议兼容的流式客户端。

P3 起支持 tool calling。OpenAI 的 streaming 把 ``tool_calls`` 拆成多个增量
delta(按 index 拼参数 JSON),需要本地累积后再以 ``ToolCallEvent`` 发出。
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from .base import (
    ChatMessage,
    EndEvent,
    LlmClient,
    StreamEvent,
    TextEvent,
    TokenUsage,
    ToolCallEvent,
    UsageEvent,
)

logger = logging.getLogger(__name__)


class OpenAILlmClient(LlmClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        super().__init__()
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def close(self) -> None:
        """关闭底层 HTTP 连接池。Agent shutdown / reload 时调用,防连接泄漏。"""
        await self._client.close()

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = await self._client.chat.completions.create(**kwargs)

        # 按 index 累积 tool call。OpenAI 增量结构:
        #   delta.tool_calls = [{"index": 0, "id": "call_x", "function": {"name": "...", "arguments": "..."}}]
        tool_buf: dict[int, dict[str, Any]] = {}
        text_acc: list[str] = []
        reasoning_acc: list[str] = []
        finish_reason: str | None = None

        async for chunk in stream:
            # Parse usage — location varies by provider:
            # - 标准 OpenAI: chunk.usage (最后一个 chunk)
            # - Groq: chunk.x_groq.usage
            # - 某些代理: chunk.choices[0].usage
            usage = getattr(chunk, "usage", None)
            if usage is None and chunk.choices:
                usage = getattr(chunk.choices[0], "usage", None)
            if usage is None:
                # Groq 等特殊 provider
                usage = getattr(getattr(chunk, "x_groq", None), "usage", None)

            if usage and usage.prompt_tokens is not None:
                self._last_usage = TokenUsage(
                    prompt_tokens=usage.prompt_tokens,
                    completion_tokens=usage.completion_tokens or 0,
                    total_tokens=usage.total_tokens
                    or (usage.prompt_tokens + (usage.completion_tokens or 0)),
                    model=self._model,
                )
                yield UsageEvent(type="usage", usage=self._last_usage)

            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta is None:
                pass
            else:
                # DeepSeek V4: 捕获 reasoning_content (多轮对话必须传回)
                rc = getattr(delta, "reasoning_content", None)
                if rc:
                    reasoning_acc.append(rc)

                if delta.content:
                    text_acc.append(delta.content)
                    yield TextEvent(type="text", delta=delta.content)

                if getattr(delta, "tool_calls", None):
                    for tc in delta.tool_calls:
                        idx = tc.index if tc.index is not None else 0
                        slot = tool_buf.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if tc.id:
                            slot["id"] = tc.id
                        fn = getattr(tc, "function", None)
                        if fn is not None:
                            if getattr(fn, "name", None):
                                slot["name"] = fn.name
                            if getattr(fn, "arguments", None):
                                slot["arguments"] += fn.arguments

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        assistant_text = "".join(text_acc)
        tool_calls_out: list[dict[str, Any]] = []

        for idx in sorted(tool_buf.keys()):
            slot = tool_buf[idx]
            args_text = slot["arguments"] or "{}"
            try:
                args_obj = json.loads(args_text)
            except json.JSONDecodeError:
                logger.warning("malformed tool args from LLM: %r", args_text)
                args_obj = {"_raw": args_text}
            call_id = slot["id"] or f"call_auto_{uuid.uuid4().hex[:12]}"
            name = slot["name"] or ""
            tool_calls_out.append({"id": call_id, "name": name, "arguments": args_obj})
            yield ToolCallEvent(type="tool_call", id=call_id, name=name, arguments=args_obj)

        # 把"原样的 assistant 消息"打包给上层,方便回填给下一轮 LLM
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
        if reasoning_acc:
            reasoning_text = "".join(reasoning_acc)
            assistant_msg["reasoning_content"] = reasoning_text
            self._last_reasoning_content = reasoning_text
        else:
            self._last_reasoning_content = ""
        if tool_calls_out:
            assistant_msg["tool_calls"] = [
                {
                    "id": c["id"],
                    "type": "function",
                    "function": {
                        "name": c["name"],
                        "arguments": json.dumps(c["arguments"], ensure_ascii=False),
                    },
                }
                for c in tool_calls_out
            ]

        yield EndEvent(
            type="end",
            stop_reason=finish_reason or "stop",
            raw_assistant_message=assistant_msg,
        )
