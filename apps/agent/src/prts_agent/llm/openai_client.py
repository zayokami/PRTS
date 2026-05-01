"""OpenAI 协议兼容的流式客户端。

P3 起支持 tool calling。OpenAI 的 streaming 把 ``tool_calls`` 拆成多个增量
delta(按 index 拼参数 JSON),需要本地累积后再以 ``ToolCallEvent`` 发出。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from .base import ChatMessage, EndEvent, LlmClient, StreamEvent, TextEvent, ToolCallEvent

logger = logging.getLogger(__name__)


class OpenAILlmClient(LlmClient):
    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

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
        finish_reason: str | None = None

        async for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta

            if delta is None:
                pass
            else:
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
            call_id = slot["id"] or f"call_{idx}"
            name = slot["name"] or ""
            tool_calls_out.append({"id": call_id, "name": name, "arguments": args_obj})
            yield ToolCallEvent(type="tool_call", id=call_id, name=name, arguments=args_obj)

        # 把"原样的 assistant 消息"打包给上层,方便回填给下一轮 LLM
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": assistant_text}
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
