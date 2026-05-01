"""Native Anthropic 客户端。

走原生 SDK 保留 prompt caching / extended thinking / strict tool schema。
内部把 OpenAI 风格的 ``messages`` 转换成 Anthropic 的 content blocks 协议:

- ``role: assistant`` 带 ``tool_calls`` → ``content`` 数组里同时有 ``text`` 和
  ``tool_use`` 块
- ``role: tool`` → 转成 ``role: user`` + ``tool_result`` content 块
- ``role: system`` 抽出来作为 top-level ``system`` 参数

工具调用流式:监听 ``input_json_delta`` 增量 JSON,在 ``content_block_stop``
时尝试解析;text 内容仍按字符串增量推。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from anthropic import AsyncAnthropic

from .base import ChatMessage, EndEvent, LlmClient, StreamEvent, TextEvent, ToolCallEvent

logger = logging.getLogger(__name__)


def _convert_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_chunks: list[str] = []
    converted: list[dict[str, Any]] = []

    def _last_tool_result_user() -> dict[str, Any] | None:
        """如果上一条 converted 是承载 tool_result 的 user 消息,返回它。

        Anthropic 要求"一次 assistant.tool_use 多块"对应"紧跟一条 user 消息,
        content 里塞所有 tool_result"。我们的输入是 OpenAI 风格,每个 tool 各
        是一条独立的 tool 消息 —— 必须在转换时合并。
        """
        if not converted:
            return None
        last = converted[-1]
        if last.get("role") != "user":
            return None
        content = last.get("content")
        if not isinstance(content, list) or not content:
            return None
        if all(blk.get("type") == "tool_result" for blk in content):
            return last
        return None

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")

        if role == "system":
            if isinstance(content, str) and content:
                system_chunks.append(content)
            continue

        if role == "user":
            if isinstance(content, list):
                converted.append({"role": "user", "content": content})
            else:
                converted.append({"role": "user", "content": [{"type": "text", "text": content}]})
            continue

        if role == "assistant":
            blocks: list[dict[str, Any]] = []
            if isinstance(content, str) and content:
                blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                blocks.extend(content)
            for tc in msg.get("tool_calls", []) or []:
                fn = tc.get("function", {})
                args_raw = fn.get("arguments", "{}")
                try:
                    args_obj = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args_obj = {}
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": args_obj,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            converted.append({"role": "assistant", "content": blocks})
            continue

        if role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": msg.get("tool_call_id", ""),
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            }
            # 上一条已经是 tool_result-only 的 user → 把这块塞进去,合并多个 tool_result。
            existing = _last_tool_result_user()
            if existing is not None:
                existing["content"].append(block)
            else:
                converted.append({"role": "user", "content": [block]})
            continue

        logger.warning("unknown role %r in messages, skipping", role)

    system = "\n\n".join(s for s in system_chunks if s) or None
    return system, converted


class AnthropicLlmClient(LlmClient):
    def __init__(self, api_key: str, model: str, max_tokens: int = 4096) -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        system, converted = _convert_messages(messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": converted,
            "max_tokens": self._max_tokens,
        }
        if system is not None:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        # 按 content block index 累积当前块的状态
        block_state: dict[int, dict[str, Any]] = {}
        text_acc: list[str] = []
        tool_calls_out: list[dict[str, Any]] = []
        stop_reason = "stop"

        async with self._client.messages.stream(**kwargs) as stream:
            async for event in stream:
                etype = getattr(event, "type", None)

                if etype == "content_block_start":
                    blk = event.content_block
                    idx = event.index
                    if blk.type == "text":
                        block_state[idx] = {"kind": "text"}
                    elif blk.type == "tool_use":
                        block_state[idx] = {
                            "kind": "tool_use",
                            "id": blk.id,
                            "name": blk.name,
                            "json": "",
                        }

                elif etype == "content_block_delta":
                    idx = event.index
                    delta = event.delta
                    state = block_state.get(idx)
                    if state is None:
                        continue
                    if delta.type == "text_delta" and state["kind"] == "text":
                        text_acc.append(delta.text)
                        yield TextEvent(type="text", delta=delta.text)
                    elif delta.type == "input_json_delta" and state["kind"] == "tool_use":
                        state["json"] += delta.partial_json

                elif etype == "content_block_stop":
                    idx = event.index
                    state = block_state.get(idx)
                    if state and state["kind"] == "tool_use":
                        try:
                            args_obj = json.loads(state["json"]) if state["json"] else {}
                        except json.JSONDecodeError:
                            logger.warning("malformed tool input: %r", state["json"])
                            args_obj = {}
                        tool_calls_out.append(
                            {"id": state["id"], "name": state["name"], "arguments": args_obj}
                        )
                        yield ToolCallEvent(
                            type="tool_call",
                            id=state["id"],
                            name=state["name"],
                            arguments=args_obj,
                        )

                elif etype == "message_delta":
                    sr = getattr(event.delta, "stop_reason", None)
                    if sr:
                        stop_reason = sr

        # 构造原样 assistant 消息(OpenAI 风格,便于 history 回写)
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": "".join(text_acc),
        }
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

        yield EndEvent(type="end", stop_reason=stop_reason, raw_assistant_message=assistant_msg)
