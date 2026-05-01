"""Agent Loop —— 一个 ``converse`` 调用可能跨多轮 LLM ↔ 工具往返。

主流程:
1. 把用户消息持久化
2. 取 history → 构造 ``messages``(system prompt + 历史)
3. ``llm.stream_chat(messages, tools)`` 流式推理
   - text delta:转成 ``token`` 事件流向上游
   - tool_call:暂存,等本轮 stream 结束后统一调度
4. 本轮结束后:
   - 没工具调用 → 把 assistant 文本写库,发 ``done``,退出
   - 有工具调用 → 写 assistant(text + tool_calls 元数据);
     依次 invoke,每个结果以 ``tool`` role 写库 + 通过 ``tool_result`` 事件
     发出去;然后 goto 1(重新拉 history)
5. 上限 ``MAX_ITERATIONS`` 防止 LLM 死循环互调工具

skill 内部 ``client.notify(...)`` 走 ``runtime.push_notify`` → contextvar 队列,
我们在每条工具结果之后把队列里的事件 flush 到 SSE。
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from prts.context import CallContext as PrtsCallContext
from prts.context import reset as prts_reset
from prts.context import set as prts_set

from ..llm import (
    ChatMessage,
    EndEvent,
    LlmClient,
    TextEvent,
    ToolCallEvent,
)
from ..llm.anthropic_client import AnthropicLlmClient
from ..memory import SqliteStore, StoredMessage
from ..runtime import bind_notify_queue, unbind_notify_queue
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8


def _stored_to_chat(messages: list[StoredMessage]) -> list[ChatMessage]:
    """SQLite ``StoredMessage`` → LLM ``ChatMessage``(OpenAI 风格)。"""
    out: list[ChatMessage] = []
    for m in messages:
        if m.role == "assistant":
            msg: ChatMessage = {"role": "assistant", "content": m.content}
            if m.meta and m.meta.get("tool_calls"):
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                        },
                    }
                    for tc in m.meta["tool_calls"]
                ]
            out.append(msg)
        elif m.role == "tool":
            tool_call_id = (m.meta or {}).get("tool_call_id", "")
            tool_name = (m.meta or {}).get("tool_name", "")
            entry: ChatMessage = {
                "role": "tool",
                "content": m.content,
                "tool_call_id": tool_call_id,
            }
            if tool_name:
                entry["name"] = tool_name
            out.append(entry)
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _serialize_tool_result(result: Any) -> str:
    """工具结果统一转成字符串塞进 ``tool`` 消息的 content。"""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


class AgentLoop:
    def __init__(
        self,
        store: SqliteStore,
        llm: LlmClient,
        tools: ToolRegistry,
    ) -> None:
        self._store = store
        self._llm = llm
        self._tools = tools

    async def converse(
        self,
        session_id: str,
        user_content: str,
        system_prompt: str,
        *,
        channel: str = "web",
        user_ref: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields SSE-friendly dicts: ``{"event": str, "data": dict}``。"""
        await self._store.ensure_session(session_id, channel=channel, user_ref=user_ref)
        await self._store.append_message(session_id, "user", user_content)

        notify_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        nq_token = bind_notify_queue(notify_queue)

        history_rows = await self._store.history(session_id)
        ctx_token = prts_set(
            PrtsCallContext(
                session_id=session_id,
                user_id=user_ref,
                channel=channel,
                history=[
                    {"role": m.role, "content": m.content, "created_at": m.created_at}
                    for m in history_rows
                ],
            )
        )

        try:
            for iteration in range(MAX_ITERATIONS):
                # 每轮重新拉 history,这样新写入的 assistant + tool 行立刻可见。
                history_rows = await self._store.history(session_id)
                messages: list[ChatMessage] = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.extend(_stored_to_chat(history_rows))

                tool_defs = self._tools.all()
                openai_tools = self._tools.to_openai_tools() if tool_defs else None
                # OpenAILlmClient 接 OpenAI 风格;Anthropic 客户端会在内部转成 tool_use。
                # 两边都拿 OpenAI 风格 tools,Anthropic 客户端忽略 type/function 包装,
                # 但 schema 主体一致。简单起见这里给 OpenAI 风格,Anthropic 客户端
                # 自己重新打包:
                anthropic_tools = self._tools.to_anthropic_tools() if tool_defs else None

                # 选哪一种取决于 client 类型;一个粗暴但可读的判断:
                tools_arg = (
                    anthropic_tools
                    if isinstance(self._llm, AnthropicLlmClient)
                    else openai_tools
                )

                pending_calls: list[dict[str, Any]] = []
                assistant_text_acc: list[str] = []
                end_evt: EndEvent | None = None

                async for evt in self._llm.stream_chat(messages, tools=tools_arg):
                    if isinstance(evt, TextEvent):
                        assistant_text_acc.append(evt.delta)
                        yield {"event": "token", "data": {"text": evt.delta}}
                        async for ne in self._drain_notify(notify_queue):
                            yield ne
                    elif isinstance(evt, ToolCallEvent):
                        pending_calls.append(
                            {"id": evt.id, "name": evt.name, "arguments": evt.arguments}
                        )
                    elif isinstance(evt, EndEvent):
                        end_evt = evt

                # 一轮 LLM 流结束。先把队列里残留的 notify 全部 flush 出去。
                async for ne in self._drain_notify(notify_queue):
                    yield ne

                assistant_text = "".join(assistant_text_acc)
                meta: dict[str, Any] | None = None
                if pending_calls:
                    meta = {"tool_calls": pending_calls}
                # 即便没文字也要写一行(保留 tool_calls 元信息),否则下一轮组 messages 时
                # 会缺失 tool_call 上下文。
                if assistant_text or pending_calls:
                    await self._store.append_message(
                        session_id, "assistant", assistant_text, meta=meta
                    )

                if not pending_calls:
                    yield {"event": "done", "data": {"session_id": session_id}}
                    return

                for call in pending_calls:
                    yield {
                        "event": "tool_call",
                        "data": {
                            "id": call["id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }
                    is_error = False
                    try:
                        result = await self._tools.invoke(call["name"], call["arguments"])
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("tool %s failed", call["name"])
                        result = {"error": str(exc), "type": type(exc).__name__}
                        is_error = True

                    result_text = _serialize_tool_result(result)
                    await self._store.append_message(
                        session_id,
                        "tool",
                        result_text,
                        meta={
                            "tool_call_id": call["id"],
                            "tool_name": call["name"],
                            "is_error": is_error,
                        },
                    )
                    yield {
                        "event": "tool_result",
                        "data": {
                            "id": call["id"],
                            "name": call["name"],
                            "result": result if not is_error else None,
                            "error": result if is_error else None,
                        },
                    }
                    async for ne in self._drain_notify(notify_queue):
                        yield ne

                if end_evt is None:
                    logger.warning("LLM stream ended without EndEvent")

            # 触底:工具循环过深
            yield {
                "event": "error",
                "data": {"message": f"agent loop exceeded {MAX_ITERATIONS} iterations"},
            }
        finally:
            unbind_notify_queue(nq_token)
            prts_reset(ctx_token)

    async def _drain_notify(
        self, queue: asyncio.Queue[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """把 contextvar 队列里堆积的 notify 事件搬到 SSE 流。"""
        while True:
            try:
                item = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            yield {
                "event": "notify",
                "data": {
                    "message": item["message"],
                    "kind": item.get("kind", "info"),
                    "payload": item.get("payload", {}),
                },
            }
