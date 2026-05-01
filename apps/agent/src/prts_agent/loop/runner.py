"""Agent Loop —— 一个 ``converse`` 调用可能跨多轮 LLM ↔ 工具往返。

主流程:
1. 把用户消息持久化
2. 取 history → 构造 ``messages``(system prompt + 历史)
3. ``llm.stream_chat(messages, tools)`` 流式推理
   - text delta:转成 ``token`` 事件流向上游
   - tool_call:暂存,等本轮 stream 结束后统一调度
4. 本轮结束后:
   - 没工具调用 → 把 assistant 文本写库,发 ``done``,退出
   - 有工具调用 → 顺序 invoke,**最后一个事务里** 把 assistant 行(含
     tool_calls 元数据) + 全部 tool 行一起写下去。途中 yield tool_call /
     tool_result 事件以维持 UI 实时感,即便 crash 也只丢内存中的本轮内容,
     不会留下"assistant 写了但 tool_result 缺失"的半成品 history。
   - 然后 goto 1(重新拉 history)
5. 上限 ``MAX_ITERATIONS`` 防止 LLM 死循环互调工具;触底时写一行
   "stopped due to limit" 的 assistant 消息收尾,避免下次还看到悬挂 tool_calls

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
from ..memory import SqliteStore
from ..memory.sqlite import PendingMessage
from ..runtime import bind_notify_queue, unbind_notify_queue
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8


def _stored_to_chat(messages: list) -> list[ChatMessage]:
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
                anthropic_tools = self._tools.to_anthropic_tools() if tool_defs else None
                tools_arg = (
                    anthropic_tools
                    if isinstance(self._llm, AnthropicLlmClient)
                    else openai_tools
                )

                pending_calls: list[dict[str, Any]] = []
                assistant_text_acc: list[str] = []
                end_evt: EndEvent | None = None
                stream_failed: BaseException | None = None

                try:
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
                except BaseException as exc:  # noqa: BLE001
                    # LLM 流半路异常:把已经收到的文本 / tool_calls 落库,然后冒泡。
                    # 不写库的话,前端拿到的 token 已经渲染但 history 没有这条 assistant,
                    # 下一轮会出现"用户视角看到了 PRTS 说话但 LLM 视角没说过"的悖论。
                    logger.exception("stream_chat failed mid-stream")
                    stream_failed = exc

                # 一轮 LLM 流结束。先把队列里残留的 notify 全部 flush 出去。
                async for ne in self._drain_notify(notify_queue):
                    yield ne

                assistant_text = "".join(assistant_text_acc)

                # ---- 没工具调用:写完 assistant 直接结束本次 converse ----
                if not pending_calls:
                    if assistant_text:
                        await self._store.append_message(
                            session_id, "assistant", assistant_text
                        )
                    if stream_failed is not None:
                        yield {
                            "event": "error",
                            "data": {
                                "message": str(stream_failed),
                                "type": type(stream_failed).__name__,
                            },
                        }
                        return
                    # finish_reason=length 也算 done:LLM 因为 max_tokens 截断,
                    # 把已经吐出的内容当成最终答复;由前端决定是否提示用户重试。
                    stop_reason = end_evt.stop_reason if end_evt else "stop"
                    yield {
                        "event": "done",
                        "data": {"session_id": session_id, "stop_reason": stop_reason},
                    }
                    return

                # ---- 有工具调用:发事件 → invoke → 收结果 ----
                # 先 yield 所有 tool_call 事件让 UI 立刻看到;之后顺序执行,
                # 每个 tool_result 事件都立刻 yield 出去,保留交互实时性。
                for call in pending_calls:
                    yield {
                        "event": "tool_call",
                        "data": {
                            "id": call["id"],
                            "name": call["name"],
                            "arguments": call["arguments"],
                        },
                    }

                tool_outcomes: list[tuple[dict[str, Any], Any, bool]] = []
                for call in pending_calls:
                    is_error = False
                    try:
                        result = await self._tools.invoke(call["name"], call["arguments"])
                    except Exception as exc:  # noqa: BLE001
                        logger.exception("tool %s failed", call["name"])
                        result = {"error": str(exc), "type": type(exc).__name__}
                        is_error = True

                    tool_outcomes.append((call, result, is_error))
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

                # ---- 一次事务把 assistant + 所有 tool 行写下去 ----
                # 这样要么本轮的 (assistant + 全部 tool_results) 整体在 history 里,
                # 要么完全不在,绝不会出现 assistant 有 tool_calls 而 tool_result 缺失。
                batch: list[PendingMessage] = [
                    PendingMessage(
                        role="assistant",
                        content=assistant_text,
                        meta={"tool_calls": pending_calls},
                    )
                ]
                for call, result, is_error in tool_outcomes:
                    batch.append(
                        PendingMessage(
                            role="tool",
                            content=_serialize_tool_result(result),
                            meta={
                                "tool_call_id": call["id"],
                                "tool_name": call["name"],
                                "is_error": is_error,
                            },
                        )
                    )
                await self._store.append_messages(session_id, batch)

                if stream_failed is not None:
                    yield {
                        "event": "error",
                        "data": {
                            "message": str(stream_failed),
                            "type": type(stream_failed).__name__,
                        },
                    }
                    return

                if end_evt is None:
                    logger.warning("LLM stream ended without EndEvent")

            # 触底:工具循环过深。补一行 assistant 收尾,避免悬挂 tool_calls。
            await self._store.append_message(
                session_id,
                "assistant",
                f"(已达到工具循环上限 {MAX_ITERATIONS} 次,放弃后续调用。)",
            )
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
