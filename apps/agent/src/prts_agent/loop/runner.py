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
from datetime import datetime, timezone
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
    UsageEvent,
)
from ..llm.anthropic_client import AnthropicLlmClient
from ..llm.embedding import EmbeddingClient
from ..llm.tokenizer import count_messages_tokens, set_calibration_store
from ..memory import SqliteStore
from ..memory.sqlite import PendingMessage
from ..memory.summarizer import DialogueSummarizer
from ..runtime import bind_notify_queue, unbind_notify_queue
from ..tools import (
    HookedToolRegistry,
    ToolHooks,
    ToolRegistry,
    ToolResult,
    build_default_permission_engine,
)

from .budget import DynamicBudget
from .context_manager import ContextManager
from .state_machine import AgentState, AgentStateMachine

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
MAX_TOOL_RESULT_CHARS = 16000

# --- P8 DEPRECATED: 以下常量已被 ContextManager 接管,保留仅向后兼容 ---
# RECENT_WINDOW, VECTOR_TOPK, TOKEN_HEADROOM, MIN_RECENT_PAIRS
# 现在通过 PRTS_CONTEXT_MODE / ContextManager 参数控制。
# RECENT_WINDOW = 20  # 见 context_manager.DEFAULT_RECENT_WINDOW
# VECTOR_TOPK = 5     # 见 context_manager.DEFAULT_VECTOR_TOPK
# TOKEN_HEADROOM = 0.80  # 见 budget.DynamicBudget
# MIN_RECENT_PAIRS = 4   # 见 context_manager.MIN_RECENT_PAIRS


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
    """工具结果统一转成字符串塞进 ``tool`` 消息的 content。

    支持 ToolResult 和普通值。正常路径 invoker 已经把 MCP ``CallToolResult``
    拆开了,这里只剩 str / dict / list / None 等普通值。
    """
    # 新路径:ToolResult
    if hasattr(result, "to_llm_text"):
        return result.to_llm_text()

    if isinstance(result, str):
        return result
    if hasattr(result, "isError") and hasattr(result, "content"):
        if getattr(result, "isError", False):
            parts = [
                getattr(b, "text", "")
                for b in (getattr(result, "content", []) or [])
                if getattr(b, "type", None) == "text"
            ]
            return "ERROR: " + ("\n\n".join(p for p in parts if p) or "no error text")
        structured = getattr(result, "structuredContent", None)
        if isinstance(structured, dict) and list(structured.keys()) == ["result"]:
            return _serialize_tool_result(structured["result"])
        if structured is not None:
            try:
                return json.dumps(structured, ensure_ascii=False, default=str)
            except (TypeError, ValueError):
                return str(structured)
        blocks = getattr(result, "content", []) or []
        if len(blocks) == 1 and getattr(blocks[0], "type", None) == "text":
            return getattr(blocks[0], "text", "")
        rendered = []
        for b in blocks:
            if hasattr(b, "model_dump"):
                rendered.append(b.model_dump(mode="json"))
            else:
                rendered.append({"type": getattr(b, "type", "unknown"), "repr": repr(b)})
        return json.dumps(rendered, ensure_ascii=False, default=str)
    try:
        return json.dumps(result, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(result)


def _truncate_for_llm(serialized: str) -> str:
    """tool 结果太长就裁掉尾巴,留个明确标记。

    动机:某个 tool 一次返回 200KB(典型场景:读大文件、密集向量检索结果)写进
    history 后,后续每一轮 LLM 都要重发整段;不仅烧 token,还会让 Anthropic
    prompt cache 命中率掉到 0(每轮 tool 消息内容相同但前缀变长不算缓存)。

    截断后丢进 history 的就是"短前缀 + 提示",LLM 能看到结果开头并知道被截了,
    需要细节时用 offset / 缩窄关键词重调即可。
    """
    if len(serialized) <= MAX_TOOL_RESULT_CHARS:
        return serialized
    head = serialized[:MAX_TOOL_RESULT_CHARS]
    return (
        f"{head}\n\n"
        f"[... truncated: full result was {len(serialized)} chars, "
        f"showing first {MAX_TOOL_RESULT_CHARS}. "
        "Re-call this tool with narrower scope if more detail needed.]"
    )


# ---- Tool Hook: 审计日志 ----

async def _log_tool_execution(invocation, result, duration_ms):
    """post-tool hook：记录每次工具调用的耗时与结果摘要。"""
    # 结果摘要：如果是字符串直接取前 200 字符，否则序列化后截取
    if isinstance(result, str):
        summary = result[:200]
    else:
        try:
            summary = json.dumps(result, ensure_ascii=False, default=str)[:200]
        except (TypeError, ValueError):
            summary = str(result)[:200]

    logger.info(
        "tool=%s source=%s session=%s channel=%s duration=%.1fms "
        "args_keys=%s result_summary=%s",
        invocation.name,
        invocation.source,
        invocation.session_id,
        invocation.channel,
        duration_ms,
        list(invocation.arguments.keys()),
        summary,
    )


class AgentLoop:
    def __init__(
        self,
        store: SqliteStore,
        llm: LlmClient,
        tools: ToolRegistry,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._tools = tools
        self._embedding = embedding_client

        # ---- Tool Hooks ----
        self._hooks = ToolHooks()
        # 1. 权限检查
        perm_engine = build_default_permission_engine()
        self._hooks.register_pre(perm_engine.check)
        # 2. 审计日志
        self._hooks.register_post(_log_tool_execution)

        # Initialize tokenizer calibration persistence (SQLite-backed)
        set_calibration_store(store)

        # Initialize three-tier memory components (short-term + mid-term + long-term)
        # DynamicBudget persists history to SQLite for cross-session learning
        self._budget = DynamicBudget(llm, store=store)
        self._summarizer = DialogueSummarizer(llm)
        self._context_manager = ContextManager(
            store=store,
            llm=llm,
            summarizer=self._summarizer,
            budget=self._budget,
            embedding=embedding_client,
            tools=tools,
        )

        # Agent state machine for explicit state tracking
        self._state_machine = AgentStateMachine()

        # Prompt injection guard
        try:
            from ..tools.prompt_injection import (
                PromptInjectionClassifier,
                build_injection_guard,
            )

            injection_classifier = PromptInjectionClassifier(sensitivity="medium")
            self._hooks.register_pre(build_injection_guard(injection_classifier))
        except Exception:
            logger.exception("failed to initialize prompt injection guard")

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

        # context 供 skill 脚本读取,取最近 N 条足够
        ctx_history = await self._store.history(session_id, limit=RECENT_WINDOW)
        ctx_token = prts_set(
            PrtsCallContext(
                session_id=session_id,
                user_id=user_ref,
                channel=channel,
                history=[
                    {"role": m.role, "content": m.content, "created_at": m.created_at}
                    for m in ctx_history
                ],
            )
        )

        try:
            self._state_machine.start()
            for iteration in range(MAX_ITERATIONS):
                self._state_machine.transition_to(
                    AgentState.RUNNING, f"iteration {iteration}"
                )
                # Build context with three-tier memory assembly (short/mid/long-term)
                messages = await self._context_manager.build_context(
                    session_id, user_content, system_prompt
                )

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
                stream_failed: Exception | None = None

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
                        elif isinstance(evt, UsageEvent):
                            # Record usage for dynamic budget calibration and persistence
                            self._budget.record(evt.usage)
                            logger.debug(
                                "usage recorded: prompt=%d completion=%d",
                                evt.usage.prompt_tokens,
                                evt.usage.completion_tokens,
                            )
                        elif isinstance(evt, EndEvent):
                            end_evt = evt
                except Exception as exc:  # noqa: BLE001
                    # LLM 流半路异常:把已经收到的文本 / tool_calls 落库,然后告诉前端。
                    # 不写库的话,前端拿到的 token 已经渲染但 history 没有这条 assistant,
                    # 下一轮会出现"用户视角看到了 PRTS 说话但 LLM 视角没说过"的悖论。
                    # 重要:这里 *不* 捕 BaseException —— ``asyncio.CancelledError`` /
                    # ``KeyboardInterrupt`` 必须直接冒泡到 finally,保留协作取消语义,
                    # 否则 uvicorn shutdown / 客户端断开都会被吞,半成品状态反而被持久化。
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
                    if assistant_text:
                        await self._auto_remember(
                            session_id, user_content, assistant_text, channel
                        )
                    self._state_machine.complete("done")
                    yield {
                        "event": "done",
                        "data": {
                            "session_id": session_id,
                            "stop_reason": stop_reason,
                            "state": self._state_machine.to_dict(),
                        },
                    }
                    return

                # ---- 有工具调用:发事件 → invoke → 收结果 ----
                self._state_machine.transition_to(
                    AgentState.AWAITING_TOOL,
                    f"{len(pending_calls)} tool_calls pending",
                )
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

                # 每次 converse 动态创建 HookedToolRegistry 以传入当前 session/channel
                hooked_tools = HookedToolRegistry(
                    inner=self._tools,
                    hooks=self._hooks,
                    session_id=session_id,
                    channel=channel,
                    timeout_seconds=60.0,  # 单个工具最多 60 秒
                    max_concurrent=4,      # 最多同时执行 4 个工具
                )

                tool_outcomes: list[tuple[dict[str, Any], ToolResult]] = []
                for call in pending_calls:
                    result = await hooked_tools.invoke(
                        call["name"], call["arguments"]
                    )
                    # result 一定是 ToolResult,不会抛异常
                    tool_outcomes.append((call, result))
                    yield {
                        "event": "tool_result",
                        "data": {
                            "id": call["id"],
                            "name": call["name"],
                            **result.to_sse_dict(),
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
                for call, result in tool_outcomes:
                    batch.append(
                        PendingMessage(
                            role="tool",
                            content=_truncate_for_llm(result.to_llm_text()),
                            meta={
                                "tool_call_id": call["id"],
                                "tool_name": call["name"],
                                "is_error": result.is_error,
                                "error_type": result.error_type,
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
            self._state_machine.fail(f"exceeded {MAX_ITERATIONS} iterations")
            await self._store.append_message(
                session_id,
                "assistant",
                f"(已达到工具循环上限 {MAX_ITERATIONS} 次,放弃后续调用。)",
            )
            yield {
                "event": "error",
                "data": {
                    "message": f"agent loop exceeded {MAX_ITERATIONS} iterations",
                    "state": self._state_machine.to_dict(),
                },
            }
        except Exception:
            self._state_machine.fail("unexpected exception")
            raise
        finally:
            if self._state_machine.state not in (
                AgentState.COMPLETED,
                AgentState.ERROR,
                AgentState.CANCELLED,
            ):
                self._state_machine.complete("converse ended")
            unbind_notify_queue(nq_token)
            prts_reset(ctx_token)

    async def _auto_remember(
        self,
        session_id: str,
        user_content: str,
        assistant_text: str,
        channel: str,
    ) -> None:
        """把本轮对话向量化后写入向量存储。失败只打日志,不阻塞 SSE。"""
        if self._embedding is None:
            return
        try:
            text = f"[{channel}] User: {user_content}\nAssistant: {assistant_text}"
            vec = await self._embedding.embed(text)
            mem_id = f"{session_id}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
            await self._tools.invoke(
                "prts-vector__upsert",
                {
                    "id": mem_id,
                    "vector": vec,
                    "payload": {
                        "session_id": session_id,
                        "channel": channel,
                        "text": text,
                    },
                },
            )
            logger.debug("auto-remember %s ok", mem_id)
        except Exception:
            logger.exception("auto-remember failed")

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
