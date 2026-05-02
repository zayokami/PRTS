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
)
from ..llm.anthropic_client import AnthropicLlmClient
from ..llm.embedding import EmbeddingClient
from ..llm.tokenizer import count_messages_tokens
from ..memory import SqliteStore
from ..memory.sqlite import PendingMessage
from ..runtime import bind_notify_queue, unbind_notify_queue
from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 8
# 短期即时上下文条数。向量检索负责召回跨时间/跨 session 的相关历史,
# 这里只保留最近 N 条保证多轮工具调用的连贯性。
RECENT_WINDOW = 20
VECTOR_TOPK = 5
# 单个 tool 结果最大字符数 ≈ 4k token。超过会截断,避免一次 ``filesystem__read_text_file``
# 把整篇大文件灌进 history,后续每轮都重发一遍把 prompt cache 顶飞 + 吃光上下文。
# 截断标记里告诉 LLM 完整大小,必要时它能用更窄的参数(行号区间、关键词)重调。
MAX_TOOL_RESULT_CHARS = 16000

# Token 预算:只用上下文窗口的 80%,留 20% headroom 给输出 + 安全余量。
# 自研 count_tokens 是保守估计,headroom 还能抵消计数误差。
TOKEN_HEADROOM = 0.80
# 截断时至少保留的对话轮次 (user+assistant = 1 轮)
MIN_RECENT_PAIRS = 4


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

    正常路径 invoker 已经把 MCP ``CallToolResult`` 拆开了,这里只剩 str / dict / list /
    None 等普通值。但如果有人(单测、未来新接入的别的工具协议)直接把原始
    ``CallToolResult`` 丢进来,``json.dumps`` 会忽略它的 ``content`` / ``structuredContent``
    给一个空 ``{}``,LLM 就会拿到空字符串瞎猜。下面那段防御性分支兜底。
    """
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
            for iteration in range(MAX_ITERATIONS):
                messages = await self._build_messages(
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
                            content=_truncate_for_llm(_serialize_tool_result(result)),
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

    async def _build_messages(
        self,
        session_id: str,
        user_content: str,
        system_prompt: str,
    ) -> list[ChatMessage]:
        """混合上下文构造: 最近 N 条保证连贯 + 向量召回跨时空相关历史。

        构造完成后会做一次 token 预算检查。若超出 ``context_limit * TOKEN_HEADROOM``
        则按以下优先级丢弃内容,直到落回预算内:

        1. 去掉 system prompt 中的向量召回段落(可选内容)。
        2. 从 oldest chat 消息开始丢弃,至少保留 ``MIN_RECENT_PAIRS`` 轮对话。
        """
        recent = await self._store.history(session_id, limit=RECENT_WINDOW)
        budget = int(self._llm.context_limit * TOKEN_HEADROOM)

        # ---- 向量召回(可选) ----
        recalled_texts: list[str] = []
        if self._embedding is not None:
            try:
                vec = await self._embedding.embed(user_content)
                raw = await self._tools.invoke(
                    "prts-vector__search",
                    {"query_vector": vec, "top_k": VECTOR_TOPK},
                )
                if isinstance(raw, str):
                    raw = json.loads(raw)
                if isinstance(raw, dict) and raw.get("ok"):
                    for r in raw.get("results", []):
                        payload_str = r.get("payload")
                        if payload_str:
                            try:
                                payload = json.loads(payload_str)
                                text = payload.get("text", "")
                            except (TypeError, ValueError):
                                text = payload_str
                            if text:
                                recalled_texts.append(text)
            except Exception:
                logger.exception("vector recall failed")

        # ---- 组装 system prompt ----
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if recalled_texts:
            seen: set[str] = set()
            unique_lines: list[str] = []
            for t in recalled_texts:
                if t not in seen:
                    seen.add(t)
                    unique_lines.append(t)
            system_parts.append(
                "以下是与当前问题相关的历史回忆:\n"
                + "\n".join(f"- {line}" for line in unique_lines)
            )

        full_system = "\n\n".join(system_parts)
        chat_history = _stored_to_chat(recent)

        messages: list[ChatMessage] = []
        if full_system:
            messages.append({"role": "system", "content": full_system})
        messages.extend(chat_history)

        # ---- token 预算检查与截断 ----
        total = count_messages_tokens(messages)
        if total > budget:
            logger.warning(
                "messages token count %d > budget %d (limit=%d * %.0f%%), truncating",
                total,
                budget,
                self._llm.context_limit,
                TOKEN_HEADROOM * 100,
            )
            messages = self._truncate_messages_to_budget(
                messages,
                budget,
                base_system=system_prompt,
            )
            new_total = count_messages_tokens(messages)
            logger.info(
                "truncated from %d to %d tokens (%d messages retained)",
                total,
                new_total,
                len(messages),
            )

        return messages

    def _truncate_messages_to_budget(
        self,
        messages: list[ChatMessage],
        budget: int,
        base_system: str,
    ) -> list[ChatMessage]:
        """逐步截断消息列表,直到 token 数 ≤ budget。"""
        # 分离 system 与对话消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]

        # 提前准备好 "精简版 system"(不含召回段落),供后续步骤复用
        trimmed_system: list[ChatMessage] = []
        if base_system:
            trimmed_system = [{"role": "system", "content": base_system}]

        # 步骤 1: 尝试去掉 system 中的召回段落,只保留 base_system
        if system_msgs and trimmed_system:
            candidate = trimmed_system + chat_msgs
            if count_messages_tokens(candidate) <= budget:
                return candidate

        # 步骤 2: 从 oldest 开始丢弃 chat 消息
        # 1 轮对话 = user + assistant (2 条); tool 消息也算在内。
        # 为了简单且安全,直接按条数丢弃,至少保留 MIN_RECENT_PAIRS*2 条。
        min_keep = MIN_RECENT_PAIRS * 2
        base_system_msgs = trimmed_system if trimmed_system else system_msgs
        for drop in range(max(0, len(chat_msgs) - min_keep + 1)):
            kept = chat_msgs[drop:]
            candidate = base_system_msgs + kept
            if count_messages_tokens(candidate) <= budget:
                return candidate

        # 步骤 3: 即使只剩最小集合也超预算 —— 强行只保留 system + 最近 1 轮
        # (2 条 chat)。这种情况通常意味着单条消息极长或 system prompt 本身超预算。
        fallback = base_system_msgs + chat_msgs[-2:]
        if len(chat_msgs) >= 2 and count_messages_tokens(fallback) <= budget:
            return fallback

        # 步骤 4: 最后的最后 —— 只保留 system prompt(让 LLM 至少知道角色)。
        last_ditch = base_system_msgs
        logger.warning(
            "severe context overflow: only system prompt kept (%d tokens)",
            count_messages_tokens(last_ditch),
        )
        return last_ditch

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
