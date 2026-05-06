"""上下文管理器 —— 三层记忆组装器。

P8 核心组件:统筹短期上下文(最近消息)、中期记忆(摘要)、长期记忆(向量召回)
的组装与预算控制。

组装优先级(越靠前越保留):
1. System Prompt(含中期摘要)
2. 最近 N 条消息(保证多轮连贯)
3. 相关向量召回(跨时空关联)
4. 早期消息(最优先丢弃)
"""

from __future__ import annotations

import collections
import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.base import ChatMessage, EmbeddingClient, LlmClient
    from ..llm.tokenizer import TokenUsage
    from ..memory.sqlite import SqliteStore, StoredMessage
    from ..memory.summarizer import DialogueSummarizer

    from .budget import DynamicBudget

logger = logging.getLogger(__name__)

# 短期窗口:保证多轮工具调用的连贯性
DEFAULT_RECENT_WINDOW = 20
# 摘要触发阈值:超过此轮数未摘要时生成新摘要
DEFAULT_SUMMARY_INTERVAL = 10
# 向量召回 top-k
DEFAULT_VECTOR_TOPK = 5
# 最小保留对话轮次 (user+assistant = 1 轮)
MIN_RECENT_PAIRS = 4

import os

_CONTEXT_MODE = os.getenv("PRTS_CONTEXT_MODE", "smart").lower()
_SUMMARY_INTERVAL = int(os.getenv("PRTS_SUMMARY_INTERVAL", str(DEFAULT_SUMMARY_INTERVAL)))


class ContextManager:
    """三层记忆上下文组装器。"""

    # 向量召回结果缓存 (P1): query_text → recalled_results, TTL 5min
    _VECTOR_RECALL_TTL = 300  # 5 分钟
    _VECTOR_RECALL_MAX = 200
    # 上下文组装缓存 (P1): build_context 结果缓存, TTL 10s
    _CONTEXT_CACHE_TTL = 10  # 10 秒
    _CONTEXT_CACHE_MAX = 50

    def __init__(
        self,
        store: SqliteStore,
        llm: LlmClient,
        summarizer: DialogueSummarizer,
        budget: DynamicBudget,
        embedding: EmbeddingClient | None = None,
        tools: "ToolRegistry | None" = None,
        *,
        recent_window: int = DEFAULT_RECENT_WINDOW,
        summary_interval: int = DEFAULT_SUMMARY_INTERVAL,
        vector_topk: int = DEFAULT_VECTOR_TOPK,
    ) -> None:
        self._store = store
        self._llm = llm
        self._summarizer = summarizer
        self._budget = budget
        self._embedding = embedding
        self._tools = tools
        self._recent_window = recent_window
        self._summary_interval = summary_interval
        self._vector_topk = vector_topk
        # 向量召回结果缓存: key=hash(session_id+query), value=(results, expire_at)
        self._recall_cache: collections.OrderedDict[str, tuple[list[str], float]] = (
            collections.OrderedDict()
        )
        self._recall_hits = 0
        self._recall_misses = 0
        # 上下文组装缓存: key=hash(session+user+system+last_msg_id), value=(messages, expire_at)
        self._context_cache: collections.OrderedDict[str, tuple[list, float]] = (
            collections.OrderedDict()
        )
        self._context_hits = 0
        self._context_misses = 0

    async def build_context(
        self,
        session_id: str,
        user_content: str,
        system_prompt: str,
    ) -> list[ChatMessage]:
        """为 LLM 调用组装上下文消息列表。

        返回:
            符合 OpenAI 风格的 ChatMessage 列表,已截断到预算内。
        """
        # PRTS_CONTEXT_MODE=legacy 时回退到旧模式(仅最近消息+固定预算)
        if _CONTEXT_MODE == "legacy":
            from ..llm.tokenizer import count_messages_tokens

            recent = await self._store.history(session_id, limit=self._recent_window)
            messages: list[ChatMessage] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            for m in recent:
                msg: ChatMessage = {"role": m.role, "content": m.content}
                if m.meta:
                    if m.meta.get("tool_calls"):
                        msg["tool_calls"] = m.meta["tool_calls"]
                    if m.meta.get("tool_call_id"):
                        msg["tool_call_id"] = m.meta["tool_call_id"]
                    if m.meta.get("tool_name"):
                        msg["name"] = m.meta["tool_name"]
                messages.append(msg)
            budget = int(self._llm.context_limit * 0.80)
            return self._truncate_to_budget(messages, budget, system_prompt)

        # ---- 1. 先获取最近消息(用于缓存 key) ----
        recent = await self._store.history(session_id, limit=self._recent_window)
        last_msg_id = str(getattr(recent[-1], "id", 0)) if recent else "0"

        # ---- 2. 查上下文缓存 ----
        cache_key = hashlib.sha256(
            f"{session_id}:{user_content}:{system_prompt}:{last_msg_id}".encode("utf-8")
        ).hexdigest()
        cached = self._context_cache.get(cache_key)
        if cached is not None:
            messages, expire_at = cached
            if time.monotonic() <= expire_at:
                self._context_cache.move_to_end(cache_key)
                self._context_hits += 1
                logger.debug(
                    "context cache hit for session=%s (hits=%d, misses=%d)",
                    session_id,
                    self._context_hits,
                    self._context_misses,
                )
                return list(messages)
            del self._context_cache[cache_key]

        # ---- 3. 获取动态预算 ----
        budget = self._budget.get_budget()

        # ---- 4. 获取中期摘要 ----
        summary_text = await self._load_summaries(session_id)

        # ---- 5. 向量召回 ----
        recalled = await self._recall_vectors(session_id, user_content)

        # ---- 6. 组装消息 ----
        messages = self._assemble(
            system_prompt=system_prompt,
            summary_text=summary_text,
            recent=recent,
            recalled=recalled,
        )

        # 6. 检查是否需要生成新摘要
        await self._maybe_summarize(session_id, recent)

        # 7. 截断到预算
        truncated = self._truncate_to_budget(messages, budget, system_prompt)

        # 8. 记录 token 使用(如果有上一次的 usage)
        if self._llm.last_usage:
            self._budget.record(self._llm.last_usage)

        # 9. 写上下文缓存
        if len(self._context_cache) >= self._CONTEXT_CACHE_MAX:
            self._context_cache.popitem(last=False)
        self._context_cache[cache_key] = (truncated, time.monotonic() + self._CONTEXT_CACHE_TTL)
        self._context_misses += 1

        return truncated

    async def _load_summaries(self, session_id: str) -> str:
        """加载并格式化会话摘要。"""
        cards = await self._store.get_summaries(session_id, limit=3)
        if not cards:
            return ""
        return self._summarizer.format_for_system(
            [self._dict_to_card(c) for c in cards]
        )

    def _dict_to_card(self, d: dict[str, Any]) -> Any:  # MemoryCard
        from ..memory.summarizer import MemoryCard

        return MemoryCard(
            id=d["id"],
            session_id=d["session_id"],
            summary=d["summary"],
            key_facts=d.get("key_facts", []),
            decisions=d.get("decisions", []),
            todos=d.get("todos", []),
            message_range=(d["message_start"], d["message_end"]),
            importance=d.get("importance", 0.5),
            created_at=d.get("created_at", ""),
        )

    async def _recall_vectors(
        self,
        session_id: str,
        user_content: str,
    ) -> list[str]:
        """向量召回相关历史。

        通过 prts-vector__search 工具查询与当前问题语义相近的历史记录。
        结果缓存 5 分钟,避免同一 query 的重复 embed + 搜索。
        """
        if self._embedding is None or self._tools is None:
            return []

        # ---- 1. 查缓存 ----
        cache_key = hashlib.sha256(
            f"{session_id}:{user_content}".encode("utf-8")
        ).hexdigest()
        cached = self._recall_cache.get(cache_key)
        if cached is not None:
            results, expire_at = cached
            if time.monotonic() <= expire_at:
                self._recall_cache.move_to_end(cache_key)
                self._recall_hits += 1
                logger.debug(
                    "vector recall cache hit for session=%s (hits=%d, misses=%d)",
                    session_id,
                    self._recall_hits,
                    self._recall_misses,
                )
                return list(results)
            # TTL 过期:删除
            del self._recall_cache[cache_key]

        # ---- 2. 执行召回 ----
        try:
            vec = await self._embedding.embed(user_content)
            if not vec:
                # Embedding API 不可用(如 DeepSeek 返回 404),跳过向量召回
                return []
            raw = await self._tools.invoke(
                "prts-vector__search",
                {"query_vector": vec, "top_k": self._vector_topk},
            )

            texts: list[str] = []
            if isinstance(raw, str):
                raw = json.loads(raw)
            if isinstance(raw, dict) and raw.get("ok"):
                results = raw.get("results", [])
                for r in results:
                    payload_str = r.get("payload")
                    if payload_str:
                        try:
                            payload = json.loads(payload_str)
                            text = payload.get("text", "")
                        except (TypeError, ValueError):
                            text = payload_str
                        if text:
                            texts.append(text)

            # ---- 3. 写缓存 ----
            if len(self._recall_cache) >= self._VECTOR_RECALL_MAX:
                self._recall_cache.popitem(last=False)
            self._recall_cache[cache_key] = (texts, time.monotonic() + self._VECTOR_RECALL_TTL)
            self._recall_misses += 1

            logger.debug(
                "vector recall: %d results for session=%s (cached, hits=%d, misses=%d)",
                len(texts),
                session_id,
                self._recall_hits,
                self._recall_misses,
            )
            return texts
        except KeyError:
            # prts-vector__search 工具未注册(MCP server 未启用)
            logger.debug("prts-vector__search not available, skipping vector recall")
        except Exception:  # noqa: BLE001
            logger.exception("vector recall failed")
        return []

    def _assemble(
        self,
        system_prompt: str,
        summary_text: str,
        recent: list[StoredMessage],
        recalled: list[str],
    ) -> list[ChatMessage]:
        """组装三层记忆为消息列表。"""
        messages: list[ChatMessage] = []

        # System prompt:基础提示 + 摘要 + 向量召回
        system_parts: list[str] = []
        if system_prompt:
            system_parts.append(system_prompt)
        if summary_text:
            system_parts.append(summary_text)
        if recalled:
            system_parts.append("相关历史:\n" + "\n".join(f"- {r}" for r in recalled))

        if system_parts:
            messages.append(
                {"role": "system", "content": "\n\n".join(system_parts)}
            )

        # 最近消息
        for m in recent:
            # DeepSeek 要求 assistant 消息有非空 content
            content = m.content if m.content and m.content.strip() else "(调用工具中...)"
            msg: ChatMessage = {"role": m.role, "content": content}
            if m.meta:
                if m.meta.get("tool_calls"):
                    # OpenAI 格式要求每个 tool_call 必须有 type="function"
                    msg["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("arguments", {}), ensure_ascii=False),
                            },
                        }
                        for tc in m.meta["tool_calls"]
                    ]
                # DeepSeek V4: 传递 reasoning_content 用于多轮对话
                if m.meta.get("reasoning_content"):
                    msg["reasoning_content"] = m.meta["reasoning_content"]
                if m.meta.get("tool_call_id"):
                    msg["tool_call_id"] = m.meta["tool_call_id"]
                if m.meta.get("tool_name"):
                    msg["name"] = m.meta["tool_name"]
            messages.append(msg)

        return messages

    async def _maybe_summarize(
        self,
        session_id: str,
        recent: list[StoredMessage],
    ) -> None:
        """如果积累足够多未摘要消息,生成新摘要。

        PRTS_SUMMARY_INTERVAL=0 时关闭自动摘要。
        """
        if _SUMMARY_INTERVAL <= 0:
            return

        last_summary_end = await self._store.get_last_summary_end(session_id)

        # 找出未被摘要覆盖的消息
        new_messages = [m for m in recent if getattr(m, "id", 0) > last_summary_end]

        if len(new_messages) >= self._summary_interval:
            logger.info(
                "triggering summary for session=%s: %d new messages since %d",
                session_id,
                len(new_messages),
                last_summary_end,
            )
            try:
                card = await self._summarizer.summarize(session_id, new_messages)
                await self._store.save_summary(
                    session_id=card.session_id,
                    summary_id=card.id,
                    summary=card.summary,
                    message_start=card.message_range[0],
                    message_end=card.message_range[1],
                    key_facts=card.key_facts,
                    decisions=card.decisions,
                    todos=card.todos,
                    importance=card.importance,
                )
                # 清理旧摘要,只保留最近 10 条
                deleted = await self._store.delete_old_summaries(session_id, keep=10)
                if deleted:
                    logger.debug("cleaned %d old summaries", deleted)
            except Exception:  # noqa: BLE001
                logger.exception("auto-summary failed")

    def _truncate_to_budget(
        self,
        messages: list[ChatMessage],
        budget: int,
        base_system: str,
    ) -> list[ChatMessage]:
        """截断消息列表到 token 预算内,优先保留高分消息。"""
        from ..llm.tokenizer import count_messages_tokens
        from ..memory.importance import ImportanceScorer

        total = count_messages_tokens(messages)
        if total <= budget:
            return messages

        logger.warning(
            "messages token count %d > budget %d, truncating", total, budget
        )

        # 分离 system 与对话消息
        system_msgs = [m for m in messages if m.get("role") == "system"]
        chat_msgs = [m for m in messages if m.get("role") != "system"]

        # 步骤 1: 尝试去掉 system 中的摘要,只保留 base_system
        trimmed_system: list[ChatMessage] = []
        if base_system:
            trimmed_system = [{"role": "system", "content": base_system}]

        if system_msgs and trimmed_system:
            candidate = trimmed_system + chat_msgs
            if count_messages_tokens(candidate) <= budget:
                return candidate

        # 步骤 2: 使用重要性评分智能截断
        # 先给所有 chat 消息评分
        scorer = ImportanceScorer()
        scored = scorer.score_batch(chat_msgs)

        # 按重要性排序(高到低),但保留最近 MIN_RECENT_PAIRS 轮不被丢弃
        min_keep = MIN_RECENT_PAIRS * 2
        recent_msgs = chat_msgs[-min_keep:] if len(chat_msgs) > min_keep else chat_msgs
        old_msgs = chat_msgs[:-min_keep] if len(chat_msgs) > min_keep else []

        # 对旧消息按重要性排序
        old_scored = [(m, s) for m, s in scored[:-min_keep]] if len(scored) > min_keep else []
        old_scored.sort(key=lambda x: x[1], reverse=True)

        # 逐步丢弃低分旧消息,直到预算内
        base_system_msgs = trimmed_system if trimmed_system else system_msgs
        keep_ids = {id(m) for m in recent_msgs}
        keep_ids.update(id(m) for m, _ in old_scored)

        for m, _ in reversed(old_scored):  # 从最低分开始丢弃
            test_keep = keep_ids - {id(m)}
            test_msgs = [m for m in chat_msgs if id(m) in test_keep]
            candidate = base_system_msgs + test_msgs
            if count_messages_tokens(candidate) <= budget:
                keep_ids = test_keep
            else:
                break

        final_kept = [m for m in chat_msgs if id(m) in keep_ids]
        final = base_system_msgs + final_kept
        if count_messages_tokens(final) <= budget:
            dropped = len(chat_msgs) - len(final_kept)
            logger.info(
                "smart-truncated from %d to %d messages (dropped %d low-importance)",
                len(chat_msgs),
                len(final_kept),
                dropped,
            )
            return final

        # 步骤 3: 即使智能截断也超预算 ——  fallback 到时间截断
        for drop in range(max(0, len(chat_msgs) - min_keep + 1)):
            kept = chat_msgs[drop:]
            candidate = base_system_msgs + kept
            if count_messages_tokens(candidate) <= budget:
                logger.info(
                    "time-truncated from %d to %d messages (dropped oldest %d)",
                    len(chat_msgs),
                    len(kept),
                    drop,
                )
                return candidate

        # 步骤 4: 强行只保留 system + 最近 1 轮
        fallback = base_system_msgs + chat_msgs[-2:]
        if len(chat_msgs) >= 2 and count_messages_tokens(fallback) <= budget:
            return fallback

        # 步骤 5: 最后的最后 —— 只保留 system prompt
        last_ditch = base_system_msgs
        logger.warning(
            "severe context overflow: only system prompt kept (%d tokens)",
            count_messages_tokens(last_ditch),
        )
        return last_ditch
