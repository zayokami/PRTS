"""对话摘要器 —— 用 LLM 生成中期记忆卡片。

P8 核心组件:当对话积累到一定轮数时,自动总结历史生成结构化摘要,
供 ContextManager 注入 system prompt,替代被截断的早期消息。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..llm.base import LlmClient

logger = logging.getLogger(__name__)


@dataclass
class MemoryCard:
    """结构化对话摘要。"""

    id: str
    session_id: str
    summary: str
    key_facts: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    todos: list[str] = field(default_factory=list)
    message_range: tuple[int, int] = (0, 0)
    importance: float = 0.5
    created_at: str = ""


class DialogueSummarizer:
    """用 LLM 总结对话历史,生成结构化记忆卡片。"""

    SUMMARY_PROMPT = """请总结以下对话片段。要求:
1. 用 2-3 句话概括主要内容
2. 列出关键事实(如用户偏好、项目信息、技术决策)
3. 列出用户明确做出的决策或选择
4. 列出待办事项或未完成的事项
5. 评估重要性(0-1,0.8+ 表示非常重要)

返回严格 JSON 格式,不要 Markdown 代码块:
{
  "summary": "string",
  "key_facts": ["string"],
  "decisions": ["string"],
  "todos": ["string"],
  "importance": 0.0
}"""

    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm

    async def summarize(
        self,
        session_id: str,
        messages: list[Any],  # StoredMessage 列表
    ) -> MemoryCard:
        """总结一组消息,返回 MemoryCard。

        参数:
            session_id: 会话 ID
            messages: StoredMessage 列表,要求至少 2 条

        返回:
            MemoryCard,包含结构化摘要
        """
        if len(messages) < 2:
            raise ValueError(f"至少需要 2 条消息才能总结,实际 {len(messages)}")

        # 构造总结输入
        summary_input = self._format_messages(messages)

        try:
            result = await self._llm.chat(
                [
                    {"role": "system", "content": self.SUMMARY_PROMPT},
                    {"role": "user", "content": summary_input},
                ]
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM 总结失败,回退到简单摘要")
            # 回退:简单拼接前 3 条用户消息作为摘要
            return self._fallback_summary(session_id, messages)

        # 解析 JSON
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            logger.warning("LLM 返回非 JSON,尝试提取 JSON 子串")
            parsed = self._extract_json(result)

        if not isinstance(parsed, dict):
            logger.warning("LLM 返回非对象,回退")
            return self._fallback_summary(session_id, messages)

        # 提取消息 ID 范围
        msg_ids = [getattr(m, "id", 0) for m in messages]
        message_start = min(msg_ids) if msg_ids else 0
        message_end = max(msg_ids) if msg_ids else 0

        # 重要性校验
        importance = float(parsed.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        card = MemoryCard(
            id=f"sum-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            summary=str(parsed.get("summary", "")),
            key_facts=parsed.get("key_facts") or [],
            decisions=parsed.get("decisions") or [],
            todos=parsed.get("todos") or [],
            message_range=(message_start, message_end),
            importance=importance,
        )

        logger.info(
            "generated summary for session=%s messages=%d-%d importance=%.2f",
            session_id,
            message_start,
            message_end,
            importance,
        )
        return card

    def _format_messages(self, messages: list[Any]) -> str:
        """把 StoredMessage 列表格式化成 LLM 可读的文本。"""
        parts: list[str] = []
        for i, m in enumerate(messages, 1):
            role = getattr(m, "role", "unknown")
            content = getattr(m, "content", "")
            # 截断过长的内容(避免烧 token)
            if len(content) > 2000:
                content = content[:2000] + "\n[... truncated]"
            parts.append(f"[{i}] {role}: {content}")
        return "\n\n".join(parts)

    def _fallback_summary(
        self, session_id: str, messages: list[Any]
    ) -> MemoryCard:
        """LLM 失败时的回退摘要。"""
        user_msgs = [
            m.content for m in messages if getattr(m, "role", "") == "user"
        ]
        summary = "对话摘要(自动提取): " + "; ".join(user_msgs[:3])

        msg_ids = [getattr(m, "id", 0) for m in messages]
        return MemoryCard(
            id=f"sum-{uuid.uuid4().hex[:12]}",
            session_id=session_id,
            summary=summary,
            key_facts=[],
            decisions=[],
            todos=[],
            message_range=(min(msg_ids) if msg_ids else 0, max(msg_ids) if msg_ids else 0),
            importance=0.3,
        )

    def _extract_json(self, text: str) -> dict[str, Any]:
        """从可能包含 Markdown 代码块或多余文本的字符串中提取 JSON。"""
        # 尝试找 ```json ... ``` 块
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # 尝试找 ``` ... ``` 块
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                try:
                    return json.loads(text[start:end].strip())
                except json.JSONDecodeError:
                    pass

        # 尝试找 { ... } 大括号包裹的部分
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                return json.loads(text[brace_start : brace_end + 1])
            except json.JSONDecodeError:
                pass

        return {}

    def format_for_system(self, cards: list[MemoryCard]) -> str:
        """把 MemoryCard 列表格式化成 system prompt 段落。"""
        if not cards:
            return ""

        parts: list[str] = ["## 历史摘要"]
        for i, card in enumerate(cards, 1):
            parts.append(f"\n### 摘要 {i} (重要性: {card.importance:.0%})")
            parts.append(card.summary)
            if card.key_facts:
                parts.append("关键事实:")
                for fact in card.key_facts:
                    parts.append(f"  - {fact}")
            if card.decisions:
                parts.append("已做决策:")
                for dec in card.decisions:
                    parts.append(f"  - {dec}")
            if card.todos:
                parts.append("待办事项:")
                for todo in card.todos:
                    parts.append(f"  - {todo}")

        return "\n".join(parts)
