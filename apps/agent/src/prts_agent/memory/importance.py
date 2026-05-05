"""消息重要性评分 —— 决定截断时保留哪些消息。

P8: 长上下文管理中,不是所有消息都同等重要。重要性评分让"用户决策"、
"关键事实"等高分消息优先保留,而"闲聊"、"重复确认"等低分消息优先丢弃。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 关键词权重表
KEYWORD_WEIGHTS: dict[str, float] = {
    # 决策类 (高权重)
    "决定": 0.30,
    "确认": 0.30,
    "选择": 0.25,
    "确定": 0.25,
    "拍板": 0.35,
    "定下来": 0.30,
    # 任务类
    "待办": 0.35,
    "TODO": 0.35,
    "任务": 0.30,
    "需要": 0.20,
    "必须": 0.25,
    "应该": 0.15,
    # 重要度
    "重要": 0.30,
    "关键": 0.30,
    "核心": 0.25,
    "主要": 0.20,
    "优先": 0.25,
    # 错误/调试
    "错误": 0.25,
    "修复": 0.25,
    "解决": 0.20,
    "bug": 0.25,
    "报错": 0.25,
    # 用户信息
    "用户说": 0.15,
    "我需要": 0.20,
    "我想要": 0.20,
    "请帮我": 0.15,
    # 否定/取消 (也重要,因为改变了状态)
    "不要": 0.20,
    "取消": 0.25,
    "删掉": 0.25,
    "移除": 0.20,
}

# 角色基础分
ROLE_BASE_SCORE: dict[str, float] = {
    "user": 0.30,
    "assistant": 0.10,
    "tool": 0.15,
    "system": 0.05,
}

# 长度惩罚阈值
LENGTH_PENALTY_THRESHOLD = 5000
LENGTH_PENALTY = 0.10

# 工具错误加分
TOOL_ERROR_BONUS = 0.20

# 连续重复惩罚
REPEAT_PENALTY = 0.15


class ImportanceScorer:
    """给消息打分,0-1 越高越重要。"""

    def score(self, message: Any) -> float:
        """评估单条消息的重要性。

        参数:
            message: 任何有 role/content/meta 属性的对象(如 StoredMessage)

        返回:
            0.0-1.0 的重要性分数
        """
        role = getattr(message, "role", "unknown")
        content = getattr(message, "content", "")
        meta = getattr(message, "meta", None) or {}

        score = 0.0

        # 1. 角色基础分
        score += ROLE_BASE_SCORE.get(role, 0.0)

        # 2. 关键词匹配
        content_lower = content.lower()
        for keyword, weight in KEYWORD_WEIGHTS.items():
            if keyword.lower() in content_lower:
                score += weight

        # 3. 长度惩罚:太长的消息通常不太关键(如大段代码、文件内容)
        if len(content) > LENGTH_PENALTY_THRESHOLD:
            score -= LENGTH_PENALTY

        # 4. 工具错误加分
        if meta.get("is_error"):
            score += TOOL_ERROR_BONUS

        # 5. 工具调用加分(assistant 发起了工具调用)
        if meta.get("tool_calls"):
            score += 0.10

        # 6. 用户明确提问加分
        if role == "user" and any(
            content.strip().endswith(p) for p in ("?", "？", "吗", "么", "呢")
        ):
            score += 0.10

        # 限制在 0-1
        return max(0.0, min(1.0, score))

    def score_batch(self, messages: list[Any]) -> list[tuple[Any, float]]:
        """批量评分,返回 (message, score) 列表。"""
        return [(m, self.score(m)) for m in messages]

    def filter_important(
        self,
        messages: list[Any],
        threshold: float = 0.3,
        top_k: int | None = None,
    ) -> list[Any]:
        """过滤出重要性超过阈值的消息。

        参数:
            messages: 消息列表
            threshold: 重要性阈值,低于此值的消息被过滤
            top_k: 如果指定,只返回前 k 条(按重要性排序)

        返回:
            重要性足够高的消息列表,保持原始顺序
        """
        scored = self.score_batch(messages)
        filtered = [(m, s) for m, s in scored if s >= threshold]

        if top_k and len(filtered) > top_k:
            # 按重要性排序取 top_k,但返回时保持原始顺序
            filtered.sort(key=lambda x: x[1], reverse=True)
            top_set = {id(m) for m, _ in filtered[:top_k]}
            return [m for m in messages if id(m) in top_set]

        return [m for m, _ in filtered]

    def get_summary(self, messages: list[Any]) -> dict[str, Any]:
        """返回评分统计摘要(供调试)。"""
        scored = self.score_batch(messages)
        scores = [s for _, s in scored]

        if not scores:
            return {"count": 0}

        return {
            "count": len(scores),
            "avg": sum(scores) / len(scores),
            "max": max(scores),
            "min": min(scores),
            "high_importance": sum(1 for s in scores if s >= 0.5),
            "low_importance": sum(1 for s in scores if s < 0.2),
        }
