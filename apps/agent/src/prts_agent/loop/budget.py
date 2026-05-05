"""动态 Token 预算管理。

P8: 根据历史 usage 模式和模型自报告,动态调整上下文窗口的 headroom,
避免 tool 密集场景下固定 80% headroom 导致超预算。
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..llm.base import LlmClient, TokenUsage

logger = logging.getLogger(__name__)

_DEFAULT_HEADROOM = 0.80
_TOOL_HEAVY_THRESHOLD = 0.30
_FAST_GROWTH_THRESHOLD = 0.30
_MAX_HISTORY = 10


class DynamicBudget:
    """根据最近 usage 动态建议 token 预算。"""

    def __init__(self, llm: LlmClient) -> None:
        self._llm = llm
        self._history: deque[TokenUsage] = deque(maxlen=_MAX_HISTORY)

    def record(self, usage: TokenUsage) -> None:
        """记录一轮对话的 usage。"""
        self._history.append(usage)

    def get_budget(self) -> int:
        """返回建议的 prompt token 预算(不含 completion)。

        策略:
        - 无历史:默认 80% headroom
        - 最近 prompt_tokens 快速增长:降到 65%(tool 密集)
        - 中等增长:降到 75%
        - 平稳:升到 85%
        """
        base_limit = self._llm.context_limit

        if len(self._history) < 3:
            return int(base_limit * _DEFAULT_HEADROOM)

        # 计算最近 prompt token 增长趋势
        prompt_tokens = [u.prompt_tokens for u in self._history if u.prompt_tokens > 0]
        if len(prompt_tokens) < 3:
            return int(base_limit * _DEFAULT_HEADROOM)

        # 最近 3 轮的增长率
        recent = prompt_tokens[-1]
        previous = prompt_tokens[-3]
        growth = (recent - previous) / max(previous, 1)

        if growth > _FAST_GROWTH_THRESHOLD:
            headroom = 0.65
            reason = f"fast growth ({growth:.1%})"
        elif growth > _FAST_GROWTH_THRESHOLD / 2:
            headroom = 0.75
            reason = f"moderate growth ({growth:.1%})"
        else:
            headroom = 0.85
            reason = f"stable ({growth:.1%})"

        budget = int(base_limit * headroom)
        logger.debug(
            "dynamic budget: limit=%d headroom=%.0f%% budget=%d (%s)",
            base_limit,
            headroom * 100,
            budget,
            reason,
        )
        return budget

    def get_stats(self) -> dict[str, object]:
        """返回预算管理器的统计信息(供 /health 使用)。"""
        if not self._history:
            return {"headroom": _DEFAULT_HEADROOM, "records": 0}

        prompt_tokens = [u.prompt_tokens for u in self._history]
        return {
            "headroom": self.get_budget() / max(self._llm.context_limit, 1),
            "records": len(self._history),
            "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens),
            "max_prompt_tokens": max(prompt_tokens),
        }
