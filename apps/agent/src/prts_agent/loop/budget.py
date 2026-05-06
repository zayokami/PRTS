"""动态 Token 预算管理。

P8: 根据历史 usage 模式和模型自报告,动态调整上下文窗口的 headroom,
避免 tool 密集场景下固定 80% headroom 导致超预算。
"""

from __future__ import annotations

import asyncio
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
    """根据最近 usage 动态建议 token 预算。

    P9: 支持将预算历史持久化到 SQLite,进程重启后可恢复长期模式。
    """

    def __init__(self, llm: LlmClient, store: Any = None) -> None:
        self._llm = llm
        self._store = store
        self._history: deque[TokenUsage] = deque(maxlen=_MAX_HISTORY)
        # Async load budget history from SQLite on startup for cross-session learning
        if store is not None:
            try:
                import asyncio
                asyncio.create_task(self._load_history())
            except Exception:
                logger.exception("failed to initiate budget history loading")

    async def _load_history(self) -> None:
        """从 SQLite 加载历史 usage。"""
        try:
            model_name = getattr(self._llm, "model_name", None) or getattr(self._llm, "model", "unknown")
            rows = await self._store.load_budget_history(model_name, limit=_MAX_HISTORY)
            from ..llm.base import TokenUsage
            for prompt, completion, total in rows:
                self._history.append(
                    TokenUsage(
                        prompt_tokens=prompt,
                        completion_tokens=completion,
                        total_tokens=total,
                        model=model_name,
                    )
                )
            logger.info(
                "loaded %d budget history records for %s",
                len(rows),
                model_name,
            )
        except Exception:
            logger.exception("failed to load budget history")

    def record(self, usage: TokenUsage) -> None:
        """记录一轮对话的 usage。

        P9: 同时异步持久化到 SQLite。
        """
        self._history.append(usage)

        if self._store is not None:
            try:
                asyncio.create_task(
                    self._store.save_budget_usage(
                        model=usage.model,
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                    )
                )
            except Exception:
                logger.exception("failed to persist budget usage")

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
