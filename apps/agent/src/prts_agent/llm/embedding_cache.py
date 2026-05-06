"""Embedding 结果缓存 —— LRU + TTL,避免重复 API 调用。

典型场景:同一 session 的连续多轮对话中,system prompt 和最近几条
历史消息会反复被 embed 用于向量召回。缓存后可将这部分开销降到接近 0。

设计:
- LRU: OrderedDict,超限淘汰最久未用
- TTL: 每个 entry 带 expire_at 时间戳,过期视为 miss
- 并发安全: asyncio.Lock (单进程),无需外部存储(向量本身不可序列化)
- 大小: 默认 1000 条,约 1000 × 1536 × 4B ≈ 6MB (text-embedding-3-small)
"""

from __future__ import annotations

import asyncio
import collections
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ENTRIES = 1000
_DEFAULT_TTL_SECONDS = 3600  # 1 小时


class EmbeddingCache:
    """线程安全的 embedding 结果缓存。"""

    def __init__(
        self,
        max_entries: int = _DEFAULT_MAX_ENTRIES,
        ttl_seconds: float = _DEFAULT_TTL_SECONDS,
    ) -> None:
        self._max_entries = max_entries
        self._ttl = ttl_seconds
        # OrderedDict: key -> (vector, expire_at)
        self._cache: collections.OrderedDict[str, tuple[list[float], float]] = (
            collections.OrderedDict()
        )
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        total = self._hits + self._misses
        return {
            "size": self.size,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "evictions": self._evictions,
        }

    async def get(self, text: str) -> list[float] | None:
        """从缓存读取向量。返回 None 表示未命中或已过期。"""
        async with self._lock:
            entry = self._cache.get(text)
            if entry is None:
                self._misses += 1
                return None

            vector, expire_at = entry
            if time.monotonic() > expire_at:
                # TTL 过期:删除并视为 miss
                del self._cache[text]
                self._misses += 1
                return None

            # LRU: 移到末尾(最近使用)
            self._cache.move_to_end(text)
            self._hits += 1
            return list(vector)  # 返回副本,防止外部修改影响缓存

    async def set(self, text: str, vector: list[float]) -> None:
        """写入缓存。如果已满,淘汰最久未用的 entry。"""
        async with self._lock:
            expire_at = time.monotonic() + self._ttl

            if text in self._cache:
                # 更新已存在的 entry
                self._cache.move_to_end(text)
                self._cache[text] = (vector, expire_at)
                return

            # 需要插入新 entry
            if len(self._cache) >= self._max_entries:
                # 淘汰最久未用的
                evicted = self._cache.popitem(last=False)
                self._evictions += 1
                logger.debug(
                    "embedding cache evicted oldest entry (hits=%d)",
                    self._evictions,
                )

            self._cache[text] = (vector, expire_at)

    async def get_batch(
        self, texts: list[str]
    ) -> tuple[list[list[float] | None], list[str]]:
        """批量读取缓存。

        返回:
            (results, misses): results 是与 texts 等长的列表,未命中为 None;
            misses 是未命中的 text 列表(保持原始顺序)
        """
        results: list[list[float] | None] = []
        misses: list[str] = []

        async with self._lock:
            now = time.monotonic()
            for text in texts:
                entry = self._cache.get(text)
                if entry is None:
                    self._misses += 1
                    results.append(None)
                    misses.append(text)
                    continue

                vector, expire_at = entry
                if now > expire_at:
                    del self._cache[text]
                    self._misses += 1
                    results.append(None)
                    misses.append(text)
                    continue

                self._cache.move_to_end(text)
                self._hits += 1
                results.append(list(vector))  # 返回副本

        return results, misses

    async def set_batch(self, texts: list[str], vectors: list[list[float]]) -> None:
        """批量写入缓存。texts 和 vectors 必须等长。"""
        if len(texts) != len(vectors):
            raise ValueError("texts and vectors must have same length")

        async with self._lock:
            expire_at = time.monotonic() + self._ttl
            for text, vector in zip(texts, vectors):
                if text in self._cache:
                    self._cache.move_to_end(text)
                    self._cache[text] = (vector, expire_at)
                else:
                    if len(self._cache) >= self._max_entries:
                        self._cache.popitem(last=False)
                        self._evictions += 1
                    self._cache[text] = (vector, expire_at)

    async def invalidate(self, text: str) -> bool:
        """主动删除某个 key。返回是否命中。"""
        async with self._lock:
            if text in self._cache:
                del self._cache[text]
                return True
            return False

    async def clear(self) -> None:
        """清空缓存。"""
        async with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
            self._evictions = 0
