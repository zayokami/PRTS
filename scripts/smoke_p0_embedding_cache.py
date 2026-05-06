"""Smoke test for Embedding Cache (P0).

验证:
1. 基本 get/set
2. TTL 过期
3. LRU 淘汰
4. 批量 get_batch/set_batch
5. 统计信息
6. 并发安全(多任务同时读写)
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "agent" / "src"))

from prts_agent.llm.embedding_cache import EmbeddingCache


async def test_basic_get_set():
    """1. 基本 get/set 命中。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    vec = [0.1, 0.2, 0.3]
    await cache.set("hello", vec)

    result = await cache.get("hello")
    assert result == vec
    assert result is not vec  # 应该返回副本

    stats = cache.stats
    assert stats["hits"] == 1
    assert stats["misses"] == 0
    print("  OK basic get/set")


async def test_miss():
    """2. 未命中返回 None。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    result = await cache.get("unknown")
    assert result is None

    stats = cache.stats
    assert stats["hits"] == 0
    assert stats["misses"] == 1
    print("  OK miss returns None")


async def test_ttl_expiration():
    """3. TTL 过期后视为 miss。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=0.05)  # 50ms TTL

    await cache.set("expires", [1.0, 2.0])
    result = await cache.get("expires")
    assert result == [1.0, 2.0]

    # 等待过期
    await asyncio.sleep(0.1)

    result = await cache.get("expires")
    assert result is None
    print("  OK TTL expiration")


async def test_lru_eviction():
    """4. LRU 淘汰:最久未用的被移除。"""
    cache = EmbeddingCache(max_entries=3, ttl_seconds=60.0)

    await cache.set("a", [1.0])
    await cache.set("b", [2.0])
    await cache.set("c", [3.0])

    # 访问 a,使其变为最近使用
    await cache.get("a")

    # 插入 d,应淘汰 b(最久未用)
    await cache.set("d", [4.0])

    assert await cache.get("a") is not None
    assert await cache.get("b") is None  # 被淘汰
    assert await cache.get("c") is not None
    assert await cache.get("d") is not None

    stats = cache.stats
    assert stats["evictions"] == 1
    print("  OK LRU eviction")


async def test_batch_operations():
    """5. 批量 get_batch/set_batch。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    # 先写入一些
    await cache.set_batch(["x", "y"], [[1.0], [2.0]])

    # 批量读取,部分命中部分未命中
    results, misses = await cache.get_batch(["x", "z", "y"])
    assert results[0] == [1.0]   # x 命中
    assert results[1] is None     # z 未命中
    assert results[2] == [2.0]   # y 命中
    assert misses == ["z"]

    print("  OK batch operations")


async def test_concurrent_safety():
    """6. 并发安全:多任务同时读写不冲突。"""
    cache = EmbeddingCache(max_entries=100, ttl_seconds=60.0)

    async def writer(i):
        for j in range(10):
            await cache.set(f"key_{i}_{j}", [float(i), float(j)])

    async def reader(i):
        hits = 0
        for j in range(10):
            result = await cache.get(f"key_{i}_{j}")
            if result is not None:
                hits += 1
        return hits

    # 先写入
    writers = [asyncio.create_task(writer(i)) for i in range(5)]
    await asyncio.gather(*writers)

    # 再读取
    readers = [asyncio.create_task(reader(i)) for i in range(5)]
    hit_counts = await asyncio.gather(*readers)

    # 至少有一些命中(因为写入已完成)
    total_hits = sum(hit_counts)
    assert total_hits >= 40, f"expected >=40 hits, got {total_hits}"

    stats = cache.stats
    assert stats["hits"] == total_hits
    print(f"  OK concurrent: {total_hits} hits across 5 readers")


async def test_stats():
    """7. 统计信息正确性。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    # 2 hits + 1 miss
    await cache.set("a", [1.0])
    await cache.get("a")  # hit
    await cache.get("a")  # hit
    await cache.get("b")  # miss

    stats = cache.stats
    assert stats["size"] == 1
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert abs(stats["hit_rate"] - 2 / 3) < 0.01
    print(f"  OK stats: hit_rate={stats['hit_rate']:.2%}")


async def test_invalidate():
    """8. 主动删除。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    await cache.set("to_delete", [1.0])
    assert await cache.get("to_delete") is not None

    deleted = await cache.invalidate("to_delete")
    assert deleted is True
    assert await cache.get("to_delete") is None

    not_found = await cache.invalidate("nonexistent")
    assert not_found is False
    print("  OK invalidate")


async def test_clear():
    """9. 清空缓存。"""
    cache = EmbeddingCache(max_entries=10, ttl_seconds=60.0)

    await cache.set("a", [1.0])
    await cache.set("b", [2.0])
    assert cache.size == 2

    await cache.clear()
    assert cache.size == 0
    assert await cache.get("a") is None
    assert await cache.get("b") is None

    stats = cache.stats
    assert stats["hits"] == 0
    assert stats["misses"] == 2  # clear 后 get 是 miss
    print("  OK clear")


async def main():
    print("P0 Embedding Cache smoke test")
    await test_basic_get_set()
    await test_miss()
    await test_ttl_expiration()
    await test_lru_eviction()
    await test_batch_operations()
    await test_concurrent_safety()
    await test_stats()
    await test_invalidate()
    await test_clear()
    print("\nAll P0 cache tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
