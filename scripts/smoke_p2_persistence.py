"""Smoke test for P2 cache persistence (calibration + budget history).

验证:
1. Token 校准缓存保存/加载
2. 预算历史保存/加载
3. 数据在 SQLite 中持久化
"""

import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "agent" / "src"))

from prts_agent.memory.sqlite import SqliteStore
from prts_agent.llm.tokenizer import (
    _calibration_history,
    get_calibration_ratio,
    record_usage_discrepancy,
    set_calibration_store,
)


async def test_calibration_persistence():
    """1. 校准缓存持久化。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / "test.db"
        store = SqliteStore(db)
        await store.ensure_schema()

        # 注入 store
        set_calibration_store(store)
        # 清空全局状态,避免之前测试的残留
        _calibration_history.clear()

        # 记录几条校准数据
        for i in range(5):
            record_usage_discrepancy("gpt-4", 100 + i * 10, 90 + i * 9)

        # 直接同步保存到 SQLite(不依赖异步后台任务,避免测试竞争)
        history = list(_calibration_history.get("gpt-4", []))
        await store.save_calibration("gpt-4", history)

        # 重新加载
        loaded = await store.load_calibration("gpt-4")
        assert len(loaded) == 5, f"expected 5, got {len(loaded)}"

        # 验证 ratio 计算正确
        ratio = get_calibration_ratio("gpt-4")
        # (90+99+108+117+126) / (100+110+120+130+140) = 540 / 600 = 0.9
        assert ratio is not None
        assert abs(ratio - 0.9) < 0.01, f"ratio={ratio}"

        print(f"  OK calibration persisted: {len(loaded)} entries, ratio={ratio:.2f}")


async def test_budget_history_persistence():
    """2. 预算历史持久化。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / "test.db"
        store = SqliteStore(db)
        await store.ensure_schema()

        # 保存几条历史
        for i in range(5):
            await store.save_budget_usage(
                model="gpt-4",
                prompt_tokens=100 + i * 20,
                completion_tokens=50 + i * 10,
                total_tokens=150 + i * 30,
            )

        # 加载
        history = await store.load_budget_history("gpt-4", limit=10)
        assert len(history) == 5, f"expected 5, got {len(history)}"

        # 验证数据正确
        first = history[0]
        assert first[0] == 100  # prompt_tokens
        assert first[1] == 50   # completion_tokens
        assert first[2] == 150  # total_tokens

        # 验证限制
        limited = await store.load_budget_history("gpt-4", limit=3)
        assert len(limited) == 3

        # 验证清理
        deleted = await store.delete_old_budget_history("gpt-4", keep=2)
        assert deleted == 3  # 5 - 2 = 3

        remaining = await store.load_budget_history("gpt-4", limit=10)
        assert len(remaining) == 2

        print(f"  OK budget history persisted: saved=5, loaded={len(history)}, after_cleanup={len(remaining)}")


async def test_calibration_multiple_models():
    """3. 多模型校准独立存储。"""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = Path(tmp) / "test.db"
        store = SqliteStore(db)
        await store.ensure_schema()

        set_calibration_store(store)
        _calibration_history.clear()

        # 记录不同模型的数据
        record_usage_discrepancy("gpt-4", 100, 90)
        record_usage_discrepancy("claude-3", 100, 110)

        # 直接同步保存
        for model in ["gpt-4", "claude-3"]:
            history = list(_calibration_history.get(model, []))
            await store.save_calibration(model, history)

        all_data = await store.load_all_calibrations()
        assert "gpt-4" in all_data
        assert "claude-3" in all_data
        assert len(all_data["gpt-4"]) == 1
        assert len(all_data["claude-3"]) == 1

        print(f"  OK multiple models: {list(all_data.keys())}")


async def main():
    print("P2 Cache Persistence smoke test")
    await test_calibration_persistence()
    await test_budget_history_persistence()
    await test_calibration_multiple_models()
    print("\nAll P2 persistence tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
