"""P8 smoke test —— 验证长上下文管理三层记忆架构。

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p8_context.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "agent" / "src"))
sys.path.insert(0, str(REPO / "packages" / "prts-sdk" / "src"))
sys.dont_write_bytecode = True

from prts_agent.llm.base import TokenUsage  # noqa: E402
from prts_agent.llm.tokenizer import (  # noqa: E402
    estimate_prompt_tokens,
    get_calibration_ratio,
    get_suggested_headroom,
    record_usage_discrepancy,
)
from prts_agent.memory.importance import ImportanceScorer  # noqa: E402
from prts_agent.memory.summarizer import MemoryCard  # noqa: E402

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


def test_token_usage() -> None:
    """TokenUsage 数据类。"""
    u = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150, model="gpt-4o")
    assert_eq(u.prompt_tokens, 100, "prompt_tokens")
    assert_eq(u.completion_tokens, 50, "completion_tokens")
    assert_eq(u.total_tokens, 150, "total_tokens")
    ok("TokenUsage dataclass")


def test_usage_calibration() -> None:
    """usage 校准记录。"""
    record_usage_discrepancy("gpt-4o", heuristic_tokens=120, actual_tokens=100)
    record_usage_discrepancy("gpt-4o", heuristic_tokens=110, actual_tokens=105)
    record_usage_discrepancy("gpt-4o", heuristic_tokens=130, actual_tokens=110)

    ratio = get_calibration_ratio("gpt-4o")
    if ratio is None:
        fail("calibration ratio should not be None")
    # 平均 actual/heuristic ≈ (100+105+110)/(120+110+130) ≈ 0.87
    assert 0.8 <= ratio <= 0.95, f"unexpected calibration ratio: {ratio}"
    ok(f"usage calibration ratio={ratio:.2f}")

    # 无记录模型返回 None
    assert_eq(get_calibration_ratio("unknown-model"), None, "no calibration for unknown model")
    ok("no calibration for unknown model")

    # 极端值应被过滤
    record_usage_discrepancy("test", 100, 500)  # ratio=5.0, 应被过滤
    assert_eq(get_calibration_ratio("test"), None, "extreme ratio filtered")
    ok("extreme ratio filtered")


def test_estimate_prompt_tokens() -> None:
    """估算 prompt token。"""
    messages = [{"role": "user", "content": "hello world"}]

    # 无 actual_usage: 用 heuristic
    est1 = estimate_prompt_tokens(messages)
    assert est1 > 0, "heuristic should return positive"
    ok(f"heuristic estimate={est1}")

    # 有 actual_usage: 用实际值
    usage = TokenUsage(prompt_tokens=50, completion_tokens=10, total_tokens=60, model="gpt-4o")
    est2 = estimate_prompt_tokens(messages, actual_usage=usage)
    assert_eq(est2, 50, "uses actual prompt_tokens")
    ok("uses actual prompt_tokens when available")


def test_dynamic_headroom() -> None:
    """动态 headroom 建议。"""
    usages = [
        TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, model="gpt-4o"),
        TokenUsage(prompt_tokens=2000, completion_tokens=100, total_tokens=2100, model="gpt-4o"),
        TokenUsage(prompt_tokens=4000, completion_tokens=100, total_tokens=4100, model="gpt-4o"),
    ]
    headroom = get_suggested_headroom("gpt-4o", usages)
    assert headroom == 0.65, f"fast growth should give 0.65, got {headroom}"
    ok(f"fast growth headroom={headroom}")

    # 平稳增长
    stable = [
        TokenUsage(prompt_tokens=1000, completion_tokens=100, total_tokens=1100, model="gpt-4o"),
        TokenUsage(prompt_tokens=1050, completion_tokens=100, total_tokens=1150, model="gpt-4o"),
        TokenUsage(prompt_tokens=1100, completion_tokens=100, total_tokens=1200, model="gpt-4o"),
    ]
    headroom2 = get_suggested_headroom("gpt-4o", stable)
    assert headroom2 == 0.85, f"stable should give 0.85, got {headroom2}"
    ok(f"stable headroom={headroom2}")

    # 无历史
    headroom3 = get_suggested_headroom("gpt-4o", [])
    assert headroom3 == 0.80, f"default should be 0.80, got {headroom3}"
    ok(f"default headroom={headroom3}")


def test_importance_scorer() -> None:
    """重要性评分。"""
    scorer = ImportanceScorer()

    # 用户决策消息应高分
    class FakeMsg:
        def __init__(self, role, content, meta=None):
            self.role = role
            self.content = content
            self.meta = meta or {}

    decision = FakeMsg("user", "我决定使用 React 而不是 Vue")
    score1 = scorer.score(decision)
    assert score1 >= 0.5, f"decision should be high importance, got {score1}"
    ok(f"decision importance={score1:.2f}")

    # 普通闲聊应低分
    chat = FakeMsg("user", "好的,谢谢")
    score2 = scorer.score(chat)
    assert score2 <= 0.3, f"casual chat should be low importance, got {score2}"
    ok(f"casual importance={score2:.2f}")

    # 工具错误应加分
    error = FakeMsg("tool", "FileNotFoundError: no such file", {"is_error": True})
    score3 = scorer.score(error)
    assert score3 >= 0.3, f"tool error should have bonus, got {score3}"
    ok(f"tool error importance={score3:.2f}")

    # 批量评分
    batch = [decision, chat, error]
    filtered = scorer.filter_important(batch, threshold=0.3)
    assert len(filtered) >= 2, f"should keep at least 2 important messages, got {len(filtered)}"
    ok(f"filtered {len(batch)} -> {len(filtered)} important")


def test_memory_card() -> None:
    """MemoryCard 数据结构。"""
    card = MemoryCard(
        id="sum-test-123",
        session_id="sess-456",
        summary="用户决定使用 React",
        key_facts=["选择 React", "放弃 Vue"],
        decisions=["使用 React"],
        todos=["搭建项目"],
        message_range=(1, 10),
        importance=0.85,
    )
    assert_eq(card.id, "sum-test-123", "card.id")
    assert_eq(card.importance, 0.85, "card.importance")
    assert_eq(len(card.key_facts), 2, "card.key_facts")
    ok("MemoryCard structure")


async def main() -> None:
    test_token_usage()
    test_usage_calibration()
    test_estimate_prompt_tokens()
    test_dynamic_headroom()
    test_importance_scorer()
    test_memory_card()

    print(f"\n{GREEN}P8 context management smoke all passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
