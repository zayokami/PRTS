"""Eval Framework —— 系统化 Agent 评估。

OpenAI Evals 启发的轻量评估框架:
- EvalCase: 单个测试用例(输入、期望输出、检查器)
- EvalSuite: 测试套件(一组相关用例)
- EvalRunner: 执行器(并行运行、收集结果)
- Metrics: 指标计算(成功率、延迟、token 使用)

使用方式:
    runner = EvalRunner()
    suite = build_smoke_suite()
    result = await runner.run(suite)
    print(result.summary())
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---- 数据模型 ----

@dataclass
class EvalCase:
    """单个评估用例。"""

    name: str
    description: str
    # 测试函数:返回 (passed, details_dict)
    func: Callable[[], Awaitable[tuple[bool, dict[str, Any]]]]
    timeout_seconds: float = 30.0
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalCaseResult:
    """单个用例的执行结果。"""

    name: str
    passed: bool
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class EvalSuite:
    """测试套件。"""

    name: str
    description: str
    cases: list[EvalCase]


@dataclass
class EvalReport:
    """评估报告。"""

    suite_name: str
    total: int
    passed: int
    failed: int
    skipped: int = 0
    duration_ms: float = 0.0
    case_results: list[EvalCaseResult] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return self.passed / self.total

    def summary(self) -> str:
        lines = [
            f"Eval Report: {self.suite_name}",
            f"  Total:   {self.total}",
            f"  Passed:  {self.passed} ({self.success_rate:.1%})",
            f"  Failed:  {self.failed}",
            f"  Skipped: {self.skipped}",
            f"  Duration: {self.duration_ms:.1f}ms",
        ]
        if self.metrics:
            lines.append("  Metrics:")
            for k, v in self.metrics.items():
                lines.append(f"    {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "success_rate": round(self.success_rate, 3),
            "duration_ms": round(self.duration_ms, 1),
            "case_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "duration_ms": round(r.duration_ms, 1),
                    "error": r.error,
                }
                for r in self.case_results
            ],
            "metrics": self.metrics,
        }


# ---- 执行器 ----

class EvalRunner:
    """评估执行器。"""

    def __init__(self, max_concurrent: int = 4) -> None:
        self._max_concurrent = max_concurrent

    async def run(
        self,
        suite: EvalSuite,
        *,
        fail_fast: bool = False,
    ) -> EvalReport:
        """运行测试套件。

        参数:
            suite: 测试套件
            fail_fast: 第一个失败时停止

        返回:
            EvalReport
        """
        logger.info("running eval suite: %s (%d cases)", suite.name, len(suite.cases))
        start = time.perf_counter()

        semaphore = asyncio.Semaphore(self._max_concurrent)
        results: list[EvalCaseResult] = []

        async def run_case(case: EvalCase) -> EvalCaseResult:
            async with semaphore:
                case_start = time.perf_counter()
                try:
                    passed, details = await asyncio.wait_for(
                        case.func(),
                        timeout=case.timeout_seconds,
                    )
                    duration_ms = (time.perf_counter() - case_start) * 1000
                    return EvalCaseResult(
                        name=case.name,
                        passed=passed,
                        duration_ms=duration_ms,
                        details=details,
                    )
                except asyncio.TimeoutError:
                    duration_ms = (time.perf_counter() - case_start) * 1000
                    return EvalCaseResult(
                        name=case.name,
                        passed=False,
                        duration_ms=duration_ms,
                        error=f"timeout after {case.timeout_seconds}s",
                    )
                except Exception as exc:
                    duration_ms = (time.perf_counter() - case_start) * 1000
                    logger.exception("eval case %s failed", case.name)
                    return EvalCaseResult(
                        name=case.name,
                        passed=False,
                        duration_ms=duration_ms,
                        error=f"{type(exc).__name__}: {exc}",
                    )

        # 顺序执行(避免并行时日志混乱),但用 semaphore 限制并发
        for case in suite.cases:
            if fail_fast and any(not r.passed for r in results):
                results.append(
                    EvalCaseResult(
                        name=case.name,
                        passed=False,
                        duration_ms=0.0,
                        error="skipped (fail_fast)",
                    )
                )
                continue

            result = await run_case(case)
            results.append(result)

            status = "PASS" if result.passed else "FAIL"
            logger.info(
                "  [%s] %s (%.1fms)%s",
                status,
                result.name,
                result.duration_ms,
                f" — {result.error}" if result.error else "",
            )

        total_duration_ms = (time.perf_counter() - start) * 1000
        passed = sum(1 for r in results if r.passed)
        failed = sum(1 for r in results if not r.passed and not r.error == "skipped (fail_fast)")
        skipped = sum(1 for r in results if r.error == "skipped (fail_fast)")

        # 计算指标
        durations = [r.duration_ms for r in results]
        metrics: dict[str, Any] = {
            "avg_duration_ms": round(sum(durations) / len(durations), 1) if durations else 0,
            "max_duration_ms": round(max(durations), 1) if durations else 0,
            "min_duration_ms": round(min(durations), 1) if durations else 0,
        }

        # 按 tag 分组统计
        tag_stats: dict[str, dict[str, int]] = {}
        for case, result in zip(suite.cases, results):
            for tag in case.tags:
                if tag not in tag_stats:
                    tag_stats[tag] = {"total": 0, "passed": 0}
                tag_stats[tag]["total"] += 1
                if result.passed:
                    tag_stats[tag]["passed"] += 1

        if tag_stats:
            metrics["by_tag"] = {
                tag: {
                    "total": stats["total"],
                    "passed": stats["passed"],
                    "rate": round(stats["passed"] / stats["total"], 3),
                }
                for tag, stats in tag_stats.items()
            }

        report = EvalReport(
            suite_name=suite.name,
            total=len(suite.cases),
            passed=passed,
            failed=failed,
            skipped=skipped,
            duration_ms=total_duration_ms,
            case_results=results,
            metrics=metrics,
        )

        logger.info("eval suite completed: %s", report.summary().replace("\n", " | "))
        return report


# ---- 便捷断言工具 ----

def assert_eq(actual: Any, expected: Any, msg: str | None = None) -> None:
    """断言相等。"""
    if actual != expected:
        raise AssertionError(
            msg or f"expected {expected!r}, got {actual!r}"
        )


def assert_true(value: bool, msg: str | None = None) -> None:
    """断言为真。"""
    if not value:
        raise AssertionError(msg or f"expected True, got {value!r}")


def assert_in(member: Any, container: Any, msg: str | None = None) -> None:
    """断言包含。"""
    if member not in container:
        raise AssertionError(
            msg or f"expected {member!r} in {container!r}"
        )


def assert_raises(
    exc_type: type[Exception],
    func: Callable[[], Any],
    msg: str | None = None,
) -> Exception:
    """断言抛出异常。"""
    try:
        func()
    except exc_type as exc:
        return exc
    raise AssertionError(msg or f"expected {exc_type.__name__}, no exception raised")
