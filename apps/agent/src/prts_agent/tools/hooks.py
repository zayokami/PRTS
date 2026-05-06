"""Tool Hooks —— 工具调用前后拦截与扩展。

支持权限检查、审计日志、参数重写、结果后处理、失败恢复、超时控制、并发限制。
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class ToolInvocation(Protocol):
    """工具调用上下文，供 hook 读取和修改。"""

    name: str
    arguments: dict[str, Any]
    source: str  # "skill" | "mcp" | "builtin"
    session_id: str | None
    channel: str


@dataclass
class HookResult:
    """Hook 执行结果。"""

    decision: str = "allow"  # "allow" | "deny" | "modify"
    reason: str | None = None
    modified_arguments: dict[str, Any] | None = None
    modified_result: Any = None  # post hook 可重写返回结果
    side_effects: list[dict[str, Any]] = field(default_factory=list)


PreToolHook = Callable[[ToolInvocation], Awaitable[HookResult]]
PostToolHook = Callable[
    [ToolInvocation, Any, float],  # invocation, result, duration_ms
    Awaitable[HookResult],
]
PostFailureHook = Callable[
    [ToolInvocation, Exception, float],  # invocation, exception, duration_ms
    Awaitable[HookResult],
]


@dataclass
class ToolHooks:
    """一组 pre/post/failure 钩子。"""

    pre: list[PreToolHook] = field(default_factory=list)
    post: list[PostToolHook] = field(default_factory=list)
    failure: list[PostFailureHook] = field(default_factory=list)

    def register_pre(self, hook: PreToolHook) -> None:
        self.pre.append(hook)

    def register_post(self, hook: PostToolHook) -> None:
        self.post.append(hook)

    def register_failure(self, hook: PostFailureHook) -> None:
        self.failure.append(hook)


@dataclass
class HookStats:
    """单个 hook 的执行统计。"""

    hook_name: str
    phase: str  # "pre" | "post" | "failure"
    duration_ms: float
    decision: str | None = None
    error: str | None = None


class HookedToolRegistry:
    """包裹 ToolRegistry，在 invoke 前后执行 hooks。

    增强特性:
    - 超时控制 (timeout)
    - 并发限制 (semaphore)
    - pre hooks: 权限检查、参数修改
    - post hooks: 结果后处理、可修改返回结果
    - failure hooks: 错误恢复、降级处理
    - 异常隔离: 单个 hook 失败不影响其他 hooks
    """

    def __init__(
        self,
        inner: Any,
        hooks: ToolHooks,
        session_id: str | None = None,
        channel: str = "web",
        timeout_seconds: float = 60.0,
        max_concurrent: int = 4,
    ) -> None:
        self._inner = inner
        self._hooks = hooks
        self._session_id = session_id
        self._channel = channel
        self._timeout = timeout_seconds
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._stats: list[HookStats] = []

    @property
    def stats(self) -> list[HookStats]:
        """返回本次工具调用的 hook 执行统计。"""
        return list(self._stats)

    def clear_stats(self) -> None:
        self._stats.clear()

    async def invoke(self, name: str, arguments: dict[str, Any]) -> Any:
        tool = self._inner.get(name)
        if tool is None:
            raise KeyError(f"unknown tool: {name}")

        invocation = _SimpleInvocation(
            name=name,
            arguments=arguments,
            source=tool.source,
            session_id=self._session_id,
            channel=self._channel,
        )

        self._stats.clear()

        # ---- Pre hooks ----
        for hook in self._hooks.pre:
            start = time.perf_counter()
            try:
                result = await hook(invocation)
                duration_ms = (time.perf_counter() - start) * 1000
                self._stats.append(
                    HookStats(
                        hook_name=hook.__name__,
                        phase="pre",
                        duration_ms=duration_ms,
                        decision=result.decision,
                    )
                )
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                self._stats.append(
                    HookStats(
                        hook_name=hook.__name__,
                        phase="pre",
                        duration_ms=duration_ms,
                        error=str(exc),
                    )
                )
                logger.exception("pre-tool hook %s failed for %s", hook.__name__, name)
                # 异常隔离：继续执行下一个 hook
                continue

            if result.decision == "deny":
                logger.warning(
                    "tool %s denied by hook %s: %s",
                    name,
                    hook.__name__,
                    result.reason,
                )
                raise ToolPermissionDenied(
                    f"tool {name} denied by {hook.__name__}: {result.reason}"
                )
            if result.decision == "modify" and result.modified_arguments:
                invocation.arguments = result.modified_arguments
                arguments = result.modified_arguments

        # ---- 实际调用 (带并发限制 + 超时) ----
        start = time.perf_counter()
        async with self._semaphore:
            try:
                result = await asyncio.wait_for(
                    self._inner.invoke(name, arguments),
                    timeout=self._timeout,
                )
                duration_ms = (time.perf_counter() - start) * 1000
            except asyncio.TimeoutError:
                duration_ms = (time.perf_counter() - start) * 1000
                logger.error("tool %s timed out after %.1fs", name, self._timeout)
                raise ToolTimeoutError(
                    f"tool {name} timed out after {self._timeout}s"
                ) from None
            except Exception as exc:
                duration_ms = (time.perf_counter() - start) * 1000
                # ---- Failure hooks ----
                for hook in self._hooks.failure:
                    hook_start = time.perf_counter()
                    try:
                        fh_result = await hook(invocation, exc, duration_ms)
                        fh_duration = (time.perf_counter() - hook_start) * 1000
                        self._stats.append(
                            HookStats(
                                hook_name=hook.__name__,
                                phase="failure",
                                duration_ms=fh_duration,
                                decision=fh_result.decision,
                            )
                        )
                        if fh_result.decision == "modify" and fh_result.modified_result is not None:
                            # failure hook 提供了降级结果
                            logger.info(
                                "failure hook %s provided fallback for %s",
                                hook.__name__,
                                name,
                            )
                            return fh_result.modified_result
                    except Exception as fh_exc:
                        fh_duration = (time.perf_counter() - hook_start) * 1000
                        self._stats.append(
                            HookStats(
                                hook_name=hook.__name__,
                                phase="failure",
                                duration_ms=fh_duration,
                                error=str(fh_exc),
                            )
                        )
                        logger.exception(
                            "failure hook %s failed for %s", hook.__name__, name
                        )
                raise

        # ---- Post hooks (成功时) ----
        for hook in self._hooks.post:
            hook_start = time.perf_counter()
            try:
                post_result = await hook(invocation, result, duration_ms)
                hook_duration = (time.perf_counter() - hook_start) * 1000
                self._stats.append(
                    HookStats(
                        hook_name=hook.__name__,
                        phase="post",
                        duration_ms=hook_duration,
                        decision=post_result.decision,
                    )
                )
                if post_result.decision == "modify" and post_result.modified_result is not None:
                    result = post_result.modified_result
                    logger.debug(
                        "post hook %s modified result for %s", hook.__name__, name
                    )
            except Exception as post_exc:
                hook_duration = (time.perf_counter() - hook_start) * 1000
                self._stats.append(
                    HookStats(
                        hook_name=hook.__name__,
                        phase="post",
                        duration_ms=hook_duration,
                        error=str(post_exc),
                    )
                )
                logger.exception("post-tool hook %s failed for %s", hook.__name__, name)

        return result


class ToolPermissionDenied(Exception):
    """钩子拒绝工具调用时抛出。"""


class ToolTimeoutError(Exception):
    """工具调用超时时抛出。"""


@dataclass
class _SimpleInvocation:
    name: str
    arguments: dict[str, Any]
    source: str
    session_id: str | None
    channel: str
