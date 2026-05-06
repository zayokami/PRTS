"""Smoke test for enhanced Tool Hooks + Permission Engine.

验证:
1. Pre-hook 权限拒绝
2. Pre-hook 参数修改
3. Post-hook 结果修改
4. Post-hook 审计日志 + 统计
5. Failure hook 降级处理
6. 工具调用超时
7. 并发限制
8. 配置文件加载
9. 异常隔离 (单个 hook 失败不影响其他 hooks)
10. ToolPermissionDenied / ToolTimeoutError
"""

import asyncio
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "apps" / "agent" / "src"))

from prts_agent.tools.hooks import (
    HookStats,
    HookedToolRegistry,
    HookResult,
    ToolHooks,
    ToolPermissionDenied,
    ToolTimeoutError,
)
from prts_agent.tools.permissions import ToolPermissionEngine, build_default_permission_engine
from prts_agent.tools.registry import ToolDefinition, ToolRegistry


async def _echo(arguments):
    await asyncio.sleep(0.001)  # 模拟微小延迟
    return f"echo: {arguments.get('msg', '')}"


async def _slow(arguments):
    delay = arguments.get("delay", 0)
    await asyncio.sleep(delay)
    return f"slept {delay}s"


async def _dangerous(arguments):
    return f"deleted: {arguments.get('path', '')}"


async def _failing(arguments):
    raise RuntimeError("intentional failure")


def make_registry():
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="echo",
            description="echo",
            input_schema={"type": "object", "properties": {"msg": {"type": "string"}}},
            invoker=_echo,
            source="builtin",
        )
    )
    reg.register(
        ToolDefinition(
            name="slow",
            description="slow",
            input_schema={"type": "object", "properties": {"delay": {"type": "number"}}},
            invoker=_slow,
            source="builtin",
        )
    )
    reg.register(
        ToolDefinition(
            name="bash__exec",
            description="bash",
            input_schema={"type": "object", "properties": {"cmd": {"type": "string"}}},
            invoker=_dangerous,
            source="skill",
        )
    )
    reg.register(
        ToolDefinition(
            name="fail_me",
            description="fail",
            input_schema={"type": "object", "properties": {}}
,
            invoker=_failing,
            source="builtin",
        )
    )
    return reg


async def test_permission_deny():
    """1. 默认策略应拒绝 bash__exec。"""
    reg = make_registry()
    hooks = ToolHooks()
    perm = build_default_permission_engine()
    hooks.register_pre(perm.check)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    try:
        await hooked.invoke("bash__exec", {"cmd": "rm -rf /"})
        assert False, "should have raised ToolPermissionDenied"
    except ToolPermissionDenied as exc:
        assert "bash__exec" in str(exc)
        print(f"  OK deny: {exc}")


async def test_permission_allow():
    """2. 普通工具应允许通过。"""
    reg = make_registry()
    hooks = ToolHooks()
    perm = build_default_permission_engine()
    hooks.register_pre(perm.check)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("echo", {"msg": "hello"})
    assert result == "echo: hello"
    print("  OK allow: echo passed")


async def test_pre_hook_modify():
    """3. Pre-hook 可修改参数。"""
    reg = make_registry()
    hooks = ToolHooks()

    async def modify_args(invocation):
        if invocation.name == "echo":
            new_args = {**invocation.arguments, "msg": "modified"}
            return HookResult(decision="modify", modified_arguments=new_args)
        return HookResult(decision="allow")

    hooks.register_pre(modify_args)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("echo", {"msg": "original"})
    assert result == "echo: modified", f"got: {result}"
    print("  OK modify: argument rewritten")


async def test_post_hook_modify_result():
    """4. Post-hook 可修改返回结果。"""
    reg = make_registry()
    hooks = ToolHooks()

    async def add_suffix(invocation, result, duration_ms):
        if invocation.name == "echo":
            return HookResult(
                decision="modify",
                modified_result=result + " [processed]"
            )
        return HookResult(decision="allow")

    hooks.register_post(add_suffix)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("echo", {"msg": "test"})
    assert result == "echo: test [processed]", f"got: {result}"
    print("  OK post modify: result rewritten")


async def test_audit_and_stats():
    """5. 审计日志 + 统计信息。"""
    reg = make_registry()
    hooks = ToolHooks()

    audit_log = []

    async def audit(invocation, result, duration_ms):
        audit_log.append({
            "name": invocation.name,
            "duration_ms": duration_ms,
        })
        return HookResult(decision="allow")

    hooks.register_post(audit)
    perm = build_default_permission_engine()
    hooks.register_pre(perm.check)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    await hooked.invoke("echo", {"msg": "audit"})

    stats = hooked.stats
    assert len(stats) >= 2  # pre + post
    pre_stats = [s for s in stats if s.phase == "pre"]
    post_stats = [s for s in stats if s.phase == "post"]
    assert len(pre_stats) >= 1
    assert len(post_stats) >= 1
    assert all(s.duration_ms >= 0 for s in stats)
    print(f"  OK stats: {len(stats)} hooks logged, pre={len(pre_stats)}, post={len(post_stats)}")


async def test_failure_hook_fallback():
    """6. Failure hook 提供降级结果。"""
    reg = make_registry()
    hooks = ToolHooks()

    async def fallback(invocation, exc, duration_ms):
        return HookResult(
            decision="modify",
            modified_result="fallback result"
        )

    hooks.register_failure(fallback)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("fail_me", {})
    assert result == "fallback result", f"got: {result}"
    print("  OK fallback: failure hook provided alternative result")


async def test_timeout():
    """7. 工具调用超时。"""
    reg = make_registry()
    hooks = ToolHooks()

    hooked = HookedToolRegistry(
        reg, hooks, session_id="s1", channel="test", timeout_seconds=0.1
    )

    try:
        await hooked.invoke("slow", {"delay": 1.0})
        assert False, "should have raised ToolTimeoutError"
    except ToolTimeoutError as exc:
        assert "timed out" in str(exc)
        print(f"  OK timeout: {exc}")


async def test_concurrent_limit():
    """8. 并发限制 (Semaphore)。"""
    reg = make_registry()
    hooks = ToolHooks()

    # 注册一个慢工具
    reg.register(
        ToolDefinition(
            name="concurrent_slow",
            description="slow",
            input_schema={"type": "object", "properties": {"delay": {"type": "number"}}},
            invoker=_slow,
            source="builtin",
        )
    )

    hooked = HookedToolRegistry(
        reg, hooks, session_id="s1", channel="test",
        timeout_seconds=5.0,
        max_concurrent=2,  # 最多 2 个并发
    )

    start = asyncio.get_event_loop().time()
    # 启动 4 个并发调用，每个 0.1s，限制为 2 个并发，总时间应约 0.2s
    tasks = [
        asyncio.create_task(hooked.invoke("concurrent_slow", {"delay": 0.1}))
        for _ in range(4)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start

    assert all(r == "slept 0.1s" for r in results)
    assert elapsed >= 0.18, f"concurrency not limited: {elapsed:.3f}s"
    print(f"  OK concurrent: 4 calls with max=2 took {elapsed:.3f}s")


async def test_exception_isolation():
    """9. 单个 hook 失败不影响其他 hooks 和工具调用。"""
    reg = make_registry()
    hooks = ToolHooks()

    async def broken_pre(invocation):
        raise RuntimeError("broken pre-hook")

    async def working_pre(invocation):
        return HookResult(decision="allow")

    async def broken_post(invocation, result, duration_ms):
        raise RuntimeError("broken post-hook")

    async def working_post(invocation, result, duration_ms):
        return HookResult(decision="allow")

    hooks.register_pre(broken_pre)
    hooks.register_pre(working_pre)
    hooks.register_post(broken_post)
    hooks.register_post(working_post)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("echo", {"msg": "isolation"})
    assert result == "echo: isolation"

    stats = hooked.stats
    assert len(stats) == 4
    errors = [s for s in stats if s.error]
    assert len(errors) == 2  # broken_pre + broken_post
    print(f"  OK isolation: tool succeeded despite 2 broken hooks")


async def test_config_load_save():
    """10. 权限配置文件的加载和保存。"""
    engine = build_default_permission_engine()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "permissions.json"
        engine.save(path)

        loaded = ToolPermissionEngine.load(path)

        # 验证加载后的规则行为一致
        class FakeInv:
            name = "bash__exec"
            arguments = {}
            source = "skill"
            session_id = "s1"
            channel = "test"

        result = await loaded.check(FakeInv())
        assert result.decision == "deny"
        print("  OK config: load/save roundtrip preserved rules")


async def main():
    print("Enhanced Tool Hooks smoke test")
    print()
    print("1. Permission deny")
    await test_permission_deny()
    print("2. Permission allow")
    await test_permission_allow()
    print("3. Pre-hook modify args")
    await test_pre_hook_modify()
    print("4. Post-hook modify result")
    await test_post_hook_modify_result()
    print("5. Audit + stats")
    await test_audit_and_stats()
    print("6. Failure hook fallback")
    await test_failure_hook_fallback()
    print("7. Timeout")
    await test_timeout()
    print("8. Concurrent limit")
    await test_concurrent_limit()
    print("9. Exception isolation")
    await test_exception_isolation()
    print("10. Config load/save")
    await test_config_load_save()
    print()
    print("All enhanced hook tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
