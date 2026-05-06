"""Smoke test for enhanced Tool Hooks + Permission Engine + ToolResult.

验证:
1. Pre-hook 权限拒绝 → ToolResult(is_error=True)
2. Pre-hook 参数修改
3. Post-hook 结果修改
4. Post-hook 审计日志 + 统计
5. Failure hook 降级处理
6. 工具调用超时 → ToolResult(is_error=True)
7. 并发限制
8. 配置文件加载
9. 异常隔离 (单个 hook 失败不影响其他 hooks)
10. ToolResult 结构化格式
11. Schema 严格验证 (strict=True)
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
from prts_agent.tools.tool_result import ToolResult
from prts_agent.tools.validator import ToolInputValidationError, validate_tool_input


async def _echo(arguments):
    await asyncio.sleep(0.001)
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
            input_schema={"type": "object", "properties": {}},
            invoker=_failing,
            source="builtin",
        )
    )
    return reg


async def test_permission_deny():
    """1. 默认策略应拒绝 bash__exec,返回 ToolResult(is_error=True)。"""
    reg = make_registry()
    hooks = ToolHooks()
    perm = build_default_permission_engine()
    hooks.register_pre(perm.check)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("bash__exec", {"cmd": "rm -rf /"})
    assert isinstance(result, ToolResult)
    assert result.is_error is True
    assert result.error_type == "ToolPermissionDenied"
    assert "bash__exec" in result.error_message
    print(f"  OK deny: {result.error_type} — {result.error_message}")


async def test_permission_allow():
    """2. 普通工具应允许通过,返回 ToolResult(is_error=False)。"""
    reg = make_registry()
    hooks = ToolHooks()
    perm = build_default_permission_engine()
    hooks.register_pre(perm.check)

    hooked = HookedToolRegistry(reg, hooks, session_id="s1", channel="test")

    result = await hooked.invoke("echo", {"msg": "hello"})
    assert isinstance(result, ToolResult)
    assert result.is_error is False
    assert result.content == "echo: hello"
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
    assert result.content == "echo: modified", f"got: {result.content}"
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
    assert result.content == "echo: test [processed]", f"got: {result.content}"
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

    result = await hooked.invoke("echo", {"msg": "audit"})
    assert result.is_error is False

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
    assert result.is_error is False
    assert result.content == "fallback result", f"got: {result.content}"
    print("  OK fallback: failure hook provided alternative result")


async def test_timeout():
    """7. 工具调用超时 → ToolResult(is_error=True)。"""
    reg = make_registry()
    hooks = ToolHooks()

    hooked = HookedToolRegistry(
        reg, hooks, session_id="s1", channel="test", timeout_seconds=0.1
    )

    result = await hooked.invoke("slow", {"delay": 1.0})
    assert result.is_error is True
    assert result.error_type == "ToolTimeoutError"
    assert "timed out" in result.error_message
    print(f"  OK timeout: {result.error_type}")


async def test_concurrent_limit():
    """8. 并发限制 (Semaphore)。"""
    reg = make_registry()
    hooks = ToolHooks()

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
        max_concurrent=2,
    )

    start = asyncio.get_event_loop().time()
    tasks = [
        asyncio.create_task(hooked.invoke("concurrent_slow", {"delay": 0.1}))
        for _ in range(4)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = asyncio.get_event_loop().time() - start

    assert all(r.content == "slept 0.1s" for r in results)
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
    assert result.content == "echo: isolation"

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

        class FakeInv:
            name = "bash__exec"
            arguments = {}
            source = "skill"
            session_id = "s1"
            channel = "test"

        result = await loaded.check(FakeInv())
        assert result.decision == "deny"
        print("  OK config: load/save roundtrip preserved rules")


async def test_tool_result_format():
    """11. ToolResult 结构化格式。"""
    # 正常结果
    r1 = ToolResult.success({"data": [1, 2, 3]})
    assert r1.is_error is False
    assert r1.to_llm_text() == '{"data": [1, 2, 3]}'
    sse = r1.to_sse_dict()
    assert sse["is_error"] is False
    assert sse["content"] == {"data": [1, 2, 3]}

    # 错误结果
    r2 = ToolResult(
        is_error=True,
        error_type="ToolPermissionDenied",
        error_message="denied",
    )
    assert r2.to_llm_text() == "[ERROR: ToolPermissionDenied]\ndenied"
    sse2 = r2.to_sse_dict()
    assert sse2["is_error"] is True
    assert sse2["error_type"] == "ToolPermissionDenied"

    # 从异常构造
    r3 = ToolResult.from_exception(ValueError("bad arg"))
    assert r3.is_error is True
    assert r3.error_type == "ValueError"
    assert r3.error_message == "bad arg"

    print("  OK ToolResult: success, error, from_exception")


async def test_schema_validation():
    """12. JSON Schema 严格验证。"""
    schema = {
        "type": "object",
        "required": ["name"],
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }

    # 有效参数
    validate_tool_input("test", schema, {"name": "Alice", "age": 30}, strict=True)

    # 缺少 required
    try:
        validate_tool_input("test", schema, {"age": 30}, strict=True)
        assert False, "should have raised"
    except ToolInputValidationError as exc:
        assert "name" in str(exc) and "required" in str(exc).lower()

    # 类型错误
    try:
        validate_tool_input("test", schema, {"name": "Alice", "age": "thirty"}, strict=True)
        assert False, "should have raised"
    except ToolInputValidationError as exc:
        assert "age" in str(exc) and "integer" in str(exc).lower()

    # strict=False 时跳过
    validate_tool_input("test", schema, {"bad": "data"}, strict=False)

    print("  OK schema validation: required, type, strict toggle")


async def _greet_async(arguments):
    return f"Hello {arguments['name']}"


async def test_strict_tool_invoke():
    """13. strict=True 的工具在 invoke 时自动验证 schema。"""
    reg = ToolRegistry()
    reg.register(
        ToolDefinition(
            name="greet",
            description="greet",
            input_schema={
                "type": "object",
                "required": ["name"],
                "properties": {"name": {"type": "string"}},
            },
            invoker=_greet_async,
            source="builtin",
            strict=True,  # 启用严格验证
        )
    )

    hooks = ToolHooks()
    hooked = HookedToolRegistry(reg, hooks)

    # 有效调用
    result = await hooked.invoke("greet", {"name": "World"})
    print(f"DEBUG greet result: {result}")
    assert result.is_error is False, f"expected success, got: {result}"
    assert result.content == "Hello World"

    # 无效调用:缺少 required
    result = await hooked.invoke("greet", {})
    assert result.is_error is True
    assert result.error_type == "ToolInputValidationError"
    assert "name" in result.error_message and "required" in result.error_message.lower()

    # 无效调用:类型错误
    result = await hooked.invoke("greet", {"name": 123})
    assert result.is_error is True
    assert result.error_type == "ToolInputValidationError"

    print("  OK strict tool: auto-validation on invoke")


async def main():
    print("Enhanced Tool Hooks + ToolResult smoke test")
    print()
    await test_permission_deny()
    await test_permission_allow()
    await test_pre_hook_modify()
    await test_post_hook_modify_result()
    await test_audit_and_stats()
    await test_failure_hook_fallback()
    await test_timeout()
    await test_concurrent_limit()
    await test_exception_isolation()
    await test_config_load_save()
    await test_tool_result_format()
    await test_schema_validation()
    await test_strict_tool_invoke()
    print()
    print("All tests passed!")


if __name__ == "__main__":
    asyncio.run(main())
