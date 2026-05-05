"""P4+ MCP health check smoke test —— 验证 MCPManager.health_check 能检测到
server 死亡并自动重启。

跑法(项目根)::<

    .venv/Scripts/python.exe scripts/smoke_p4_health.py
"""

from __future__ import annotations

import asyncio
import sys
from contextlib import AsyncExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "agent" / "src"))
sys.path.insert(0, str(REPO / "packages" / "prts-sdk" / "src"))
sys.dont_write_bytecode = True

from prts_agent.mcp import MCPConfig, MCPManager  # noqa: E402
from prts_agent.tools import ToolRegistry  # noqa: E402

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

FIXTURE = REPO / "scripts" / "_fixtures" / "echo_mcp_server.py"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


def make_echo_config() -> MCPConfig:
    return MCPConfig(
        mcpServers={
            "echo": {
                "command": sys.executable,
                "args": [str(FIXTURE)],
                "disabled": False,
                "timeout_seconds": 10.0,
            }
        }
    )


async def test_health_check() -> None:
    if not FIXTURE.is_file():
        fail(f"echo fixture 缺失: {FIXTURE}")

    registry = ToolRegistry()
    stack = AsyncExitStack()
    await stack.__aenter__()

    try:
        manager = MCPManager(REPO, registry, stack)
        cfg = make_echo_config()
        await manager.start_all(cfg)

        states = manager.states()
        assert_eq(len(states), 1, "启动后应有 1 个 server")
        if states[0].status != "ready":
            fail(f"启动未 ready: {states[0].status}")
        ok("server 启动 ready")

        # 健康检查:应该返回 healthy
        results = await manager.health_check(auto_restart=False)
        assert_eq(results.get("echo"), "healthy", "首次 health check 应为 healthy")
        ok("health check 返回 healthy")

        # 模拟 server 死亡:杀掉子进程
        # 获取 session 引用并破坏它(通过关闭底层 stack)
        # 更简单的方式:直接杀掉子进程——但我们不知道 PID
        # 替代方案:把 session 从 _sessions 中移除,模拟 session 丢失
        manager._sessions.pop("echo", None)

        # 再次 health check(不自动重启):应该检测到 unhealthy
        results = await manager.health_check(auto_restart=False)
        assert_eq(results.get("echo"), "unhealthy", "session 丢失后应为 unhealthy")
        ok("health check 检测到 unhealthy")

        # health check(自动重启):应该重启成功
        results = await manager.health_check(auto_restart=True)
        assert_eq(results.get("echo"), "restarting", "auto_restart 后应为 restarting")
        ok("auto_restart 触发 restarting")

        # 给一点时间让重启完成
        await asyncio.sleep(0.5)

        # 再次检查:应该恢复 healthy
        results = await manager.health_check(auto_restart=False)
        assert_eq(results.get("echo"), "healthy", "重启后应恢复 healthy")
        ok("server 重启后恢复 healthy")

        # 验证 registry 中的工具也恢复了
        assert "echo__echo" in registry.names(), "重启后 registry 应有 echo__echo"
        ok("重启后 registry 工具恢复")

    finally:
        await stack.aclose()


async def main() -> None:
    await test_health_check()
    print(f"\n{GREEN}P4 health check smoke passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
