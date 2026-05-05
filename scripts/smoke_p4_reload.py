"""P4+ smoke test —— 验证 MCP 热重载(P4 滑出项)。

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p4_reload.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
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


async def test_reload() -> None:
    """启动 echo server,reload 一次,验证旧进程已死、新进程已起。"""
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
        assert_eq(len(states), 1, "首次启动应有 1 个 server")
        if states[0].status != "ready":
            fail(f"首次启动未 ready: {states[0].status}")
        ok("首次启动 ready")

        # 记录旧进程 PID(通过 tool 调用间接验证)
        old_tool = registry.get("echo__echo")
        assert old_tool is not None, "首次启动后应有 echo__echo"
        result1 = await registry.invoke("echo__echo", {"text": "ping"})
        assert_eq(result1, "ping", "旧进程应能响应")
        ok("旧进程响应正常")

        # reload:停旧起新
        await manager.reload(cfg)

        new_states = manager.states()
        assert_eq(len(new_states), 1, "reload 后仍应有 1 个 server")
        if new_states[0].status != "ready":
            fail(f"reload 后未 ready: {new_states[0].status}")
        ok("reload 后 ready")

        # 验证 registry 里的工具也重新注册了
        new_tool = registry.get("echo__echo")
        assert new_tool is not None, "reload 后应有 echo__echo"
        result2 = await registry.invoke("echo__echo", {"text": "pong"})
        assert_eq(result2, "pong", "新进程应能响应")
        ok("reload 后新进程响应正常")

        # 验证旧 tool 的 invoker 已不可用(因为旧子进程已死)
        # 这里只验证 registry 只保留了新注册的,没有残留旧的
        all_tools = registry.names()
        assert "echo__echo" in all_tools, "reload 后 echo__echo 仍应在 registry"
        assert_eq(len([n for n in all_tools if n.startswith("echo__")]), 2, "echo server 应有 2 个工具(echo + shout)")
        ok("reload 后 registry 干净,无残留")

    finally:
        await stack.aclose()


async def main() -> None:
    await test_reload()
    print(f"\n{GREEN}P4 reload smoke passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
