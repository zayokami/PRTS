"""P4 workspace MCP server smoke test —— 验证 prts-workspace MCP server 链路。

需要:
- ``uv sync --all-packages`` 已安装 ``prts-workspace-mcp``

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p4_workspace.py
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

from prts_agent.mcp import MCPConfig, MCPManager, MCPServerConfig  # noqa: E402
from prts_agent.tools import ToolRegistry  # noqa: E402

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> str:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


async def test_workspace_mcp_server() -> None:
    """启动 prts-workspace,验证文档操作工具。"""
    registry = ToolRegistry()
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    td = tempfile.mkdtemp()

    try:
        manager = MCPManager(REPO, registry, parent_stack)
        cfg = MCPConfig(
            mcpServers={
                "prts-workspace": MCPServerConfig(
                    command=sys.executable,
                    args=[str(REPO / "mcp-servers" / "prts-workspace" / "src" / "prts_workspace_mcp" / "server.py")],
                    env={"PRTS_WORKSPACE_DIR": td},
                    timeout_seconds=10.0,
                )
            }
        )
        await manager.start_all(cfg)

        state = manager.get_state("prts-workspace")
        if state is None or state.status != "ready":
            fail(
                f"prts-workspace 未 ready: status={state.status if state else None} "
                f"error={state.error if state else None}"
            )
        ok(f"prts-workspace ready, tools={state.tool_names}")

        # write_document
        result = await registry.invoke(
            "prts-workspace__write_document",
            {"path": "test.md", "content": "# Hello\n\nThis is a test."},
        )
        if isinstance(result, str):
            result = json.loads(result)
        assert_eq(result.get("ok"), True, "write_document 应成功")
        ok("prts-workspace__write_document 成功")

        # list_documents
        result = await registry.invoke(
            "prts-workspace__list_documents",
            {},
        )
        if isinstance(result, str):
            result = json.loads(result)
        assert_eq(result.get("ok"), True, "list_documents 应成功")
        files = result.get("files", [])
        if "test.md" not in files:
            fail(f"list_documents 应包含 test.md,实际: {files}")
        ok(f"prts-workspace__list_documents 返回: {files}")

        # read_document
        result = await registry.invoke(
            "prts-workspace__read_document",
            {"path": "test.md"},
        )
        if isinstance(result, str):
            result = json.loads(result)
        assert_eq(result.get("ok"), True, "read_document 应成功")
        assert_eq(result.get("content"), "# Hello\n\nThis is a test.", "read_document 内容")
        ok("prts-workspace__read_document 内容正确")

        # search_documents
        result = await registry.invoke(
            "prts-workspace__search_documents",
            {"query": "test"},
        )
        if isinstance(result, str):
            result = json.loads(result)
        assert_eq(result.get("ok"), True, "search_documents 应成功")
        results = result.get("results", [])
        if not results:
            fail("search_documents 应返回结果")
        assert_eq(results[0].get("path"), "test.md", "search_documents 命中路径")
        ok(f"prts-workspace__search_documents 命中 {len(results)} 条")

        # 安全性:.. 越界应被拒
        result = await registry.invoke(
            "prts-workspace__read_document",
            {"path": "../etc/passwd"},
        )
        if isinstance(result, str):
            result = json.loads(result)
        if result.get("ok"):
            fail("越界路径 read_document 应失败")
        ok("越界路径被正确拒绝")
    finally:
        await parent_stack.aclose()
        __import__("shutil").rmtree(td, ignore_errors=True)
    ok("prts-workspace MCP 工具链路完整")


async def test_workspace_via_command_lookup() -> None:
    """回归 P4 Bug 1:``mcp.json`` 写 ``"command": "prts-workspace"``
    (不给绝对路径)能被 ``MCPManager`` 找到并启动。

    bug 复现:.venv/Scripts/ 不在 PATH 时,旧版 ``_resolve_command`` 的
    ``shutil.which`` 返回 None,fallback 回原字符串,asyncio spawn 抛
    ``WinError 2``。修复后通过 ``sys.prefix/Scripts`` 兜底。
    """
    registry = ToolRegistry()
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    td = tempfile.mkdtemp()

    try:
        manager = MCPManager(REPO, registry, parent_stack)
        cfg = MCPConfig(
            mcpServers={
                "prts-workspace-cmd": MCPServerConfig(
                    command="prts-workspace",  # 关键:走 venv scripts 兜底
                    env={"PRTS_WORKSPACE_DIR": td},
                    timeout_seconds=10.0,
                )
            }
        )
        await manager.start_all(cfg)

        state = manager.get_state("prts-workspace-cmd")
        if state is None or state.status != "ready":
            fail(
                f"command lookup 失败: status={state.status if state else None} "
                f"error={state.error if state else None}"
            )
        ok(f"prts-workspace 通过 command='prts-workspace' 直接启动 (Bug 1 回归)")
    finally:
        await parent_stack.aclose()
        __import__("shutil").rmtree(td, ignore_errors=True)


async def test_command_not_found_friendly_error() -> None:
    """命令不存在时 manager 应返回友好 error,不抛 WinError 2。"""
    registry = ToolRegistry()
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    try:
        manager = MCPManager(REPO, registry, parent_stack)
        cfg = MCPConfig(
            mcpServers={
                "ghost-server": MCPServerConfig(
                    command="this-command-definitely-does-not-exist-prts",
                    timeout_seconds=2.0,
                )
            }
        )
        await manager.start_all(cfg)
        state = manager.get_state("ghost-server")
        if state is None or state.status != "error":
            fail(
                f"找不到的命令应标 error: status={state.status if state else None}"
            )
        if "not found" not in (state.error or "").lower():
            fail(f"error 字段应含 'not found': {state.error!r}")
        if "Tried" not in (state.error or ""):
            fail(f"error 字段应列出候选路径: {state.error!r}")
        ok(f"ghost command 报友好 error: {state.error[:80]}...")
    finally:
        await parent_stack.aclose()


async def main() -> None:
    await test_workspace_mcp_server()
    await test_workspace_via_command_lookup()
    await test_command_not_found_friendly_error()
    print(f"\n{GREEN}P4 workspace smoke passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
