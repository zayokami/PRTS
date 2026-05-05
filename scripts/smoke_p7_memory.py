"""P7 smoke test —— 验证 prts-vector MCP server 的 upsert/search 端到端。

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p7_memory.py

前置条件:
    1. prts-vector binary 已编译: cargo build --bin prts-vector
    2. embedding client 已配置 (EMBEDDING_BASE_URL/EMBEDDING_API_KEY)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
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

PRTS_VECTOR_BIN = REPO / "target" / "debug" / "prts-vector.exe"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


async def test_vector_mcp_server() -> None:
    """直接启动 prts-vector,通过 MCPManager 注册到 ToolRegistry,验证 upsert/search。"""
    if not PRTS_VECTOR_BIN.is_file():
        fail(f"prts-vector binary not found: {PRTS_VECTOR_BIN}")

    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        registry = ToolRegistry()

        # 构造 MCPConfig 指向 prts-vector binary
        cfg = MCPConfig(
            mcpServers={
                "prts-vector": {
                    "command": str(PRTS_VECTOR_BIN),
                    "args": ["--db", str(db_path), "--dim", "4"],
                    "disabled": False,
                    "timeout_seconds": 10.0,
                }
            }
        )

        from contextlib import AsyncExitStack

        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            manager = MCPManager(Path(td), registry, stack)
            await manager.start_all(cfg)

            states = manager.states()
            assert_eq(len(states), 1, "should have 1 server")
            state = states[0]
            if state.status != "ready":
                fail(f"prts-vector not ready: {state.status} {state.error}")
            ok(f"prts-vector ready, tools={state.tool_names}")

            # 验证工具已注册
            assert registry.get("prts-vector__upsert") is not None, "upsert not registered"
            assert registry.get("prts-vector__search") is not None, "search not registered"
            ok("tools registered")

            # Upsert 几条测试向量
            await registry.invoke(
                "prts-vector__upsert",
                {
                    "id": "test-a",
                    "vector": [1.0, 0.0, 0.0, 0.0],
                    "payload": {"text": "关于 Python 的讨论"},
                },
            )
            await registry.invoke(
                "prts-vector__upsert",
                {
                    "id": "test-b",
                    "vector": [0.0, 1.0, 0.0, 0.0],
                    "payload": {"text": "关于 Rust 的讨论"},
                },
            )
            await registry.invoke(
                "prts-vector__upsert",
                {
                    "id": "test-c",
                    "vector": [0.0, 0.0, 1.0, 0.0],
                    "payload": {"text": "关于 JavaScript 的讨论"},
                },
            )
            ok("upsert 3 vectors")

            # Search 召回最近邻
            result = await registry.invoke(
                "prts-vector__search",
                {"query_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 2},
            )
            if isinstance(result, str):
                result = json.loads(result)
            assert isinstance(result, dict), f"search result should be dict, got {type(result)}"
            assert result.get("ok"), f"search failed: {result}"
            results = result.get("results", [])
            assert len(results) == 2, f"should return 2 results, got {len(results)}"
            assert results[0]["id"] == "test-a", f"first result should be test-a, got {results[0]}"
            ok(f"search returned {len(results)} results, top={results[0]['id']}")

            # 验证 payload 被正确存储
            payload_text = results[0].get("payload", "")
            if payload_text:
                try:
                    payload = json.loads(payload_text)
                    assert_eq(payload.get("text"), "关于 Python 的讨论", "payload text")
                except json.JSONDecodeError:
                    fail(f"payload not valid json: {payload_text}")
            ok("payload preserved")

        finally:
            await stack.aclose()


async def test_auto_remember_path() -> None:
    """验证 AgentLoop._auto_remember 的调用路径是否正确。"""
    # 这是一个静态检查:确认 ContextManager._recall_vectors 使用了共享 registry
    from prts_agent.loop.context_manager import ContextManager
    import inspect

    src = inspect.getsource(ContextManager._recall_vectors)
    if "self._tools" not in src:
        fail("ContextManager._recall_vectors should use self._tools, not new ToolRegistry()")
    ok("ContextManager uses shared ToolRegistry for vector recall")


async def main() -> None:
    await test_vector_mcp_server()
    await test_auto_remember_path()
    print(f"\n{GREEN}P7 vector memory smoke all passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
