"""P7 smoke test —— 验证 Rust prts-vector MCP server + Python embedding 集成。

需要:
- ``cargo build --bin prts-vector`` 已跑过(binary 在 ``target/debug/prts-vector``)
- 不需要真实 embedding API(用 FakeEmbeddingClient 代替)

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p7.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from contextlib import AsyncExitStack
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "agent" / "src"))
sys.path.insert(0, str(REPO / "packages" / "prts-sdk" / "src"))
sys.dont_write_bytecode = True

from prts_agent.mcp import MCPConfig, MCPManager, MCPServerConfig  # noqa: E402
from prts_agent.mcp.invoker import register_server_tools  # noqa: E402
from prts_agent.memory import init_store  # noqa: E402
from prts_agent.runtime import AgentRuntimeBridge  # noqa: E402
from prts_agent.tools import ToolRegistry  # noqa: E402

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"

if sys.platform == "win32":
    VECTOR_BIN = REPO / "target" / "debug" / "prts-vector.exe"
else:
    VECTOR_BIN = REPO / "target" / "debug" / "prts-vector"


class FakeEmbeddingClient:
    """返回固定 4 维向量,与 ``--dim 4`` 对齐。"""

    async def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0, 0.0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> str:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


async def test_vector_mcp_server() -> None:
    """启动 prts-vector,验证 upsert + search 工具。"""
    if not VECTOR_BIN.is_file():
        fail(f"prts-vector binary 未找到: {VECTOR_BIN}")

    registry = ToolRegistry()
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    td = tempfile.mkdtemp()

    try:
        manager = MCPManager(REPO, registry, parent_stack)
        db_path = Path(td) / "vec.db"
        cfg = MCPConfig(
            mcpServers={
                "prts-vector": MCPServerConfig(
                    command=str(VECTOR_BIN),
                    args=["--db", str(db_path), "--dim", "4"],
                    timeout_seconds=10.0,
                )
            }
        )
        await manager.start_all(cfg)

        state = manager.get_state("prts-vector")
        if state is None or state.status != "ready":
            fail(
                f"prts-vector 未 ready: status={state.status if state else None} "
                f"error={state.error if state else None}"
            )
        ok(f"prts-vector ready, tools={state.tool_names}")

        # upsert
        await registry.invoke(
            "prts-vector__upsert",
            {"id": "a", "vector": [1.0, 0.0, 0.0, 0.0], "payload": None},
        )
        ok("prts-vector__upsert 成功")

        # search
        result = await registry.invoke(
            "prts-vector__search",
            {"query_vector": [1.0, 0.0, 0.0, 0.0], "top_k": 3},
        )
        ok(f"prts-vector__search 返回: {result!r}")

        # 解析返回值(可能是字符串 JSON 或已被 invoker 拆开的 dict)
        if isinstance(result, str):
            result = __import__("json").loads(result)
        if not (isinstance(result, dict) and result.get("ok")):
            fail(f"search 返回非成功结构: {result}")
        results = result.get("results", [])
        if not results:
            fail("search 未返回任何结果")
        assert_eq(results[0]["id"], "a", "最近邻应为 a")
    finally:
        await parent_stack.aclose()
        __import__("shutil").rmtree(td, ignore_errors=True)
    ok("prts-vector MCP 工具链路完整")


async def test_bridge_memory() -> None:
    """AgentRuntimeBridge.remember / search_memory 走通。"""
    registry = ToolRegistry()
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    td = tempfile.mkdtemp()

    try:
        manager = MCPManager(REPO, registry, parent_stack)
        db_path = Path(td) / "vec2.db"
        cfg = MCPConfig(
            mcpServers={
                "prts-vector": MCPServerConfig(
                    command=str(VECTOR_BIN),
                    args=["--db", str(db_path), "--dim", "4"],
                    timeout_seconds=10.0,
                )
            }
        )
        await manager.start_all(cfg)

        store = init_store(workspace_dir=Path(td))
        await store.ensure_schema()

        bridge = AgentRuntimeBridge(
            workspace_dir=Path(td),
            store=store,
            tools=registry,
            llm_client=None,  # type: ignore[arg-type]
            embedding_client=FakeEmbeddingClient(),
        )

        await bridge.remember("测试文本", payload={"id": "smoke-1", "tag": "smoke"})
        ok("bridge.remember 成功")

        hits = await bridge.search_memory("测试", top_k=3)
        if not hits:
            fail("search_memory 未返回结果")
        assert_eq(hits[0]["id"], "smoke-1", "search_memory 命中 id")
        ok(f"bridge.search_memory 返回 {len(hits)} 条")
    finally:
        await parent_stack.aclose()
        __import__("shutil").rmtree(td, ignore_errors=True)


async def test_auto_remember_skipped_without_embedding() -> None:
    """没有 embedding client 时,_auto_remember 静默跳过。"""
    from prts_agent.loop.runner import AgentLoop  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        store = init_store(workspace_dir=Path(td))
        await store.ensure_schema()
        loop = AgentLoop(store=store, llm=None, tools=ToolRegistry(), embedding_client=None)  # type: ignore[arg-type]
        # _auto_remember 是协程,直接 await 不应抛
        await loop._auto_remember("sid-1", "hi", "hello", "web")
        ok("无 embedding client 时 _auto_remember 静默跳过")


async def main() -> None:
    await test_vector_mcp_server()
    await test_bridge_memory()
    await test_auto_remember_skipped_without_embedding()
    print(f"\n{GREEN}P7 smoke all passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
