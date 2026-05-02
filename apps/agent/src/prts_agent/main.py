"""PRTS Agent 入口 —— FastAPI HTTP 服务,默认 :4788。

启动序列:
1. 解析 / 种子 ``~/.prts/workspace``
2. 打开 SQLite,迁移到最新 schema
3. 实例化 LLM 客户端(根据 LLM_PROVIDER 选 OpenAI / Anthropic)
4. 构造 ``ToolRegistry`` + ``AgentRuntimeBridge``,把 bridge 注入 prts SDK
5. **(P4)** 启动 ``workspace/mcp.json`` 里的外部 MCP server,把它们的工具
   以 ``<server>__<tool>`` 注册进 registry(``source="mcp"``)
6. 扫描 ``workspace/skills/*.py`` 把 @skill 注册进来(``source="skill"``)
7. 实例化 ``AgentLoop``,挂到 ``app.state``,供路由使用
"""

from __future__ import annotations

import logging
import os
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

import prts.runtime as prts_runtime

from .api import router as agent_router
from .llm import build_llm_client
from .llm.embedding import build_embedding_client
from .loop import AgentLoop
from .mcp import MCPConfigError, MCPManager, load_mcp_config
from .memory import SqliteStore, init_store
from .runtime import AgentRuntimeBridge
from .skills import load_user_skills
from .tools import ToolRegistry
from .workspace import load_system_prompt, resolve_workspace_dir

load_dotenv()
logging.basicConfig(level=os.getenv("PRTS_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    workspace = resolve_workspace_dir()
    store: SqliteStore = init_store(workspace_dir=workspace)
    await store.ensure_schema()

    llm_client = build_llm_client()
    embedding_client = build_embedding_client()
    tools = ToolRegistry()

    bridge = AgentRuntimeBridge(
        workspace_dir=workspace,
        store=store,
        tools=tools,
        llm_client=llm_client,
        embedding_client=embedding_client,
    )
    prts_runtime.set_runtime(bridge)

    # MCP 先于 skill 启动:skill loader 现在只清 source="skill" 的工具,
    # 所以 MCP 注册的工具不会被 skill 重扫连带杀掉。
    parent_stack = AsyncExitStack()
    await parent_stack.__aenter__()
    started_ok = False
    try:
        mcp_manager = MCPManager(workspace, tools, parent_stack)

        try:
            mcp_config = load_mcp_config(workspace)
        except MCPConfigError as exc:
            logger.error("mcp.json 解析失败,将以空配置启动: %s", exc)
            from .mcp import MCPConfig

            mcp_config = MCPConfig()
        await mcp_manager.start_all(mcp_config)

        loaded = load_user_skills(workspace, tools)

        agent_loop = AgentLoop(
            store=store,
            llm=llm_client,
            tools=tools,
            embedding_client=embedding_client,
        )

        app.state.workspace_dir = workspace
        app.state.store = store
        app.state.llm_client = llm_client
        app.state.tools = tools
        app.state.runtime_bridge = bridge
        app.state.skills_loaded = loaded
        app.state.agent_loop = agent_loop
        app.state.mcp_manager = mcp_manager

        mcp_states = mcp_manager.states()
        mcp_ready = sum(1 for s in mcp_states if s.status == "ready")
        mcp_errors = sum(1 for s in mcp_states if s.status == "error")
        logger.info(
            "agent ready | workspace=%s db=%s skills=%d tasks=%d skill_errors=%d "
            "mcp_servers=%d (ready=%d error=%d)",
            workspace,
            store.db_path,
            len(loaded.skills),
            len(loaded.tasks),
            len(loaded.errors),
            len(mcp_states),
            mcp_ready,
            mcp_errors,
        )
        for err in loaded.errors:
            logger.warning("skill load error in %s: %s", err.file, err.message)
        for state in mcp_states:
            if state.status == "error":
                logger.warning("MCP server %r error: %s", state.name, state.error)
        started_ok = True
    finally:
        if not started_ok:
            # 启动半截抛了:确保已经登记的子进程 / runtime 都被释放,不留泄漏。
            try:
                await parent_stack.aclose()
            except Exception:  # noqa: BLE001
                logger.exception("parent_stack 关闭异常 (启动失败路径)")
            prts_runtime.set_runtime(None)

    try:
        yield
    finally:
        # 先关 MCP(可能要等子进程优雅退出),再清 SDK runtime 引用
        await parent_stack.aclose()
        prts_runtime.set_runtime(None)


app = FastAPI(title="PRTS Agent", version="0.1.0", lifespan=lifespan)
app.include_router(agent_router)


@app.get("/health")
async def health() -> dict[str, object]:
    workspace = getattr(app.state, "workspace_dir", None)
    store = getattr(app.state, "store", None)
    tools = getattr(app.state, "tools", None)
    loaded = getattr(app.state, "skills_loaded", None)
    mcp_manager = getattr(app.state, "mcp_manager", None)
    mcp_states = mcp_manager.states() if mcp_manager else []
    return {
        "service": "prts-agent",
        "ok": True,
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "workspace_dir": str(workspace) if workspace else None,
        "db_path": str(store.db_path) if store else None,
        "system_prompt_chars": len(load_system_prompt(workspace)) if workspace else 0,
        "tools_count": len(tools.all()) if tools else 0,
        "skills_loaded": len(loaded.skills) if loaded else 0,
        "skills_errors": len(loaded.errors) if loaded else 0,
        "tasks_loaded": len(loaded.tasks) if loaded else 0,
        "mcp_servers": len(mcp_states),
        "mcp_servers_ready": sum(1 for s in mcp_states if s.status == "ready"),
        "mcp_servers_error": sum(1 for s in mcp_states if s.status == "error"),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


def run() -> None:
    """uvicorn 启动入口,被 [project.scripts].prts-agent 调用。"""
    import uvicorn

    port = int(os.getenv("AGENT_PORT", "4788"))
    reload = os.getenv("PRTS_DEV", "1") == "1"
    uvicorn.run(
        "prts_agent.main:app",
        host="127.0.0.1",
        port=port,
        reload=reload,
        reload_dirs=["apps/agent/src", "packages/prts-sdk/src"] if reload else None,
    )


if __name__ == "__main__":
    run()
