"""PRTS Agent 入口 —— FastAPI HTTP 服务,默认 :4788。

启动序列(P3):
1. 解析 / 种子 ``~/.prts/workspace``
2. 打开 SQLite,迁移到最新 schema
3. 实例化 LLM 客户端(根据 LLM_PROVIDER 选 OpenAI / Anthropic)
4. 构造 ``ToolRegistry``,扫描 ``workspace/skills/*.py`` 把 @skill 注册进来
5. 构造 ``AgentRuntimeBridge`` 并 ``prts.runtime.set_runtime(...)`` 注入 SDK
6. 实例化 ``AgentLoop``,挂到 ``app.state``,供路由使用
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

import prts.runtime as prts_runtime

from .api import router as agent_router
from .llm import build_llm_client
from .loop import AgentLoop
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
    store: SqliteStore = init_store()
    await store.ensure_schema()

    llm_client = build_llm_client()
    tools = ToolRegistry()

    bridge = AgentRuntimeBridge(
        workspace_dir=workspace, store=store, tools=tools, llm_client=llm_client
    )
    prts_runtime.set_runtime(bridge)

    loaded = load_user_skills(workspace, tools)

    agent_loop = AgentLoop(store=store, llm=llm_client, tools=tools)

    app.state.workspace_dir = workspace
    app.state.store = store
    app.state.llm_client = llm_client
    app.state.tools = tools
    app.state.runtime_bridge = bridge
    app.state.skills_loaded = loaded
    app.state.agent_loop = agent_loop

    logger.info(
        "agent ready | workspace=%s db=%s skills=%d tasks=%d errors=%d",
        workspace,
        store.db_path,
        len(loaded.skills),
        len(loaded.tasks),
        len(loaded.errors),
    )
    for err in loaded.errors:
        logger.warning("skill load error in %s: %s", err.file, err.message)

    try:
        yield
    finally:
        prts_runtime.set_runtime(None)


app = FastAPI(title="PRTS Agent", version="0.1.0", lifespan=lifespan)
app.include_router(agent_router)


@app.get("/health")
async def health() -> dict[str, object]:
    workspace = getattr(app.state, "workspace_dir", None)
    store = getattr(app.state, "store", None)
    tools = getattr(app.state, "tools", None)
    loaded = getattr(app.state, "skills_loaded", None)
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
