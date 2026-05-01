"""PRTS Agent 入口 —— FastAPI HTTP 服务,默认 :4788。

P1: SSE 流式 LLM。
P2: workspace markdown system prompt + SQLite 持久化。
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

from .api import router as agent_router
from .memory import SqliteStore, init_store
from .workspace import load_system_prompt, resolve_workspace_dir

load_dotenv()
logging.basicConfig(level=os.getenv("PRTS_LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
    workspace = resolve_workspace_dir()
    store: SqliteStore = init_store()
    await store.ensure_schema()

    app.state.workspace_dir = workspace
    app.state.store = store
    logger.info("workspace=%s db=%s", workspace, store.db_path)
    yield


app = FastAPI(title="PRTS Agent", version="0.1.0", lifespan=lifespan)
app.include_router(agent_router)


@app.get("/health")
async def health() -> dict[str, object]:
    workspace = getattr(app.state, "workspace_dir", None)
    store = getattr(app.state, "store", None)
    return {
        "service": "prts-agent",
        "ok": True,
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
        "workspace_dir": str(workspace) if workspace else None,
        "db_path": str(store.db_path) if store else None,
        "system_prompt_chars": len(load_system_prompt(workspace)) if workspace else 0,
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
