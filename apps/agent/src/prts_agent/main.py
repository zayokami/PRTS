"""PRTS Agent 入口 —— FastAPI HTTP 服务,默认 :4788。

P1 阶段:暴露 /agent/v1/converse 走 SSE 流式 LLM。
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import FastAPI

from .api import router as agent_router

load_dotenv()

app = FastAPI(title="PRTS Agent", version="0.1.0")
app.include_router(agent_router)


@app.get("/health")
async def health() -> dict[str, object]:
    return {
        "service": "prts-agent",
        "ok": True,
        "llm_provider": os.getenv("LLM_PROVIDER", "openai"),
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
        # 只 watch agent 包目录,避免 reload 时扫整个 monorepo(node_modules / target / .venv)
        reload_dirs=["apps/agent/src", "packages/prts-sdk/src"] if reload else None,
    )


if __name__ == "__main__":
    run()
