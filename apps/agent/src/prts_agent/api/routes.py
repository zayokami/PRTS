"""Agent HTTP 路由。

P1: /agent/v1/converse 单轮 LLM 流式。
P2: 接 SQLite 持久化 + workspace markdown 注入 system prompt;
    新增 GET /agent/v1/sessions/{id}/history 给前端在重连时拉历史。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..llm import ChatMessage, build_llm_client
from ..memory import SqliteStore
from ..workspace import load_system_prompt

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent/v1")

Role = Literal["system", "user", "assistant", "tool"]


class ConverseRequest(BaseModel):
    session_id: str
    content: str
    channel: str = "web"
    user_ref: str | None = None


class HistoryMessage(BaseModel):
    role: Role
    content: str
    created_at: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: list[HistoryMessage]


def _store(req: Request) -> SqliteStore:
    return req.app.state.store  # type: ignore[no-any-return]


@router.post("/converse")
async def converse(req: ConverseRequest, request: Request) -> EventSourceResponse:
    store = _store(request)
    workspace_dir = request.app.state.workspace_dir

    await store.ensure_session(req.session_id, channel=req.channel, user_ref=req.user_ref)
    await store.append_message(req.session_id, "user", req.content)

    history = await store.history(req.session_id)
    system_prompt = load_system_prompt(workspace_dir)

    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.extend(
        ChatMessage(role=m.role, content=m.content)
        for m in history
        if m.role in ("user", "assistant")
    )

    client = build_llm_client()

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        assistant_text = ""
        try:
            async for token in client.stream_chat(messages):
                assistant_text += token
                yield {"event": "token", "data": json.dumps({"text": token})}
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM stream failed")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc), "type": type(exc).__name__}),
            }
            return
        finally:
            if assistant_text:
                await store.append_message(req.session_id, "assistant", assistant_text)

        yield {"event": "done", "data": json.dumps({"session_id": req.session_id})}

    return EventSourceResponse(event_stream())


@router.get("/sessions/{session_id}/history", response_model=HistoryResponse)
async def get_history(session_id: str, request: Request) -> HistoryResponse:
    store = _store(request)
    rows = await store.history(session_id)
    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(role=m.role, content=m.content, created_at=m.created_at)
            for m in rows
            if m.role in ("user", "assistant")
        ],
    )
