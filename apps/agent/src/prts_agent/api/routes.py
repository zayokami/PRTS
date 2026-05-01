"""Agent HTTP 路由。

P1 阶段只实现 /agent/v1/converse:接收一条用户消息,LLM 流式回复经 SSE 推回。
P2 起接入持久化 + workspace markdown system prompt;P3 起接 skills/MCP 工具。
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..llm import ChatMessage, build_llm_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/agent/v1")


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ConverseRequest(BaseModel):
    session_id: str
    messages: list[Message]


@router.post("/converse")
async def converse(req: ConverseRequest) -> EventSourceResponse:
    """SSE 流:每个 token 一个 `token` 事件,结束发 `done`,异常发 `error`。"""

    client = build_llm_client()
    chat_messages: list[ChatMessage] = [
        ChatMessage(role=m.role, content=m.content) for m in req.messages
    ]

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for token in client.stream_chat(chat_messages):
                yield {"event": "token", "data": json.dumps({"text": token})}
            yield {"event": "done", "data": json.dumps({"session_id": req.session_id})}
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM stream failed")
            yield {
                "event": "error",
                "data": json.dumps({"message": str(exc), "type": type(exc).__name__}),
            }

    return EventSourceResponse(event_stream())
