"""Agent HTTP 路由。

P3:
- ``POST /agent/v1/converse`` 改走 ``AgentLoop``,SSE 事件类型扩到
  ``token`` / ``tool_call`` / ``tool_result`` / ``notify`` / ``done`` / ``error``
- ``GET  /agent/v1/sessions/{id}/history`` 仍然返回 user/assistant 消息
- ``GET  /agent/v1/skills`` 列已注册的 skill(LLM 看到的工具面)
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..loop import AgentLoop
from ..memory import SqliteStore
from ..tools import ToolRegistry
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


class SkillInfo(BaseModel):
    name: str
    description: str | None
    input_schema: dict[str, Any]
    source: str


class SkillsResponse(BaseModel):
    skills: list[SkillInfo]


def _store(req: Request) -> SqliteStore:
    return req.app.state.store  # type: ignore[no-any-return]


def _tools(req: Request) -> ToolRegistry:
    return req.app.state.tools  # type: ignore[no-any-return]


def _loop(req: Request) -> AgentLoop:
    return req.app.state.agent_loop  # type: ignore[no-any-return]


def _sse_safe_dumps(data: Any) -> str:
    """SSE 帧 data 的安全序列化。

    - ``ensure_ascii=False``:中文按字面输出,不让 \\uXXXX 占满帧
    - ``default=str``:工具结果可能含 datetime 之类非 JSON 原生类型,先兜底转字符串
    - U+2028 / U+2029:被 ECMA-404 当作合法 JSON 字符,但 ECMA-262 之前把它们当
      行终止符,某些老旧 SSE 中间件 / 浏览器会把帧从中切断 —— 显式转义
    """
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


@router.post("/converse")
async def converse(req: ConverseRequest, request: Request) -> EventSourceResponse:
    workspace_dir = request.app.state.workspace_dir
    system_prompt = load_system_prompt(workspace_dir)
    loop = _loop(request)

    async def event_stream() -> AsyncIterator[dict[str, str]]:
        try:
            async for evt in loop.converse(
                session_id=req.session_id,
                user_content=req.content,
                system_prompt=system_prompt,
                channel=req.channel,
                user_ref=req.user_ref,
            ):
                # 客户端断开时停止生成,放掉 LLM 网络 + DB 写盘成本
                if await request.is_disconnected():
                    logger.info(
                        "client disconnected, aborting converse for session=%s",
                        req.session_id,
                    )
                    break
                yield {"event": evt["event"], "data": _sse_safe_dumps(evt["data"])}
        except Exception as exc:  # noqa: BLE001
            logger.exception("converse loop failed")
            yield {
                "event": "error",
                "data": _sse_safe_dumps(
                    {"message": str(exc), "type": type(exc).__name__}
                ),
            }

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


@router.get("/skills", response_model=SkillsResponse)
async def list_skills(request: Request) -> SkillsResponse:
    tools = _tools(request)
    return SkillsResponse(
        skills=[
            SkillInfo(
                name=t.name,
                description=t.description,
                input_schema=t.input_schema,
                source=t.source,
            )
            for t in tools.all()
        ]
    )
