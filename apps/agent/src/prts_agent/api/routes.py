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
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from ..loop import AgentLoop
from ..memory import SqliteStore
from ..skills import load_user_skills
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


class MCPServerInfo(BaseModel):
    name: str
    status: str
    disabled: bool
    error: str | None
    tool_names: list[str]
    tools_count: int
    started_at: str | None
    command: str


class MCPServersResponse(BaseModel):
    servers: list[MCPServerInfo]


class TaskInfo(BaseModel):
    name: str
    cron: str | None
    on: str | None


class TasksResponse(BaseModel):
    tasks: list[TaskInfo]


class FsEventRequest(BaseModel):
    changed_files: list[str] = []


class FsEventResponse(BaseModel):
    reloaded: bool
    tasks: list[TaskInfo]
    errors: list[str]


class CronEventRequest(BaseModel):
    task_name: str


class CronEventResponse(BaseModel):
    ok: bool
    result: Any | None = None
    error: str | None = None


def _store(req: Request) -> SqliteStore:
    return req.app.state.store  # type: ignore[no-any-return]


def _tools(req: Request) -> ToolRegistry:
    return req.app.state.tools  # type: ignore[no-any-return]


def _loop(req: Request) -> AgentLoop:
    return req.app.state.agent_loop  # type: ignore[no-any-return]


def _workspace_dir(req: Request) -> Path:
    return req.app.state.workspace_dir  # type: ignore[no-any-return]


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
async def get_history(
    session_id: str, request: Request, limit: int = 500
) -> HistoryResponse:
    """返回会话最近 ``limit`` 条 user/assistant 消息(按时间正序)。

    默认 500 是给 Dashboard 首屏渲染用的"足够大"值;真实长会话超出后,旧消息
    走 P7 向量召回。``limit`` 范围 [1, 5000],超出会被夹到边界,避免单次请求
    把整张表 dump 到客户端。
    """
    # FastAPI 已校验 int 类型,但范围由我们自己卡 —— 否则调用方传 limit=-1
    # 会被 SQLite 当无限,长会话立刻打回 P0 状态。
    limit = max(1, min(limit, 5000))
    store = _store(request)
    rows = await store.history(session_id, limit=limit)
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


@router.get("/mcp/servers", response_model=MCPServersResponse)
async def list_mcp_servers(request: Request) -> MCPServersResponse:
    """已启动 / 失败 / 禁用的外部 MCP server 状态快照。"""
    mcp_manager = getattr(request.app.state, "mcp_manager", None)
    if mcp_manager is None:
        return MCPServersResponse(servers=[])
    return MCPServersResponse(
        servers=[
            MCPServerInfo(
                name=s.name,
                status=s.status,
                disabled=s.disabled,
                error=s.error,
                tool_names=s.tool_names,
                tools_count=s.tools_count,
                started_at=s.started_at,
                command=s.command,
            )
            for s in mcp_manager.states()
        ]
    )


@router.get("/tasks", response_model=TasksResponse)
async def list_tasks(request: Request) -> TasksResponse:
    """返回当前已注册的 @task 列表,供 Rust watcher 获取 cron 调度信息。"""
    loaded = getattr(request.app.state, "skills_loaded", None)
    if loaded is None:
        return TasksResponse(tasks=[])
    return TasksResponse(
        tasks=[
            TaskInfo(name=t.name, cron=t.cron, on=t.on)
            for t in loaded.tasks
        ]
    )


@router.post("/events/fs", response_model=FsEventResponse)
async def handle_fs_event(req: FsEventRequest, request: Request) -> FsEventResponse:
    """文件系统事件:Rust watcher 检测到 skill 文件变化时触发重载。

    重扫 ``workspace/skills/*.py``,把新增的 / 修改的 @skill 和 @task
    重新注册。返回新的 task 列表,方便 watcher 同步 cron 调度。
    """
    workspace = _workspace_dir(request)
    tools = _tools(request)
    try:
        loaded = load_user_skills(workspace, tools)
        # 更新 app.state,让 /tasks 和 /skills 立刻看到新数据
        request.app.state.skills_loaded = loaded
    except Exception as exc:  # noqa: BLE001
        logger.exception("fs event skill reload failed")
        return FsEventResponse(
            reloaded=False,
            tasks=[],
            errors=[f"{type(exc).__name__}: {exc}"],
        )

    return FsEventResponse(
        reloaded=True,
        tasks=[TaskInfo(name=t.name, cron=t.cron, on=t.on) for t in loaded.tasks],
        errors=[err.message for err in loaded.errors],
    )


@router.post("/events/cron", response_model=CronEventResponse)
async def handle_cron_event(req: CronEventRequest, request: Request) -> CronEventResponse:
    """Cron 事件:Rust watcher 按调度触发指定 task 的执行。

    Task 在 Agent 进程内同步执行(非 LLM 流式),因为 task 通常是无头
    的后台作业(定时简报 / 数据同步)。执行时绑定一个虚拟 session,让
    ``prts.client.notify`` 等 SDK 调用有 runtime 可用,但 notify 不推
    给任何前端,仅写入日志。
    """
    loaded = getattr(request.app.state, "skills_loaded", None)
    if loaded is None:
        return CronEventResponse(ok=False, error="skills not loaded yet")

    target = next((t for t in loaded.tasks if t.name == req.task_name), None)
    if target is None:
        return CronEventResponse(
            ok=False, error=f"task {req.task_name!r} not found"
        )

    import asyncio
    import inspect

    try:
        func = target.func
        if inspect.iscoroutinefunction(func):
            result = await func()
        else:
            result = func()
    except Exception as exc:  # noqa: BLE001
        logger.exception("task %s execution failed", req.task_name)
        return CronEventResponse(
            ok=False, error=f"{type(exc).__name__}: {exc}"
        )

    logger.info("task %s finished with result=%r", req.task_name, result)
    return CronEventResponse(ok=True, result=result)
