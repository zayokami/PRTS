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


class SummaryInfo(BaseModel):
    id: str
    summary: str
    key_facts: list[str]
    decisions: list[str]
    todos: list[str]
    message_start: int
    message_end: int
    importance: float
    created_at: str


class SummariesResponse(BaseModel):
    session_id: str
    summaries: list[SummaryInfo]


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


class MCPReloadResponse(BaseModel):
    ok: bool
    servers_ready: int
    servers_error: int
    error: str | None = None


@router.post("/mcp/reload", response_model=MCPReloadResponse)
async def reload_mcp_servers(request: Request) -> MCPReloadResponse:
    """热重载:停止所有 MCP server,重新加载 workspace/mcp.json 并启动。

    改配置后无需重启 Agent,调这个接口即可生效。
    """
    mcp_manager = getattr(request.app.state, "mcp_manager", None)
    workspace = getattr(request.app.state, "workspace_dir", None)
    if mcp_manager is None or workspace is None:
        return MCPReloadResponse(
            ok=False, servers_ready=0, servers_error=0, error="mcp_manager not initialized"
        )

    try:
        from ..mcp import MCPConfigError, load_mcp_config

        mcp_config = load_mcp_config(workspace)
    except MCPConfigError as exc:
        return MCPReloadResponse(
            ok=False, servers_ready=0, servers_error=0, error=f"mcp.json parse error: {exc}"
        )

    try:
        await mcp_manager.reload(mcp_config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("mcp reload failed")
        return MCPReloadResponse(
            ok=False,
            servers_ready=0,
            servers_error=0,
            error=f"{type(exc).__name__}: {exc}",
        )

    states = mcp_manager.states()
    ready = sum(1 for s in states if s.status == "ready")
    errors = sum(1 for s in states if s.status == "error")
    return MCPReloadResponse(ok=True, servers_ready=ready, servers_error=errors)


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


@router.get("/sessions/{session_id}/summaries", response_model=SummariesResponse)
async def get_summaries(session_id: str, request: Request) -> SummariesResponse:
    """返回会话已生成的对话摘要列表(中期记忆)。"""
    store = _store(request)
    rows = await store.get_summaries(session_id, limit=10)
    return SummariesResponse(
        session_id=session_id,
        summaries=[
            SummaryInfo(
                id=r["id"],
                summary=r["summary"],
                key_facts=r.get("key_facts", []),
                decisions=r.get("decisions", []),
                todos=r.get("todos", []),
                message_start=r["message_start"],
                message_end=r["message_end"],
                importance=r.get("importance", 0.5),
                created_at=r["created_at"],
            )
            for r in rows
        ],
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
