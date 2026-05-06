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


class FsEventRequest(BaseModel):
    """文件系统事件请求(当前版本为空,预留扩展)。"""

    pass


class FsEventResponse(BaseModel):
    reloaded: bool
    tasks: list[TaskInfo]
    errors: list[str]


class CronEventRequest(BaseModel):
    task_name: str


class CronEventResponse(BaseModel):
    ok: bool
    error: str | None = None
    result: Any | None = None


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


class MCPHealthResponse(BaseModel):
    ok: bool
    results: dict[str, str] = {}
    error: str | None = None


@router.post("/mcp/health-check", response_model=MCPHealthResponse)
async def health_check_mcp_servers(request: Request) -> MCPHealthResponse:
    """健康检查:探测每个 MCP server 的连接是否仍然存活。

    默认自动重启失败的 server;传 ``?auto_restart=false`` 可只检查不重启。
    """
    mcp_manager = getattr(request.app.state, "mcp_manager", None)
    if mcp_manager is None:
        return MCPHealthResponse(ok=False, error="mcp_manager not initialized")

    auto_restart = request.query_params.get("auto_restart", "true").lower() != "false"

    try:
        results = await mcp_manager.health_check(auto_restart=auto_restart)
    except Exception as exc:  # noqa: BLE001
        logger.exception("mcp health check failed")
        return MCPHealthResponse(
            ok=False, error=f"{type(exc).__name__}: {exc}"
        )

    all_healthy = all(s == "healthy" or s == "disabled" for s in results.values())
    return MCPHealthResponse(ok=all_healthy, results=results)


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
    try:
        _validate_session_id(session_id)
    except ValueError as exc:
        return SummariesResponse(session_id=session_id, summaries=[])
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


# task_name -> asyncio.Lock:防止同一个 task 被并发调度多次
_task_execution_locks: dict[str, asyncio.Lock] = {}


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

    # 并发控制:同一 task 同时只能跑一个实例
    lock = _task_execution_locks.setdefault(req.task_name, asyncio.Lock())
    if lock.locked():
        logger.warning("task %s skipped: already running", req.task_name)
        return CronEventResponse(
            ok=False, error=f"task {req.task_name!r} is already running"
        )

    async with lock:
        try:
            func = target.func
            # 30 秒超时:task 应该是轻量后台作业,不应该跑太久
            TASK_TIMEOUT = 30.0
            if inspect.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(), timeout=TASK_TIMEOUT)
            else:
                # 同步函数在线程池中跑,避免阻塞事件循环
                loop = asyncio.get_running_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, func), timeout=TASK_TIMEOUT
                )
        except asyncio.TimeoutError:
            logger.error("task %s timed out after %.1fs", req.task_name, TASK_TIMEOUT)
            return CronEventResponse(
                ok=False, error=f"task {req.task_name!r} timed out after {TASK_TIMEOUT}s"
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("task %s execution failed", req.task_name)
            return CronEventResponse(
                ok=False, error=f"{type(exc).__name__}: {exc}"
            )

    logger.info("task %s finished with result=%r", req.task_name, result)
    return CronEventResponse(ok=True, result=result)


# ---------------------------------------------------------------------------
# Core chat routes (restored from commit 6083fac)
# ---------------------------------------------------------------------------

def _sse_safe_dumps(data: Any) -> str:
    """SSE data 的安全序列化。

    - ``ensure_ascii=False``:中文按字面输出,不让 \\uXXXX 占满帧
    - ``default=str``:工具结果可能包含 datetime 等非 JSON 原生类型,先兜底转字符串
    - U+2028 / U+2029:在 ECMA-404 中当作合法 JSON 字符,但 ECMA-262 之前把它们当作
      行终止符,某些老旧 SSE 中间件 / 浏览器会把帧从中切断 —— 显式转义
    """
    text = json.dumps(data, ensure_ascii=False, default=str)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


@router.post("/converse")
async def converse(req: ConverseRequest, request: Request) -> EventSourceResponse:
    """核心对话接口:接收用户消息,通过 AgentLoop 流式返回 SSE 事件。"""
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
                # 客户端断开时停止生成,避免 LLM 浪费 + DB 写盘成本
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
    """返回会话历史 ``limit`` 条 user/assistant 消息(不含系统消息)。

    默认 500 是给 Dashboard 前端渲染用的"足够大"值;历史会话很长时,旧消息
    由 P7 摘要缩略。``limit`` 范围 [1, 5000],超界会被钳制,避免单请求导致
    巨量 JSON dump 到客户端。
    """
    # FastAPI 会校验 int 类型,但范围校验需要自己做(比如传 limit=-1
    # 会被 SQLite 报错,把会话历史搞崩 P0 状态)。
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
    """返回已注册的 skill 列表(LLM 看到的工具面)。"""
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


# ---------------------------------------------------------------------------
# Request helpers
# ---------------------------------------------------------------------------

def _workspace_dir(request: Request) -> Path:
    """从 app.state 提取 workspace 目录。"""
    path = getattr(request.app.state, "workspace_dir", None)
    if path is None:
        raise RuntimeError("workspace_dir not initialized")
    return path


def _store(request: Request) -> SqliteStore:
    """从 app.state 提取 SQLite store。"""
    store = getattr(request.app.state, "store", None)
    if store is None:
        raise RuntimeError("store not initialized")
    return store


def _tools(request: Request) -> ToolRegistry:
    """从 app.state 提取 ToolRegistry。"""
    tools = getattr(request.app.state, "tools", None)
    if tools is None:
        raise RuntimeError("tools not initialized")
    return tools


def _loop(request: Request) -> AgentLoop:
    """从 app.state 提取 AgentLoop。"""
    loop = getattr(request.app.state, "agent_loop", None)
    if loop is None:
        raise RuntimeError("agent_loop not initialized")
    return loop  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Security helpers
# ---------------------------------------------------------------------------

import re as _re

_SESSION_ID_RE = _re.compile(r"^[\w-]{1,64}$")


def _validate_session_id(session_id: str) -> None:
    """校验 session_id 格式,防止注入非法字符。"""
    if not _SESSION_ID_RE.match(session_id):
        raise ValueError(
            f"invalid session_id: {session_id!r}. "
            "Must be 1-64 chars of [A-Za-z0-9_-]."
        )
