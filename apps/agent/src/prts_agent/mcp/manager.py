"""MCP server 生命周期管理 —— 启动 / 状态 / 停止由 AsyncExitStack 兜底。

设计要点:
- 每个 server 独占一个 ``AsyncExitStack``,挂在外部传入的 ``parent_stack`` 下。
  这样某个 server 关闭卡住或抛异常,不会拖死其他兄弟 server。
- ``_start_one`` 内部所有异常被吞 → 标 ``status="error"``,manager 永不向上抛。
  Agent boot 失败一个 server 不应该让整个 Agent 起不来。
- Windows ``npx`` 必须用 ``npx.cmd`` 否则 WinError 2,平台检测在 ``_resolve_command``。
- 启动后第一件事是 ``list_tools`` + 注册到 ``ToolRegistry``;失败也归入 error 状态。
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from .config import MCPConfig, MCPServerConfig
from .invoker import register_server_tools

if TYPE_CHECKING:
    from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

ServerStatus = Literal["disabled", "starting", "ready", "error", "stopped"]

# Windows 上必须带 .cmd 才能 spawn 的常见 Node / Python 工具
_WIN_CMD_WRAPPERS = {"npx", "npm", "yarn", "pnpm", "uvx", "bunx", "bun"}


def _resolve_command(command: str) -> tuple[str | None, list[str]]:
    """把 mcp.json 里的 ``command`` 解析成可 spawn 的可执行路径。

    返回 ``(resolved, tried)``。``resolved`` 是绝对路径或 None(找不到);
    ``tried`` 是探测过的候选名单,用于在 manager 出错时给用户看。

    解析顺序:

    1. **Windows wrapper 名单**(``npx`` / ``uvx`` / ``pnpm`` 等):先试 ``.cmd``。
       Windows ``CreateProcess`` 不像 cmd.exe 那样自动应用 PATHEXT,裸 ``npx`` 会
       ``WinError 2``,必须显式带后缀。
    2. ``shutil.which(command)``:覆盖 PATH 上的二进制 + 用户写绝对路径的 case。
    3. **venv 兜底**:``.venv/Scripts/<cmd>{.exe,.cmd,.bat}``(Win)或
       ``.venv/bin/<cmd>``(POSIX)。Agent 跑在 venv 里时,venv 里的 console
       script(比如 ``prts-workspace``)在 PATH 里通常找不到,因为 venv 没被
       activate;但 ``sys.prefix`` 一定指向当前 venv,可靠。
    """
    tried: list[str] = []

    if sys.platform == "win32":
        base = command.lower()
        if base in _WIN_CMD_WRAPPERS:
            candidate = command + ".cmd"
            tried.append(candidate)
            found = shutil.which(candidate)
            if found:
                return found, tried
        tried.append(command)
        found = shutil.which(command)
        if found:
            # 返回完整路径(含 .exe/.cmd)才能让 CreateProcess 直接跑。
            return found, tried
        venv_scripts = Path(sys.prefix) / "Scripts"
        for ext in (".exe", ".cmd", ".bat", ""):
            cand = venv_scripts / f"{command}{ext}"
            tried.append(str(cand))
            if cand.is_file():
                return str(cand), tried
        return None, tried

    # POSIX
    tried.append(command)
    found = shutil.which(command)
    if found:
        return found, tried
    cand = Path(sys.prefix) / "bin" / command
    tried.append(str(cand))
    if cand.is_file():
        return str(cand), tried
    return None, tried


@dataclass
class MCPServerState:
    """供 ``/mcp/servers`` 路由读取的 server 状态快照。"""

    name: str
    status: ServerStatus
    disabled: bool = False
    error: str | None = None
    tool_names: list[str] = field(default_factory=list)
    started_at: str | None = None
    command: str = ""

    @property
    def tools_count(self) -> int:
        return len(self.tool_names)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "status": self.status,
            "disabled": self.disabled,
            "error": self.error,
            "tool_names": list(self.tool_names),
            "tools_count": self.tools_count,
            "started_at": self.started_at,
            "command": self.command,
        }


class MCPManager:
    """启动 / 跟踪一组 MCP server。

    Parameters
    ----------
    workspace_dir
        当前用户工作区(用于错误日志关联)。
    registry
        共享的 ToolRegistry,MCP 工具以 ``source="mcp"`` 写入。
    parent_stack
        FastAPI lifespan 持有的总 AsyncExitStack。每个 server 自己的子 stack
        会 push 到这里,stack 关闭时统一退出。
    """

    def __init__(
        self,
        workspace_dir: Path,
        registry: "ToolRegistry",
        parent_stack: AsyncExitStack,
    ) -> None:
        self._workspace_dir = workspace_dir
        self._registry = registry
        self._parent_stack = parent_stack
        self._states: dict[str, MCPServerState] = {}

    def states(self) -> list[MCPServerState]:
        return list(self._states.values())

    def get_state(self, name: str) -> MCPServerState | None:
        return self._states.get(name)

    async def start_all(self, config: MCPConfig) -> None:
        """串行启动所有 server。失败的标 error,不抛。

        串行而不是并发是有意的:LLM 启动期同时 spawn 5 个 npx 子进程,Windows 上
        容易撞 spawn race。MVP 单用户场景启动延迟不是瓶颈,顺序起更省心。
        """
        for name, cfg in config.mcpServers.items():
            await self._start_one(name, cfg)

    async def _start_one(self, name: str, cfg: MCPServerConfig) -> None:
        if cfg.disabled:
            self._states[name] = MCPServerState(
                name=name,
                status="disabled",
                disabled=True,
                command=cfg.command,
            )
            logger.info("MCP server %r disabled, skipping", name)
            return

        # 占位:starting 状态先写进去,失败时也能看到曾经尝试过
        self._states[name] = MCPServerState(
            name=name,
            status="starting",
            command=cfg.command,
        )

        # 局部 import:mcp 1.x 的导入路径仅在 mcp 装上后才存在,模块级 import 会让
        # 单测 / 静态环境(没装 mcp)直接挂掉。
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:
            self._states[name] = MCPServerState(
                name=name,
                status="error",
                error=f"mcp SDK 未安装: {exc}",
                command=cfg.command,
            )
            logger.error("MCP SDK missing — install mcp>=1.27 (server=%r)", name)
            return

        resolved_command, tried = _resolve_command(cfg.command)
        if resolved_command is None:
            # spawn 之前先拦截:报"找不到命令"是 P4 启动失败最常见的原因,
            # 直接把候选名单写进 error,用户不用 Ctrl-F 翻日志。
            self._states[name] = MCPServerState(
                name=name,
                status="error",
                error=(
                    f"command {cfg.command!r} not found. "
                    f"Tried: {tried}. "
                    "Install the binary to PATH or use an absolute path in mcp.json."
                ),
                command=cfg.command,
            )
            logger.error(
                "MCP server %r: command %r not found, tried %s",
                name, cfg.command, tried,
            )
            return

        params = StdioServerParameters(
            command=resolved_command,
            args=list(cfg.args),
            env=dict(cfg.env) if cfg.env else None,
            cwd=cfg.cwd,
        )

        # 每 server 一个子 stack。子 stack 在 parent stack 里保存为 callback,
        # parent stack 关闭时调用 child.aclose() —— 如果某 server 关闭卡住,
        # 顺序退出会被它阻塞,但至少其他 server 已经先退过。
        child_stack = AsyncExitStack()
        await child_stack.__aenter__()

        async def _close_child(stack: AsyncExitStack = child_stack) -> None:
            try:
                await stack.aclose()
            except Exception as close_exc:  # noqa: BLE001
                logger.warning("MCP server %r 关闭异常 (吞掉): %s", name, close_exc)

        self._parent_stack.push_async_callback(_close_child)

        try:
            async def _bring_up() -> list[str]:
                transport = await child_stack.enter_async_context(stdio_client(params))
                read_stream, write_stream = transport
                session = await child_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()
                tools_response = await session.list_tools()
                return register_server_tools(
                    server_name=name,
                    session=session,
                    tools_response=tools_response,
                    registry=self._registry,
                    timeout_s=cfg.timeout_seconds,
                )

            # 整个握手 + list_tools 都纳入超时:坏的 MCP server(比如 spawn 起来
            # 但永远不响应 initialize)否则会让 agent 启动期永远卡住。``call_tool``
            # 自己另有 wait_for,这里只兜启动这段。
            tool_names = await asyncio.wait_for(
                _bring_up(), timeout=cfg.timeout_seconds
            )
        except asyncio.TimeoutError:
            logger.error(
                "MCP server %r 启动超时 (>%.1fs)", name, cfg.timeout_seconds
            )
            self._states[name] = MCPServerState(
                name=name,
                status="error",
                error=f"startup timed out after {cfg.timeout_seconds}s",
                command=cfg.command,
            )
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP server %r 启动失败", name)
            self._states[name] = MCPServerState(
                name=name,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
                command=cfg.command,
            )
            # child_stack 已经登记了 close callback,parent_stack.aclose 时会兜底
            return

        self._states[name] = MCPServerState(
            name=name,
            status="ready",
            tool_names=tool_names,
            started_at=datetime.now(tz=timezone.utc).isoformat(),
            command=cfg.command,
        )
        logger.info(
            "MCP server %r ready, %d tool(s) registered",
            name,
            len(tool_names),
        )
