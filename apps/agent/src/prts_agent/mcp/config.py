"""``workspace/mcp.json`` 的解析 + 变量展开。

格式参照 Claude Desktop / Cursor 的 ``claude_desktop_config.json``::

    {
      "mcpServers": {
        "filesystem": {
          "command": "npx",
          "args": ["-y", "@modelcontextprotocol/server-filesystem", "${WORKSPACE_DIR}"],
          "env": {},
          "disabled": false,
          "timeout_seconds": 30
        }
      }
    }

变量替换在加载时做,而不是 server 启动时:这样配错(比如未定义的 env 变量)立刻报。
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)

CONFIG_FILENAME = "mcp.json"

# ${WORKSPACE_DIR} 或 ${env:VAR_NAME}
_VAR_RE = re.compile(r"\$\{(WORKSPACE_DIR|env:[A-Za-z_][A-Za-z0-9_]*)\}")


class MCPConfigError(ValueError):
    """mcp.json 解析失败 —— JSON 语法错误或 schema 不合法。"""


class MCPServerConfig(BaseModel):
    """单个 MCP server 的启动参数。"""

    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    disabled: bool = False
    timeout_seconds: float = 30.0


class MCPConfig(BaseModel):
    """整个 mcp.json 的根结构。"""

    model_config = ConfigDict(extra="forbid")

    mcpServers: dict[str, MCPServerConfig] = Field(default_factory=dict)


def expand_variables(value: str, workspace_dir: Path) -> str:
    """把 ``${WORKSPACE_DIR}`` / ``${env:NAME}`` 替换成实际值。

    - ``${WORKSPACE_DIR}`` → 用户工作区绝对路径(POSIX 分隔符,跨平台一致,npx
      在 Windows 上也能吃)
    - ``${env:NAME}`` → ``os.environ.get(NAME, "")``;未定义只 warn 一次,不抛
    """

    def _resolve(match: re.Match[str]) -> str:
        var = match.group(1)
        if var == "WORKSPACE_DIR":
            return workspace_dir.resolve().as_posix()
        # ${env:NAME}
        env_name = var.split(":", 1)[1]
        env_val = os.environ.get(env_name)
        if env_val is None:
            logger.warning(
                "mcp.json 引用了未定义的环境变量 %r,将替换为空字符串", env_name
            )
            return ""
        return env_val

    return _VAR_RE.sub(_resolve, value)


def _expand_in_config(cfg: MCPServerConfig, workspace_dir: Path) -> MCPServerConfig:
    """对一个 MCPServerConfig 的字符串字段做变量展开,返回新实例。"""
    return cfg.model_copy(
        update={
            "command": expand_variables(cfg.command, workspace_dir),
            "args": [expand_variables(a, workspace_dir) for a in cfg.args],
            "env": {k: expand_variables(v, workspace_dir) for k, v in cfg.env.items()},
            "cwd": expand_variables(cfg.cwd, workspace_dir) if cfg.cwd else None,
        }
    )


def load_mcp_config(workspace_dir: Path) -> MCPConfig:
    """读取 ``workspace_dir/mcp.json`` 并展开变量。

    缺文件 → 静默返回空配置。JSON 错或 schema 错 → ``MCPConfigError``。
    """
    cfg_path = workspace_dir / CONFIG_FILENAME
    if not cfg_path.is_file():
        logger.info("mcp.json 不存在 (%s),跳过 MCP 启动", cfg_path)
        return MCPConfig()

    try:
        raw_text = cfg_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MCPConfigError(f"读取 {cfg_path} 失败: {exc}") from exc

    try:
        raw: Any = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise MCPConfigError(
            f"{cfg_path} JSON 解析失败 (line {exc.lineno} col {exc.colno}): {exc.msg}"
        ) from exc

    try:
        config = MCPConfig.model_validate(raw)
    except ValidationError as exc:
        raise MCPConfigError(f"{cfg_path} schema 不合法:\n{exc}") from exc

    expanded_servers = {
        name: _expand_in_config(srv, workspace_dir)
        for name, srv in config.mcpServers.items()
    }
    return MCPConfig(mcpServers=expanded_servers)
