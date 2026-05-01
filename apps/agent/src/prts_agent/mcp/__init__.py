"""PRTS 的 MCP 客户端层。

P4:把 ``workspace/mcp.json`` 里声明的外部 MCP server(stdio 子进程)启动起来,
list_tools 后把每个工具以 ``<server_name>__<tool_name>`` 的名字注册进 ToolRegistry,
LLM 后续调用就跟调本地 @skill 一样透明。

仅做客户端 —— 暴露 PRTS 自己功能为 MCP server 的事 P7 再说。
"""

from __future__ import annotations

from .config import (
    MCPConfig,
    MCPConfigError,
    MCPServerConfig,
    expand_variables,
    load_mcp_config,
)
from .invoker import make_mcp_invoker, register_server_tools
from .manager import MCPManager, MCPServerState

__all__ = [
    "MCPConfig",
    "MCPConfigError",
    "MCPManager",
    "MCPServerConfig",
    "MCPServerState",
    "expand_variables",
    "load_mcp_config",
    "make_mcp_invoker",
    "register_server_tools",
]
