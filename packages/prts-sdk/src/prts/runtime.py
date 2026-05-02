"""prts.runtime —— Agent ↔ SDK 桥接。

Agent 启动时实例化一个实现了 ``RuntimeBridge`` 协议的对象,然后
``set_runtime(bridge)``。SDK 的所有"对外动作"(notify / invoke_tool /
chat / workspace 读写 / history 查询)最终都从这里走。

脚本若不在 PRTS Agent 进程内执行,``get_runtime()`` 会抛
``RuntimeError`` —— 这样调试用户脚本时能立刻定位问题。
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class RuntimeBridge(Protocol):
    """Agent 暴露给 prts.* SDK 的能力集合。"""

    async def notify(self, message: str, *, kind: str = "info", payload: dict[str, Any] | None = None) -> None:
        """主动推送一条消息到当前会话(SSE/WS notify 帧)。"""
        ...

    async def invoke_skill(self, name: str, arguments: dict[str, Any]) -> Any:
        """同进程调用一个 @skill 注册的函数。"""
        ...

    async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """调用任意工具(skill 或外部 MCP 工具,P4 起统一)。"""
        ...

    async def chat(self, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        """直接调底层 LLM,不进 Agent loop;返回完整文本。"""
        ...

    async def read_workspace(self, path: str) -> str:
        ...

    async def write_workspace(self, path: str, content: str) -> None:
        ...

    async def list_workspace(self, prefix: str = "") -> list[str]:
        ...

    async def history(self, session_id: str | None, limit: int) -> list[dict[str, Any]]:
        ...

    async def remember(self, text: str, payload: dict[str, Any] | None = None) -> None:
        """把文本嵌入向量并写入向量存储。"""
        ...

    async def search_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """向量检索,返回 [{id, distance, payload}] 列表。"""
        ...


_runtime: RuntimeBridge | None = None


def set_runtime(bridge: RuntimeBridge | None) -> None:
    """Agent 启动 / 重新加载 skill 时调用。传 ``None`` 卸载。"""
    global _runtime
    _runtime = bridge


def get_runtime() -> RuntimeBridge:
    if _runtime is None:
        raise RuntimeError(
            "prts.* 调用需要在 PRTS Agent 进程内运行,当前没有 runtime 注入。"
            "如果你在直接 `python skill.py`,把它移到 workspace/skills/ 目录,"
            "并通过 PRTS Agent 启动。"
        )
    return _runtime


def has_runtime() -> bool:
    return _runtime is not None
