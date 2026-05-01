"""prts.workspace —— 读写 ``~/.prts/workspace``。

封装 path 安全:不允许 ``..`` 越界、不允许绝对路径,所有路径都相对 workspace。
"""

from __future__ import annotations

from . import runtime as _runtime


async def read(path: str) -> str:
    """读取 workspace 下的相对路径(UTF-8)。"""
    return await _runtime.get_runtime().read_workspace(path)


async def write(path: str, content: str) -> None:
    """覆盖写 workspace 下的相对路径(UTF-8)。"""
    await _runtime.get_runtime().write_workspace(path, content)


async def list_files(prefix: str = "") -> list[str]:
    """列出 workspace 下匹配前缀的文件相对路径。"""
    return await _runtime.get_runtime().list_workspace(prefix)
