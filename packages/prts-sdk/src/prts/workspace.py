"""prts.workspace —— 读写 Markdown workspace。P0 占位。"""

from __future__ import annotations


async def read(path: str) -> str:
    raise NotImplementedError("P2 阶段实现")


async def write(path: str, content: str) -> None:
    raise NotImplementedError("P2 阶段实现")


async def list_files(prefix: str = "") -> list[str]:
    raise NotImplementedError("P2 阶段实现")
