"""PRTS Workspace MCP Server —— 暴露 ``workspace/*.md`` 给 MCP client。

启动方式(被 Agent MCPManager spawn):

    PRTS_WORKSPACE_DIR=~/.prts/workspace python mcp-servers/prts-workspace/server.py

提供工具:
- ``list_documents`` — 列出 workspace 内的 markdown 文件
- ``read_document``  — 读取指定 markdown 文件内容
- ``write_document`` — 写入/覆盖 markdown 文件
- ``search_documents`` — 在文件名和内容中搜索
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("prts-workspace")


def _resolve_workspace_dir() -> Path:
    """从环境变量 ``PRTS_WORKSPACE_DIR`` 或默认 ``~/.prts/workspace`` 解析。"""
    raw = os.getenv("PRTS_WORKSPACE_DIR", "~/.prts/workspace")
    if raw.startswith("~"):
        raw = os.path.join(os.path.expanduser("~"), raw[1:])
    return Path(raw).resolve()


WORKSPACE_DIR = _resolve_workspace_dir()


def _safe_path(rel: str) -> Path:
    """阻止 ``..`` 越界和绝对路径访问 workspace 之外。"""
    if not rel:
        raise ValueError("path 不能为空")
    p = Path(rel)
    if p.is_absolute() or rel.startswith(("/", "\\")):
        raise ValueError(f"绝对路径被拒: {rel}")
    target = (WORKSPACE_DIR / rel).resolve()
    try:
        target.relative_to(WORKSPACE_DIR)
    except ValueError as exc:
        raise ValueError(f"path 越界 (相对于 {WORKSPACE_DIR}): {rel}") from exc
    return target


def _list_md_files(prefix: str = "") -> list[Path]:
    """返回 workspace 内所有 ``.md`` 文件的相对路径列表(已排序)。"""
    out: list[Path] = []
    if not WORKSPACE_DIR.is_dir():
        return out
    for p in WORKSPACE_DIR.rglob("*.md"):
        if not p.is_file():
            continue
        rel = p.relative_to(WORKSPACE_DIR).as_posix()
        if rel.startswith(prefix):
            out.append(p)
    return sorted(out, key=lambda x: x.relative_to(WORKSPACE_DIR).as_posix())


@mcp.tool()
def list_documents(prefix: str = "") -> str:
    """列出 workspace 内所有 markdown 文件的相对路径。

    :param prefix: 可选前缀过滤,例如 ``skills/`` 只列 skills 目录下的文档
    :return: JSON 字符串,格式 ``{"ok": true, "files": ["AGENTS.md", "TOOLS.md", ...]}``
    """
    files = _list_md_files(prefix)
    import json

    rels = [f.relative_to(WORKSPACE_DIR).as_posix() for f in files]
    return json.dumps({"ok": True, "files": rels}, ensure_ascii=False)


@mcp.tool()
def read_document(path: str) -> str:
    """读取指定 markdown 文件的内容。

    :param path: 相对于 workspace 的文件路径,例如 ``AGENTS.md``
    :return: JSON 字符串,格式 ``{"ok": true, "content": "..."}`` 或错误信息
    """
    import json

    try:
        target = _safe_path(path)
        if not target.exists():
            return json.dumps({"ok": False, "error": f"文件不存在: {path}"}, ensure_ascii=False)
        content = target.read_text(encoding="utf-8")
        return json.dumps({"ok": True, "content": content}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


@mcp.tool()
def write_document(path: str, content: str) -> str:
    """写入或覆盖指定 markdown 文件。

    :param path: 相对于 workspace 的文件路径,例如 ``notes/idea.md``
    :param content: 文件内容
    :return: JSON 字符串,格式 ``{"ok": true}`` 或错误信息
    """
    import json

    try:
        target = _safe_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return json.dumps({"ok": True}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


@mcp.tool()
def search_documents(query: str) -> str:
    """在 workspace markdown 文件的文件名和内容中搜索。

    搜索不区分大小写。返回匹配的文件列表,每条附带匹配到的前 3 行片段。

    :param query: 搜索关键词
    :return: JSON 字符串,格式 ``{"ok": true, "results": [{"path": "...", "snippets": ["..."]}]}``
    """
    import json

    if not query:
        return json.dumps({"ok": True, "results": []}, ensure_ascii=False)

    q = query.lower()
    results: list[dict] = []

    try:
        for p in _list_md_files():
            rel = p.relative_to(WORKSPACE_DIR).as_posix()
            matched = False
            snippets: list[str] = []

            # 文件名匹配
            if q in rel.lower():
                matched = True

            # 内容匹配
            try:
                text = p.read_text(encoding="utf-8")
                lines = text.splitlines()
                for line in lines:
                    if q in line.lower():
                        matched = True
                        snippets.append(line.strip())
                        if len(snippets) >= 3:
                            break
            except Exception:  # noqa: BLE001
                pass

            if matched:
                results.append({"path": rel, "snippets": snippets})

        return json.dumps({"ok": True, "results": results}, ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False)


def main() -> None:
    """``prts-workspace`` 命令行入口。"""
    sys.dont_write_bytecode = True
    mcp.run()


if __name__ == "__main__":
    main()
