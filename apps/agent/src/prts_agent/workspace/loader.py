"""读取 ~/.prts/workspace 下的 markdown,拼成 system prompt。

- 默认目录: $PRTS_WORKSPACE_DIR 或 ~/.prts/workspace
- 首次缺失时从 monorepo 的 workspace/ 模板 seed
- skills/ 子目录是 .py 脚本,本阶段跳过(P3 才接入)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 决定 system prompt 中各文件的顺序;不在列表里的 .md 按字母序追加。
PREFERRED_ORDER = ("SOUL.md", "USER.md", "AGENTS.md", "TOOLS.md", "HEARTBEAT.md")


def _repo_seed_dir() -> Path:
    """返回 monorepo 内 workspace/ 模板的绝对路径(相对于本文件)。

    本文件位于 apps/agent/src/prts_agent/workspace/loader.py,
    repo root 是 parents[5]。
    """
    return Path(__file__).resolve().parents[5] / "workspace"


def resolve_workspace_dir() -> Path:
    """计算用户工作区目录,必要时从仓库 seed 拷过去。"""
    env = os.getenv("PRTS_WORKSPACE_DIR")
    target = Path(env).expanduser() if env else Path.home() / ".prts" / "workspace"
    target.mkdir(parents=True, exist_ok=True)

    seed = _repo_seed_dir()
    # 把模板里有但目标里缺失的文件补上(用户已修改的不会被覆盖)
    if seed.exists():
        for src in seed.rglob("*"):
            rel = src.relative_to(seed)
            # 跳过 Python 缓存:这些是 import 时附带产生的,不应进 workspace。
            if any(part in ("__pycache__",) or part.endswith(".pyc") for part in rel.parts):
                continue
            dst = target / rel
            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
            elif src.is_file() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(src.read_bytes())
                logger.info("seeded workspace file %s", rel)
    else:
        logger.warning("workspace seed missing at %s", seed)

    return target


def _ordered_markdown_files(workspace: Path) -> list[Path]:
    if not workspace.is_dir():
        return []
    md_files = [p for p in workspace.iterdir() if p.is_file() and p.suffix.lower() == ".md"]
    by_name = {p.name: p for p in md_files}

    head: list[Path] = []
    for name in PREFERRED_ORDER:
        if name in by_name:
            head.append(by_name.pop(name))
    tail = sorted(by_name.values(), key=lambda p: p.name.lower())
    return head + tail


def load_system_prompt(workspace: Path | None = None) -> str:
    """读取 workspace markdown 并拼接为 system prompt 文本。"""
    ws = workspace or resolve_workspace_dir()
    parts: list[str] = []
    for path in _ordered_markdown_files(ws):
        try:
            text = path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning("skip workspace file %s: %s", path, exc)
            continue
        if not text:
            continue
        parts.append(f"<!-- workspace/{path.name} -->\n{text}")
    return "\n\n".join(parts)
