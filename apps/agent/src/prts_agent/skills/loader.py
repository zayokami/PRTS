"""扫描 ``workspace/skills/*.py`` 把 @skill / @task 装载到 Agent。

加载流程:
1. 调 ``prts.skill._reset_for_tests()`` 清掉 registry —— 支持热加载
2. 对 ``workspace/skills/`` 下每个非下划线 ``.py``:
   - 用 importlib 在 ``prts.user_skills.<stem>`` 命名空间下加载
   - 每个文件单独 try/except,失败的文件不影响其他文件
3. ``prts.skill.registered_skills()`` 拿到全部 SkillRegistration
4. 包成 ToolDefinition 注册到 ``ToolRegistry``

下划线开头的文件 / 目录(``_examples/`` 之类)被显式跳过 —— 这是约定,
方便 README 里放范例代码而不被误加载。
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prts.skill import SkillRegistration, TaskRegistration

    from ..tools import ToolRegistry

logger = logging.getLogger(__name__)

USER_PKG_PREFIX = "prts_user_skills"


@dataclass
class LoadError:
    file: str
    message: str
    traceback: str


@dataclass
class LoadedSkills:
    skills: list["SkillRegistration"] = field(default_factory=list)
    tasks: list["TaskRegistration"] = field(default_factory=list)
    errors: list[LoadError] = field(default_factory=list)
    files_scanned: int = 0


def _iter_skill_files(skills_dir: Path) -> list[Path]:
    if not skills_dir.is_dir():
        return []
    out: list[Path] = []
    for p in sorted(skills_dir.rglob("*.py")):
        rel_parts = p.relative_to(skills_dir).parts
        # 跳过任意以 _ 开头的目录 / 文件(包括 _examples/),以及 __pycache__。
        if any(
            part.startswith("_") or part == "__pycache__" for part in rel_parts
        ):
            continue
        out.append(p)
    return out


def _purge_user_modules() -> None:
    """清理上一次加载的 user skill 模块,以便重新 import。"""
    for mod_name in list(sys.modules.keys()):
        if mod_name == USER_PKG_PREFIX or mod_name.startswith(USER_PKG_PREFIX + "."):
            del sys.modules[mod_name]


def load_user_skills(workspace_dir: Path, registry: "ToolRegistry") -> LoadedSkills:
    """扫描并加载用户脚本,把 @skill 注册进 ``registry``。"""
    # 局部 import 避免静态环依赖(skill 模块同时被 sdk 和 agent 引用)
    from prts.skill import _reset_for_tests, registered_skills, registered_tasks

    from ..tools import ToolDefinition, make_skill_invoker

    _reset_for_tests()
    _purge_user_modules()
    registry.clear()

    skills_dir = workspace_dir / "skills"
    files = _iter_skill_files(skills_dir)
    result = LoadedSkills(files_scanned=len(files))

    for path in files:
        rel = path.relative_to(skills_dir).with_suffix("")
        mod_name = USER_PKG_PREFIX + "." + ".".join(rel.parts)
        try:
            spec = importlib.util.spec_from_file_location(mod_name, path)
            if spec is None or spec.loader is None:
                raise ImportError(f"无法构造 spec: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001
            sys.modules.pop(mod_name, None)
            tb = traceback.format_exc()
            logger.warning("skill 加载失败 %s: %s", path, exc)
            result.errors.append(
                LoadError(file=str(path), message=str(exc), traceback=tb)
            )

    # 收集 registry,转成 ToolDefinition
    for reg in registered_skills():
        registry.register(
            ToolDefinition(
                name=reg.name,
                description=reg.description,
                input_schema=reg.input_schema,
                invoker=make_skill_invoker(reg.func),
                source="skill",
                extra=reg.extra,
            )
        )
        result.skills.append(reg)

    for tsk in registered_tasks():
        result.tasks.append(tsk)

    logger.info(
        "loaded skills: %d skill(s), %d task(s), %d error(s) from %d file(s)",
        len(result.skills),
        len(result.tasks),
        len(result.errors),
        result.files_scanned,
    )
    return result
