"""@skill / @task 装饰器。

用户写法::

    from prts import skill, task

    @skill(description="查询城市天气")
    async def get_weather(city: str) -> dict:
        ...

    @task(cron="0 8 * * 1-5")
    async def morning_brief():
        ...

P3:
- ``@skill`` 把函数登记到模块级 registry,并把签名 introspect 成 JSON Schema,
  Agent 加载后通过 ToolRegistry 暴露给 LLM(tool calling)
- ``@task`` 仍只登记;真正的 cron / 文件事件触发要等 P6 的 Rust watcher 接入
"""

from __future__ import annotations

import inspect
import logging
import typing
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SkillRegistration:
    func: Callable[..., Any]
    name: str
    description: str | None
    input_schema: dict[str, Any]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRegistration:
    func: Callable[..., Any]
    name: str
    cron: str | None
    on: str | None
    extra: dict[str, Any] = field(default_factory=dict)


_skills: list[SkillRegistration] = []
_tasks: list[TaskRegistration] = []


# ---------- JSON Schema introspection ---------- #

_PRIMITIVE_MAP: dict[Any, dict[str, Any]] = {
    str: {"type": "string"},
    int: {"type": "integer"},
    float: {"type": "number"},
    bool: {"type": "boolean"},
    type(None): {"type": "null"},
    dict: {"type": "object"},
    list: {"type": "array"},
}


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    """把 Python 类型注解转成简化版 JSON Schema 片段。

    支持:基本类型、Optional / X | None、list[T]、dict[str, T]。
    复杂结构降级为 ``{}``(LLM 可以自由填),不至于让 skill 加载失败。
    """
    if ann is inspect.Parameter.empty or ann is Any:
        return {}

    if ann in _PRIMITIVE_MAP:
        return dict(_PRIMITIVE_MAP[ann])

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    # X | None / Optional[X] / Union[A, B, ...]
    if origin is typing.Union or origin is type(None):
        non_none = [a for a in args if a is not type(None)]
        if not non_none:
            return {"type": "null"}
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}

    if origin in (list, typing.List):  # noqa: UP006
        inner = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": inner}

    if origin in (dict, typing.Dict):  # noqa: UP006
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _annotation_to_schema(args[1])}
        return {"type": "object"}

    # Literal["a", "b"]
    if origin is typing.Literal:
        return {"enum": list(args)}

    # 兜底:不识别就给空 schema,LLM 仍能传任意值。
    logger.debug("annotation %r not mapped, falling back to {}", ann)
    return {}


def _build_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名生成 JSON Schema(``object`` 顶层 + properties + required)。"""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception:  # noqa: BLE001
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(pname, param.annotation)
        properties[pname] = _annotation_to_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return schema


# ---------- 装饰器 ---------- #


def skill(
    *,
    description: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """注册一个对 LLM 可见的工具。

    LLM 看到的 ``name`` 默认是函数名,``description`` 默认是 docstring 第一行。
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        tool_name = name or func.__name__
        doc = (func.__doc__ or "").strip().splitlines()
        tool_desc = description or (doc[0] if doc else None)
        schema = _build_input_schema(func)
        _skills.append(
            SkillRegistration(
                func=func,
                name=tool_name,
                description=tool_desc,
                input_schema=schema,
                extra=extra,
            )
        )
        return func

    return decorator


def task(
    *,
    cron: str | None = None,
    on: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """注册一个定时 / 事件触发任务。MVP 仅登记,真正的触发等 P6。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _tasks.append(
            TaskRegistration(
                func=func,
                name=name or func.__name__,
                cron=cron,
                on=on,
                extra=extra,
            )
        )
        return func

    return decorator


def registered_skills() -> list[SkillRegistration]:
    return list(_skills)


def registered_tasks() -> list[TaskRegistration]:
    return list(_tasks)


def _reset_for_tests() -> None:
    _skills.clear()
    _tasks.clear()
