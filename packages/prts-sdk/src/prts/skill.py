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
import types
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

_NoneType = type(None)


def _annotation_to_schema(ann: Any) -> dict[str, Any]:
    """把 Python 类型注解转成简化版 JSON Schema 片段。

    支持:基本类型、Optional / X | None(PEP 604)、Union[A,B]、list[T]、
    dict[str, T]、Literal[...]。复杂结构降级为 ``{}``(LLM 可以自由填),
    不至于让 skill 加载失败。
    """
    if ann is inspect.Parameter.empty or ann is Any:
        return {}

    # 字面 None / NoneType
    if ann is None or ann is _NoneType:
        return {"type": "null"}

    if ann in _PRIMITIVE_MAP:
        return dict(_PRIMITIVE_MAP[ann])

    origin = typing.get_origin(ann)
    args = typing.get_args(ann)

    # X | None / Optional[X] / Union[A, B, ...]
    # 注意:PEP 604 ``X | None`` 的 origin 是 ``types.UnionType``,
    # ``Union[X, None]`` / ``Optional[X]`` 的 origin 才是 ``typing.Union``。
    if origin is typing.Union or origin is types.UnionType:
        non_none = [a for a in args if a is not _NoneType]
        if not non_none:
            return {"type": "null"}
        if len(non_none) == 1:
            return _annotation_to_schema(non_none[0])
        return {"anyOf": [_annotation_to_schema(a) for a in non_none]}

    if origin in (list, tuple, set, frozenset, typing.List):  # noqa: UP006
        inner = _annotation_to_schema(args[0]) if args else {}
        return {"type": "array", "items": inner}

    if origin in (dict, typing.Dict):  # noqa: UP006
        if len(args) == 2:
            return {"type": "object", "additionalProperties": _annotation_to_schema(args[1])}
        return {"type": "object"}

    # Literal["a", "b"] → {"type": "string", "enum": [...]}(同质类型才标 type)
    if origin is typing.Literal:
        values = list(args)
        kinds = {type(v) for v in values}
        if len(kinds) == 1:
            t = next(iter(kinds))
            base = dict(_PRIMITIVE_MAP.get(t, {}))
            base["enum"] = values
            return base
        return {"enum": values}

    # 兜底:不识别就给空 schema,LLM 仍能传任意值。
    logger.warning(
        "annotation %r not mapped to JSON Schema, falling back to {} (LLM 可填任意值)",
        ann,
    )
    return {}


def _build_input_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """从函数签名生成 JSON Schema(``object`` 顶层 + properties + required)。"""
    sig = inspect.signature(func)
    try:
        hints = typing.get_type_hints(func)
    except Exception as exc:  # noqa: BLE001
        # ``from __future__ import annotations`` 下,注解是字符串,
        # get_type_hints 无法解析时 schema 会全空 —— 给个明确告警避免 silent fail。
        logger.warning(
            "get_type_hints(%s) 失败,schema 将退化为字符串注解形态: %s. "
            "通常是注解里引用了未导入的类型,或 from __future__ import annotations 下"
            "用了 prts 加载阶段不可见的符号。",
            getattr(func, "__qualname__", func),
            exc,
        )
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for pname, param in sig.parameters.items():
        if pname in ("self", "cls"):
            continue
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        ann = hints.get(pname, param.annotation)
        # 注解仍是字符串(get_type_hints 失败但 sig 取到了 raw string)— 退化为 {}。
        if isinstance(ann, str):
            logger.warning(
                "param %s.%s 注解仍是字符串 %r,无法解析 schema,退化为 {}",
                getattr(func, "__qualname__", func),
                pname,
                ann,
            )
            properties[pname] = {}
        else:
            properties[pname] = _annotation_to_schema(ann)
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    # 不加 additionalProperties=False:OpenAI strict mode 会要求 required 覆盖所有字段,
    # 而 PRTS 允许默认参数(如 ping(message="...")),会和 strict 冲突。
    # 当前 tool_choice="auto" 不开 strict,LLM 多传字段会被忽略;后期 P4 接 strict
    # 工具时再针对性补 additionalProperties。
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


# ---------- 加载期事务支持 ---------- #
# Agent 的 skill loader 在 import 单个 .py 失败时,需要回滚那个文件已经注册的
# 任意 @skill / @task 副作用 —— 否则会留下半个文件的工具暴露给 LLM。


def _snapshot() -> tuple[list[SkillRegistration], list[TaskRegistration]]:
    return list(_skills), list(_tasks)


def _restore(snap: tuple[list[SkillRegistration], list[TaskRegistration]]) -> None:
    _skills.clear()
    _skills.extend(snap[0])
    _tasks.clear()
    _tasks.extend(snap[1])
