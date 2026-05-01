"""@skill / @task 装饰器(P0 占位,真正实现在 P3 / P6)。

用户写法:

    from prts import skill, task

    @skill(description="查询城市天气")
    async def get_weather(city: str) -> dict:
        ...

    @task(cron="0 8 * * 1-5")
    async def morning_brief():
        ...

P0 阶段:装饰器仅把函数登记到模块级 registry,不做实际注册。
P3 阶段:Agent 加载 .py 后从 registry 取数,经 FastMCP.add_tool() 暴露给 LLM。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class SkillRegistration:
    func: Callable[..., Any]
    description: str | None
    name: str | None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskRegistration:
    func: Callable[..., Any]
    cron: str | None
    on: str | None
    name: str | None
    extra: dict[str, Any] = field(default_factory=dict)


_skills: list[SkillRegistration] = []
_tasks: list[TaskRegistration] = []


def skill(
    *,
    description: str | None = None,
    name: str | None = None,
    **extra: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """注册一个对 LLM 可见的工具。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _skills.append(
            SkillRegistration(func=func, description=description, name=name, extra=extra)
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
    """注册一个定时 / 事件触发任务(由 Rust watcher 驱动)。"""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        _tasks.append(
            TaskRegistration(func=func, cron=cron, on=on, name=name, extra=extra)
        )
        return func

    return decorator


def registered_skills() -> list[SkillRegistration]:
    """Agent 加载完所有 .py 后调用,获取已注册的 skill 列表。"""
    return list(_skills)


def registered_tasks() -> list[TaskRegistration]:
    return list(_tasks)


def _reset_for_tests() -> None:
    _skills.clear()
    _tasks.clear()
