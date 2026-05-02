"""P6 smoke test —— @task 装饰器 + /tasks /events/fs /events/cron 路由。

不需要启动真实 Agent HTTP 服务,直接 import 路由 handler 的底层逻辑。
验证:
- @task 注册到 prts.skill registry
- load_user_skills 正确收集 task 列表
- /events/cron 能执行同步 / 异步 task 函数
- /events/fs 热加载能重扫 skill 文件并更新 registry

跑法(项目根)::

    .venv/Scripts/python.exe scripts/smoke_p6.py
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "agent" / "src"))
sys.path.insert(0, str(REPO / "packages" / "prts-sdk" / "src"))
sys.dont_write_bytecode = True

from prts.skill import _reset_for_tests, registered_tasks, task  # noqa: E402
from prts_agent.skills import load_user_skills  # noqa: E402
from prts_agent.tools import ToolRegistry  # noqa: E402

GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> str:
    print(f"{RED}FAIL{RESET} {msg}")
    sys.exit(1)


def assert_eq(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        fail(f"{label}\n  expected={expected!r}\n  actual={actual!r}")


async def test_task_decorator() -> None:
    """@task 装饰器正确注册到 registry。"""
    _reset_for_tests()

    @task(cron="0 8 * * 1-5")
    async def morning_brief() -> str:
        return "早安"

    @task(on="startup")
    def init_check() -> None:
        pass

    tasks = registered_tasks()
    assert_eq(len(tasks), 2, "应注册 2 个 task")
    names = {t.name for t in tasks}
    assert_eq(names, {"morning_brief", "init_check"}, "task 名称集合")

    t1 = next(t for t in tasks if t.name == "morning_brief")
    assert_eq(t1.cron, "0 8 * * 1-5", "morning_brief cron")
    assert_eq(t1.on, None, "morning_brief on")  # type: ignore[comparison-overlap]

    t2 = next(t for t in tasks if t.name == "init_check")
    assert_eq(t2.on, "startup", "init_check on")
    ok("@task 装饰器注册正确")


async def test_load_user_skills_with_tasks() -> None:
    """load_user_skills 能正确加载 skill + task。"""
    with tempfile.TemporaryDirectory() as td:
        skills_dir = Path(td) / "skills"
        skills_dir.mkdir()

        # 写两个文件:一个 skill,一个 task
        (skills_dir / "weather.py").write_text(
            "from prts import skill\n"
            "@skill(description='查天气')\n"
            "async def get_weather(city: str) -> str:\n"
            "    return f'{city} 晴朗'\n",
            encoding="utf-8",
        )
        (skills_dir / "jobs.py").write_text(
            "from prts import task\n"
            "@task(cron='*/5 * * * *')\n"
            "def heartbeat():\n"
            "    return 'pong'\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        loaded = load_user_skills(Path(td), registry)

        assert_eq(len(loaded.skills), 1, "应加载 1 个 skill")
        assert_eq(len(loaded.tasks), 1, "应加载 1 个 task")
        assert_eq(len(loaded.errors), 0, "不应有错误")
        assert_eq(loaded.skills[0].name, "get_weather", "skill 名称")
        assert_eq(loaded.tasks[0].name, "heartbeat", "task 名称")
        assert_eq(loaded.tasks[0].cron, "*/5 * * * *", "task cron")

        # skill 应注册到 ToolRegistry
        assert registry.get("get_weather") is not None, "get_weather 应在 registry"
        ok("load_user_skills 正确收集 skill + task")


async def test_cron_event_handler() -> None:
    """模拟 /events/cron 的执行逻辑。"""
    _reset_for_tests()
    execution_log: list[str] = []

    @task(cron="* * * * *")
    async def async_job() -> str:
        execution_log.append("async")
        return "async_ok"

    @task(cron="* * * * *")
    def sync_job() -> str:
        execution_log.append("sync")
        return "sync_ok"

    # 模拟 CronEventRequest + handle_cron_event 的核心逻辑
    tasks = registered_tasks()

    async def run_task(name: str) -> tuple[bool, object | None, str | None]:
        target = next((t for t in tasks if t.name == name), None)
        if target is None:
            return False, None, f"task {name!r} not found"
        import inspect
        try:
            func = target.func
            if inspect.iscoroutinefunction(func):
                result = await func()
            else:
                result = func()
            return True, result, None
        except Exception as exc:
            return False, None, f"{type(exc).__name__}: {exc}"

    ok1, res1, err1 = await run_task("async_job")
    assert_eq(ok1, True, "async_job 应成功")
    assert_eq(res1, "async_ok", "async_job 返回值")
    assert_eq(err1, None, "async_job 不应报错")  # type: ignore[comparison-overlap]

    ok2, res2, err2 = await run_task("sync_job")
    assert_eq(ok2, True, "sync_job 应成功")
    assert_eq(res2, "sync_ok", "sync_job 返回值")
    assert_eq(err2, None, "sync_job 不应报错")  # type: ignore[comparison-overlap]

    assert_eq(execution_log, ["async", "sync"], "执行顺序")
    ok("cron 执行器能跑通同步/异步 task")


async def test_fs_hot_reload() -> None:
    """模拟 /events/fs 的热加载:改文件后重扫。"""
    with tempfile.TemporaryDirectory() as td:
        skills_dir = Path(td) / "skills"
        skills_dir.mkdir()

        (skills_dir / "v1.py").write_text(
            "from prts import skill\n"
            "@skill()\n"
            "def add(a: int, b: int) -> int:\n"
            "    return a + b\n",
            encoding="utf-8",
        )

        registry = ToolRegistry()
        loaded1 = load_user_skills(Path(td), registry)
        assert_eq(len(loaded1.skills), 1, "首轮应加载 1 个 skill")
        assert registry.get("add") is not None, "add 应在 registry"

        # 模拟文件被修改:删掉 add,换成 multiply
        (skills_dir / "v1.py").write_text(
            "from prts import skill\n"
            "@skill()\n"
            "def multiply(a: int, b: int) -> int:\n"
            "    return a * b\n",
            encoding="utf-8",
        )

        loaded2 = load_user_skills(Path(td), registry)
        assert_eq(len(loaded2.skills), 1, "重载后仍应 1 个 skill")
        assert registry.get("add") is None, "add 应被移除"
        assert registry.get("multiply") is not None, "multiply 应被注册"

        # 验证 invoke 指向新函数
        result = await registry.invoke("multiply", {"a": 3, "b": 4})
        assert_eq(result, 12, "multiply(3,4)")
        ok("fs 热加载能正确替换 skill")


async def main() -> None:
    await test_task_decorator()
    await test_load_user_skills_with_tasks()
    await test_cron_event_handler()
    await test_fs_hot_reload()
    print(f"\n{GREEN}P6 smoke all passed{RESET}")


if __name__ == "__main__":
    asyncio.run(main())
