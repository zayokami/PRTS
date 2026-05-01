# workspace/skills

> ★ PRTS 的脚本 / 插件目录。在这里放 `.py` 文件,Agent 启动时自动加载。

## 写一个最简单的 skill

```python
# my_skill.py
from prts import skill

@skill(description="把两个数加起来")
async def add(a: int, b: int) -> int:
    return a + b
```

PRTS 启动时会:

1. 扫描本目录所有 `*.py`(`_` 开头的文件忽略)
2. import 每个文件 —— `@skill` 装饰器把函数登记到 registry
3. 经 FastMCP 暴露给 LLM,LLM 看到工具签名后可以自主调用

## 写一个定时任务

```python
# morning_brief.py
from prts import task, client

@task(cron="0 8 * * 1-5")  # 工作日早 8 点
async def morning_brief():
    await client.notify("早安博士,今天又是元气满满的一天。")
```

## 反向调用 Agent

```python
from prts import skill, client

@skill(description="把一段文本写进笔记")
async def note(content: str) -> dict:
    await client.notify(f"已记下:{content[:50]}…")
    return {"ok": True}
```

## 安全提醒

`workspace/skills/*.py` **等价于本机执行权限**,在 Agent 进程内 import,无沙箱。
不要跑陌生人的 .py。
