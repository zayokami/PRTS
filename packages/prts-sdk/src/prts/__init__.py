"""PRTS SDK —— 在 workspace/skills/*.py 里使用的脚本 API。

在 PRTS Agent 进程内被 import;runtime 由 Agent 启动时注入。
脚本若在 PRTS 之外运行,prts.* 调用会抛 RuntimeError。

公开 API(P0 占位,P3 实现):
    prts.skill        装饰器,把函数注册成 LLM 可调用的工具
    prts.task         装饰器,注册定时 / 事件触发任务
    prts.context      当前调用上下文
    prts.client       反向控制 Agent
    prts.llm          直接调 LLM
    prts.workspace    读写 Markdown workspace
    prts.memory       会话历史 / 向量检索
"""

from __future__ import annotations

from .skill import skill, task
from . import context, client, llm, workspace, memory

__all__ = [
    "skill",
    "task",
    "context",
    "client",
    "llm",
    "workspace",
    "memory",
]
__version__ = "0.1.0"
