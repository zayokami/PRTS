"""PRTS SDK —— 在 ``workspace/skills/*.py`` 里使用的脚本 API。

Agent 启动时 import 用户脚本之前会调 ``prts.runtime.set_runtime(bridge)``,
脚本内的 ``prts.client`` / ``prts.workspace`` / ``prts.memory`` 等调用最终都
经这里转发到 Agent 内部实现。脚本若直接 ``python xxx.py`` 运行,SDK 调用
会抛 ``RuntimeError``,提示需在 PRTS 进程内执行。

公开 API:
    prts.skill        装饰器,把函数注册成 LLM 可调用的工具
    prts.task         装饰器,注册定时 / 事件触发任务(P6 起触发)
    prts.context      当前调用上下文 (session_id, user_id, history)
    prts.client       反向控制 Agent (notify / skill / chat)
    prts.llm          直接调底层 LLM(不进 Agent loop)
    prts.workspace    读写 workspace markdown
    prts.memory       会话历史(P3) / 向量检索(P7)
    prts.runtime      Agent 注入桥接的内部模块,脚本一般不直接用
"""

from __future__ import annotations

from . import client, context, llm, memory, runtime, workspace
from .skill import skill, task

__all__ = [
    "skill",
    "task",
    "context",
    "client",
    "llm",
    "workspace",
    "memory",
    "runtime",
]
__version__ = "0.1.0"
