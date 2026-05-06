"""工具登记表 + 适配器。"""

from .hooks import (
    HookStats,
    HookedToolRegistry,
    HookResult,
    PostFailureHook,
    PostToolHook,
    PreToolHook,
    ToolHooks,
    ToolPermissionDenied,
    ToolTimeoutError,
)
from .permissions import ToolPermissionEngine, build_default_permission_engine
from .registry import ToolDefinition, ToolRegistry, make_skill_invoker

__all__ = [
    "HookStats",
    "HookedToolRegistry",
    "HookResult",
    "PostFailureHook",
    "PostToolHook",
    "PreToolHook",
    "ToolDefinition",
    "ToolHooks",
    "ToolPermissionDenied",
    "ToolPermissionEngine",
    "ToolRegistry",
    "ToolTimeoutError",
    "build_default_permission_engine",
    "make_skill_invoker",
]
