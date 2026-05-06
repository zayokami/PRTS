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
from .tool_result import ToolResult
from .validator import ToolInputValidationError, validate_tool_input

__all__ = [
    "HookStats",
    "HookedToolRegistry",
    "HookResult",
    "PostFailureHook",
    "PostToolHook",
    "PreToolHook",
    "ToolDefinition",
    "ToolHooks",
    "ToolInputValidationError",
    "ToolPermissionDenied",
    "ToolPermissionEngine",
    "ToolRegistry",
    "ToolResult",
    "ToolTimeoutError",
    "build_default_permission_engine",
    "make_skill_invoker",
    "validate_tool_input",
]
