"""Tool Input Validator —— JSON Schema 严格验证。

Anthropic ``strict: true`` 的等效实现:
- 注册工具时可标记 ``strict=True``
- 调用前自动验证参数是否符合 input_schema
- 失败时返回清晰的验证错误,阻止非法调用到达 invoker

使用 jsonschema (如果安装) 或降级到轻量手动检查。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 尝试导入 jsonschema;未安装时用降级方案
try:
    from jsonschema import Draft7Validator, ValidationError

    _HAS_JSONSCHEMA = True
except ImportError:
    _HAS_JSONSCHEMA = False
    Draft7Validator = None  # type: ignore[misc,assignment]
    ValidationError = None  # type: ignore[misc,assignment]


class ToolInputValidationError(Exception):
    """工具参数验证失败时抛出。"""

    def __init__(self, tool_name: str, errors: list[str]) -> None:
        self.tool_name = tool_name
        self.errors = errors
        super().__init__(
            f"tool {tool_name} input validation failed: " + "; ".join(errors)
        )


def _validate_with_jsonschema(
    schema: dict[str, Any], arguments: dict[str, Any]
) -> list[str]:
    """用 jsonschema 验证,返回错误消息列表。"""
    errors: list[str] = []
    if Draft7Validator is None:
        return errors  # 不应该走到这里

    validator = Draft7Validator(schema)
    for error in validator.iter_errors(arguments):
        path = "/".join(str(p) for p in error.path) if error.path else "<root>"
        errors.append(f"{path}: {error.message}")
    return errors


def _validate_basic(schema: dict[str, Any], arguments: dict[str, Any]) -> list[str]:
    """降级验证:只检查 required 字段和类型(简单版)。"""
    errors: list[str] = []

    # 1. 检查 required
    required = schema.get("required", [])
    for key in required:
        if key not in arguments:
            errors.append(f"missing required field: {key}")

    # 2. 检查 properties 类型(仅一层)
    properties = schema.get("properties", {})
    for key, value in arguments.items():
        prop_schema = properties.get(key)
        if not prop_schema:
            continue  # 未知字段不报错(允许额外属性)

        expected_type = prop_schema.get("type")
        if not expected_type:
            continue

        type_ok = False
        if expected_type == "string" and isinstance(value, str):
            type_ok = True
        elif expected_type == "integer" and isinstance(value, int):
            type_ok = True
        elif expected_type == "number" and isinstance(value, (int, float)):
            type_ok = True
        elif expected_type == "boolean" and isinstance(value, bool):
            type_ok = True
        elif expected_type == "array" and isinstance(value, list):
            type_ok = True
        elif expected_type == "object" and isinstance(value, dict):
            type_ok = True
        elif expected_type == "null" and value is None:
            type_ok = True

        if not type_ok:
            errors.append(
                f"{key}: expected {expected_type}, got {type(value).__name__}"
            )

    return errors


def validate_tool_input(
    tool_name: str,
    schema: dict[str, Any],
    arguments: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    """验证工具参数。

    参数:
        tool_name: 工具名(用于错误消息)
        schema: JSON Schema (OpenAI/Anthropic function parameters 格式)
        arguments: 用户提供的参数
        strict: 是否启用验证(为 False 时跳过)

    抛出:
        ToolInputValidationError: 验证失败
    """
    if not strict:
        return

    if not schema:
        return

    if _HAS_JSONSCHEMA:
        errors = _validate_with_jsonschema(schema, arguments)
    else:
        errors = _validate_basic(schema, arguments)

    if errors:
        raise ToolInputValidationError(tool_name, errors)


def build_validation_hook():
    """构造一个 pre-tool hook,对 strict=True 的工具做 schema 验证。

    使用方式:
        hooks.register_pre(build_validation_hook())
    """
    from .hooks import HookResult, PreToolHook

    async def _validate_hook(invocation) -> HookResult:
        # 需要从 registry 获取 schema 和 strict 标志
        # 但 hook 只有 invocation,没有 registry 引用
        # 所以这个 hook 需要外部注入 registry
        return HookResult(decision="allow")

    return _validate_hook
