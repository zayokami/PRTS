"""Tool Result —— 结构化工具调用结果,支持错误标记。

Anthropic Tool Result 启发的结构化格式:
- content: 正常结果(字符串/字典/列表)
- is_error: 是否出错
- error_type: 错误类型(如 ToolPermissionDenied)
- error_message: 错误详情

优势:
1. LLM 看到 is_error=true 时知道结果不可用,不会误用
2. 错误类型可用于分类统计
3. 支持部分成功(有内容但也有警告)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """结构化工具调用结果。

    正常结果:
        ToolResult(content="success")
        ToolResult(content={"data": [1, 2, 3]})

    错误结果:
        ToolResult(
            is_error=True,
            error_type="ToolPermissionDenied",
            error_message="tool bash__exec denied",
        )

    部分成功:
        ToolResult(
            content={"partial": True},
            is_error=True,
            error_type="ToolTimeoutError",
            error_message="timed out after 60s",
        )
    """

    content: Any = None
    is_error: bool = False
    error_type: str | None = None
    error_message: str | None = None

    def to_llm_text(self) -> str:
        """转成 LLM tool 消息的 content 文本。

        错误时包含明显的 ERROR 前缀和类型信息,让 LLM 不会误用。
        """
        if self.is_error:
            parts: list[str] = []
            if self.error_type:
                parts.append(f"[ERROR: {self.error_type}]")
            if self.error_message:
                parts.append(self.error_message)
            if self.content is not None:
                # 部分成功时附带可用内容
                content_str = _serialize_any(self.content)
                parts.append(f"[partial result: {content_str}]")
            return "\n".join(parts) if parts else "ERROR: unknown error"

        # 正常结果
        return _serialize_any(self.content)

    def to_sse_dict(self) -> dict[str, Any]:
        """转成 SSE event 的 data dict。"""
        return {
            "content": self.content,
            "is_error": self.is_error,
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    @classmethod
    def from_exception(cls, exc: Exception, *, content: Any = None) -> "ToolResult":
        """从异常构造错误结果。"""
        return cls(
            content=content,
            is_error=True,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    @classmethod
    def success(cls, content: Any) -> "ToolResult":
        """构造成功结果。"""
        return cls(content=content, is_error=False)


def _serialize_any(value: Any) -> str:
    """任意值转字符串。"""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)
