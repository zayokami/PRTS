"""把 MCP ``CallToolResult`` 翻译成 ToolRegistry 的"raise on error / return on success"约定。

设计:
- 工具名 = ``<server_name>__<raw_tool_name>``,固定带前缀,LLM 看到的就是这个名字。
- 调用前 ``asyncio.wait_for`` 包一层超时,server 卡住时不会无限阻塞 agent loop。
- ``result.isError == True``(server 自报错) → 抛 ``RuntimeError``,内容来自 text 块。
- ``result.structuredContent``(MCP 2025-06-18+) 优先 → 直接返回 dict;
  ``{"result": value}`` 这种 FastMCP 自动壳被脱掉。
- 否则展平 ``content`` 块:
  * 仅 1 个 ``TextContent`` → 返回字符串
  * 多块 / 非文本(image/embedded resource) → 返回 list[dict],由 runner 端 JSON 化
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp import ClientSession
    from mcp.types import ListToolsResult

    from ..tools import ToolRegistry

logger = logging.getLogger(__name__)


def _block_to_dict(block: Any) -> dict[str, Any]:
    """单个 content 块打成 LLM 看得懂的 dict(不依赖 mcp 内部具体类型,鸭子类型即可)。"""
    btype = getattr(block, "type", None) or "unknown"
    if btype == "text":
        return {"type": "text", "text": getattr(block, "text", "")}
    if btype == "image":
        return {
            "type": "image",
            "mimeType": getattr(block, "mimeType", ""),
            "data": getattr(block, "data", ""),
        }
    if btype == "resource":
        # EmbeddedResource:.resource 里通常有 uri / text / blob
        resource = getattr(block, "resource", None)
        out: dict[str, Any] = {"type": "resource"}
        if resource is not None:
            for attr in ("uri", "mimeType", "text"):
                val = getattr(resource, attr, None)
                if val is not None:
                    out[attr] = val
        return out
    # 兜底:把这个块的 model_dump 给 LLM
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json")  # type: ignore[no-any-return]
    return {"type": btype, "repr": repr(block)}


def _flatten_text(blocks: list[Any]) -> str:
    """只挑 TextContent,用 ``\\n\\n`` join。给 isError 兜底用,不参与正常路径。"""
    parts = [
        getattr(b, "text", "")
        for b in blocks
        if getattr(b, "type", None) == "text"
    ]
    return "\n\n".join(p for p in parts if p)


def _flatten_content(blocks: list[Any]) -> Any:
    """正常 success 路径的展平。

    - 0 块 → ``None``
    - 1 个 text 块 → 字符串(LLM 最常见的工具返回形态)
    - 其他 → list[dict],runner 那边 ``json.dumps`` 走兜底分支
    """
    if not blocks:
        return None
    if len(blocks) == 1 and getattr(blocks[0], "type", None) == "text":
        return getattr(blocks[0], "text", "")
    return [_block_to_dict(b) for b in blocks]


def make_mcp_invoker(
    session: "ClientSession",
    raw_name: str,
    timeout_s: float,
) -> Callable[[dict[str, Any]], Awaitable[Any]]:
    """生成一个适配 ToolRegistry 的 invoker。

    Returns
    -------
    Callable
        ``async (arguments: dict) -> Any``。失败抛异常,成功返回展平后的内容。
    """

    async def _invoke(arguments: dict[str, Any]) -> Any:
        try:
            result = await asyncio.wait_for(
                session.call_tool(raw_name, arguments=arguments),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"mcp tool {raw_name!r} timed out after {timeout_s}s"
            ) from exc

        if getattr(result, "isError", False):
            text = _flatten_text(getattr(result, "content", []) or [])
            raise RuntimeError(
                f"mcp tool {raw_name!r} failed: {text or 'no error text'}"
            )

        structured = getattr(result, "structuredContent", None)
        if structured is not None:
            # FastMCP 把"非 dict 返回类型"(比如 ``-> str`` / ``-> int``) 的工具
            # 结果按 spec 包成 ``{"result": value}`` 以满足生成的 outputSchema。
            # LLM 看到这个一层壳没有意义,统一脱掉。
            if (
                isinstance(structured, dict)
                and len(structured) == 1
                and "result" in structured
            ):
                return structured["result"]
            return structured
        return _flatten_content(getattr(result, "content", []) or [])

    return _invoke


def register_server_tools(
    server_name: str,
    session: "ClientSession",
    tools_response: "ListToolsResult",
    registry: "ToolRegistry",
    timeout_s: float,
) -> list[str]:
    """把一个 MCP server 暴露的所有工具批量注册进 ToolRegistry。

    返回注册好的(已加前缀的)名字列表,便于 manager 写状态。
    """
    from ..tools import ToolDefinition

    registered: list[str] = []
    for tool in getattr(tools_response, "tools", []) or []:
        raw_name = tool.name
        prefixed = f"{server_name}__{raw_name}"
        # input_schema 可能是 None / 缺字段,统一兜底成最小合法 schema。
        schema_obj: Any = getattr(tool, "inputSchema", None)
        if schema_obj is None:
            input_schema: dict[str, Any] = {"type": "object", "properties": {}}
        elif hasattr(schema_obj, "model_dump"):
            input_schema = schema_obj.model_dump(mode="json")  # type: ignore[assignment]
        elif isinstance(schema_obj, dict):
            input_schema = schema_obj
        else:
            input_schema = {"type": "object", "properties": {}}

        registry.register(
            ToolDefinition(
                name=prefixed,
                description=getattr(tool, "description", None),
                input_schema=input_schema,
                invoker=make_mcp_invoker(session, raw_name, timeout_s),
                source="mcp",
                extra={"server": server_name, "raw_name": raw_name},
            )
        )
        registered.append(prefixed)

    logger.info(
        "registered %d tool(s) from MCP server %r: %s",
        len(registered),
        server_name,
        registered,
    )
    return registered
