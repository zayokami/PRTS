"""模型兼容性配置。

借鉴 OpenClaw 的 ModelCatalogCompatConfig 设计:用数据驱动的方式描述
不同 LLM provider/model 的行为差异,避免在代码中硬编码 if-else。

新增模型时只需在 REGISTRY 中加一行配置,不需要改任何客户端代码。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelCompatConfig:
    """单个模型的兼容性标志。

    每个 flag 对应一个具体的 provider 行为差异,
    客户端代码通过 ``client.compat.supports_xxx`` 读取,
    而不是硬编码 ``if "deepseek" in model_name``。
    """

    model_id: str = ""

    thinking_format: str = ""
    """推理内容格式: "" (不支持), "deepseek" (reasoning_content 字段), "openai" (reasoning_effort)"""

    requires_reasoning_in_history: bool = False
    """多轮对话时是否必须把 reasoning_content 传回 API (DeepSeek V4 会 400)"""

    requires_nonempty_assistant_content: bool = False
    """assistant 消息 content 不能为空字符串 (即使只有 tool_calls)"""

    requires_tool_result_name: bool = False
    """tool 消息必须包含 name 字段"""

    requires_assistant_after_tool_result: bool = False
    """tool_result 后必须紧跟 assistant 消息 (部分 provider 要求交替)"""

    supports_stream_options: bool = False
    """支持 stream_options.include_usage 获取 token 统计"""

    supports_reasoning_effort: bool = False
    """支持 reasoning_effort 参数 (low/medium/high)"""

    tool_call_id_field: str = "tool_call_id"
    """tool 消息中 tool call ID 的字段名 (OpenAI: tool_call_id)"""

    extra_request_params: dict[str, Any] = field(default_factory=dict)
    """额外请求参数 (如 stream_options)"""


_REGISTRY: dict[str, ModelCompatConfig] = {}


def _deepseek_v4() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="deepseek-v4-pro",
        thinking_format="deepseek",
        requires_reasoning_in_history=True,
        requires_nonempty_assistant_content=True,
    )


def _deepseek_chat() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="deepseek-chat",
        thinking_format="",
    )


def _deepseek_reasoner() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="deepseek-reasoner",
        thinking_format="deepseek",
        requires_reasoning_in_history=False,
        requires_nonempty_assistant_content=False,
    )


def _gpt4o() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="gpt-4o",
        thinking_format="",
        supports_stream_options=True,
        supports_reasoning_effort=False,
    )


def _gpt4o_mini() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="gpt-4o-mini",
        thinking_format="",
        supports_stream_options=True,
    )


def _gpt5() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="gpt-5",
        thinking_format="openai",
        supports_reasoning_effort=True,
        supports_stream_options=True,
    )


def _claude_sonnet() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="claude-sonnet-4-6",
        thinking_format="anthropic",
        requires_nonempty_assistant_content=True,
    )


def _qwen_plus() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="qwen-plus",
        thinking_format="",
    )


def _ollama_default() -> ModelCompatConfig:
    return ModelCompatConfig(
        model_id="ollama",
        thinking_format="",
    )


def _init_registry() -> None:
    for cfg in [
        _deepseek_v4(),
        _deepseek_chat(),
        _deepseek_reasoner(),
        _gpt4o(),
        _gpt4o_mini(),
        _gpt5(),
        _claude_sonnet(),
        _qwen_plus(),
        _ollama_default(),
    ]:
        _REGISTRY[cfg.model_id.lower()] = cfg


_init_registry()


def _match_config(model: str) -> ModelCompatConfig:
    """查找模型兼容性配置。先精确匹配,再前缀匹配,最后返回默认。"""
    key = model.lower().strip()

    if key in _REGISTRY:
        return _REGISTRY[key]

    for reg_key, cfg in _REGISTRY.items():
        if key.startswith(reg_key) or reg_key.startswith(key):
            return cfg

    if "deepseek-v4" in key or "deepseek-v3" in key:
        return _deepseek_v4()
    if "deepseek-reason" in key:
        return _deepseek_reasoner()
    if "deepseek" in key:
        return _deepseek_chat()
    if "gpt-5" in key or "gpt5" in key:
        return _gpt5()
    if "gpt-4o" in key:
        return _gpt4o()
    if "claude" in key:
        return _claude_sonnet()
    if "qwen" in key:
        return _qwen_plus()
    if "ollama" in key or "llama" in key:
        return _ollama_default()

    logger.debug("no compat config for model %r, using defaults", model)
    return ModelCompatConfig(model_id=model)


def get_compat(model: str) -> ModelCompatConfig:
    """获取模型的兼容性配置。"""
    return _match_config(model)


def register_model(model_id: str, config: ModelCompatConfig) -> None:
    """注册或覆盖一个模型的兼容性配置。"""
    _REGISTRY[model_id.lower()] = config
