"""Zero-dependency token counting — works with GPT, Claude, Llama, Qwen, DeepSeek, and any other model.

No tiktoken, no sentencepiece, no huggingface tokenizers.  The heuristic is
intentionally *conservative* (over-counts slightly) so we never accidentally
exceed the real context window.  This keeps small local models safe as well as
large cloud APIs.

Heuristic rationale
-------------------
- English text: ~4 chars / token  →  UTF-8 bytes // 3  is slightly conservative
- Chinese text: ~1.5 chars / token →  3 UTF-8 bytes // 3 = 1 token/char  (safe)
- Code / mixed: falls between the two, bytes // 3 still errs on the safe side
- Per-message overhead: +4 tokens for role labels, separators, JSON framing
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Known context limits (tokens).  Prefix matching is used so dated snapshots
# such as "gpt-4o-2024-08-06" still resolve correctly.
#
# Last updated: 2026-05-03 — Agent-verified against April 2026 releases:
#   GPT-5.5, GPT-5.4, GPT-4.1, Claude 4.6, Gemini 3.1, DeepSeek V4,
#   Qwen3.6, Mistral Small 4, Grok 4.x, GLM-5.1, Kimi K2.5, etc.
# ---------------------------------------------------------------------------
_CONTEXT_LIMITS: dict[str, int] = {
    # ------------------------------------------------------------------
    # OpenAI — 2026 旗舰系列
    # ------------------------------------------------------------------
    # GPT-5.5 系列 (April 2026, 1M context, agentic-first, codename "Spud")
    "gpt-5.5": 1_000_000,
    "gpt-5.5-pro": 1_000_000,
    # GPT-5.4 系列 (March 2026, 1M context, native computer-use)
    "gpt-5.4": 1_000_000,
    "gpt-5.4-pro": 1_000_000,
    "gpt-5.4-thinking": 1_000_000,
    # GPT-5 系列 (August 2025, unified reasoning, 400K input)
    "gpt-5": 400_000,
    "gpt-5-mini": 128_000,
    "gpt-5-nano": 128_000,
    # GPT-4.1 系列 (March 2026, 1M context, coding-focused)
    "gpt-4.1": 1_000_000,
    "gpt-4.1-mini": 1_000_000,
    "gpt-4.1-nano": 1_000_000,
    # GPT-4o / GPT-4 系列
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    # o-系列 reasoning (200K)
    "o1": 200_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "o3-mini": 200_000,

    # ------------------------------------------------------------------
    # Anthropic — Claude 4.6 系列 (Feb-March 2026, 1M context, flat pricing)
    # ------------------------------------------------------------------
    "claude-opus-4-6": 1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-6": 1_000_000,
    # Claude 4.5 系列
    "claude-opus-4-5": 1_000_000,
    "claude-sonnet-4-5": 1_000_000,
    "claude-haiku-4-5": 200_000,
    # Claude 3.x 系列 (legacy)
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "claude-2.1": 200_000,
    "claude-2": 100_000,
    "claude-instant": 100_000,

    # ------------------------------------------------------------------
    # Google — Gemini 3.x / 2.x / 1.x 系列
    # ------------------------------------------------------------------
    # Gemini 3.1 (Feb-April 2026)
    "gemini-3.1-pro": 1_000_000,
    "gemini-3.1-flash": 1_000_000,
    "gemini-3.1-flash-lite": 1_000_000,
    # Gemini 3 (Dec 2025)
    "gemini-3-pro": 1_000_000,
    "gemini-3-flash": 1_000_000,
    # Gemini 2.5 (June 2025)
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.5-flash-lite": 1_048_576,
    # Gemini 2.0 (Dec 2024)
    "gemini-2.0-pro": 2_000_000,
    "gemini-2.0-flash": 1_000_000,
    # Gemini 1.5 (legacy long-context)
    "gemini-1.5-pro": 2_000_000,
    "gemini-1.5-flash": 1_000_000,
    "gemini-1.0-pro": 32_768,
    # Gemma open-weight
    "gemma-4": 128_000,

    # ------------------------------------------------------------------
    # Meta — Llama 4 系列 (April 2025, natively multimodal MoE)
    # ------------------------------------------------------------------
    "llama-4-scout": 10_000_000,   # 10M — industry-leading open-weight
    "llama-4-maverick": 1_000_000,
    "llama-4-behemoth": 1_000_000,
    # Llama 3.x 系列
    "llama3.3": 128_000,
    "llama3.2": 128_000,
    "llama3.1": 128_000,
    "llama3": 8_192,

    # ------------------------------------------------------------------
    # DeepSeek — V4 / V3.x / R1 系列 (2025-2026)
    # ------------------------------------------------------------------
    # DeepSeek V4 (April 2026, shipped 24h after GPT-5.5)
    "deepseek-v4": 1_000_000,
    "deepseek-v4-pro": 256_000,
    "deepseek-v4-flash": 1_000_000,
    # DeepSeek V3.x
    "deepseek-v3.2": 163_840,
    "deepseek-v3.1": 128_000,
    "deepseek-v3": 128_000,
    # API aliases (remapped by DeepSeek over time)
    "deepseek-chat": 128_000,       # currently V3.2
    "deepseek-reasoner": 128_000,   # currently V3.2 reasoning mode
    "deepseek-r1": 128_000,
    "deepseek-coder": 128_000,

    # ------------------------------------------------------------------
    # Qwen (Alibaba) — Qwen3.6 / Qwen3.5 / Qwen3 系列 (2025-2026)
    # ------------------------------------------------------------------
    # Qwen3.6 (April 2026)
    "qwen3.6": 256_000,
    "qwen3.6-plus": 1_000_000,
    "qwen3.6-max": 1_000_000,
    "qwen3.6-35b": 262_144,
    "qwen3.6-27b": 262_144,
    # Qwen3.5 (Feb-March 2026)
    "qwen3.5": 1_000_000,
    "qwen3.5-plus": 1_000_000,
    "qwen3.5-omni": 1_000_000,
    "qwen3.5-122b": 1_000_000,
    # Qwen3 (April 2025)
    "qwen3": 128_000,
    "qwen3-235b": 128_000,
    "qwen3-32b": 128_000,
    "qwen3-14b": 128_000,
    "qwen3-8b": 128_000,
    "qwen3-4b": 32_768,
    "qwen3-1.7b": 32_768,
    "qwen3-0.6b": 32_768,
    # Qwen3 Coder
    "qwen3-coder": 256_000,
    # Qwen API proprietary tiers
    "qwen-max": 32_768,
    "qwen-plus": 1_000_000,
    "qwen-turbo": 1_000_000,
    # Qwen 2.5 系列 (legacy open-weight)
    "qwen2.5-72b": 128_000,
    "qwen2.5-32b": 128_000,
    "qwen2.5-14b": 128_000,
    "qwen2.5-7b": 128_000,
    "qwen2.5-3b": 32_768,
    "qwen2.5-1.5b": 32_768,
    "qwen2.5-0.5b": 32_768,

    # ------------------------------------------------------------------
    # Mistral AI — 2025-2026 系列
    # ------------------------------------------------------------------
    "mistral-large-3": 256_000,
    "mistral-large": 256_000,
    "mistral-small-4": 256_000,
    "mistral-small": 256_000,
    "ministral-3": 128_000,
    "mixtral": 32_000,

    # ------------------------------------------------------------------
    # xAI — Grok 4.x / 3.x 系列
    # ------------------------------------------------------------------
    "grok-4.20": 2_000_000,
    "grok-4.1-fast": 2_000_000,
    "grok-4.1-fast-reasoning": 2_000_000,
    "grok-4": 256_000,
    "grok-3": 131_072,
    "grok-3-mini": 131_072,

    # ------------------------------------------------------------------
    # Zhipu AI — GLM 系列 (2026)
    # ------------------------------------------------------------------
    "glm-5.1": 200_000,
    "glm-5": 200_000,
    "glm-4.7": 200_000,
    "glm-4.6": 200_000,
    "glm-4": 128_000,
    "glm-4-plus": 128_000,

    # ------------------------------------------------------------------
    # Moonshot AI — Kimi K2 系列 (2026)
    # ------------------------------------------------------------------
    "kimi-k2.6": 256_000,
    "kimi-k2.5": 256_000,
    "kimi-k2-thinking": 256_000,
    "kimi-k2": 256_000,

    # ------------------------------------------------------------------
    # 其他中国模型 (2026)
    # ------------------------------------------------------------------
    "mimo-v2-pro": 1_000_000,       # Xiaomi, March 2026
    "minimax-m2.7": 200_000,        # MiniMax, March 2026
    "seed-2.0-pro": 272_000,        # ByteDance/Doubao, Feb 2026
    "step-3.5-flash": 256_000,      # StepFun, Feb 2026

    # ------------------------------------------------------------------
    # Yi / Microsoft Phi / Generic local
    # ------------------------------------------------------------------
    "yi-large": 32_768,
    "yi-medium": 16_384,
    "phi-4": 16_384,
    "phi-3": 128_000,
    "llava": 4_096,
}


def get_context_limit(model_name: str) -> int:
    """Return the context-limit for *model_name*.

    Resolution order:
    1. Exact match in the built-in table.
    2. Longest-prefix match (handles dated snapshots like ``gpt-4o-2024-08-06``).
    3. ``LLM_CONTEXT_LIMIT`` env variable (user override).
    4. Safe default of **32 768** — conservative enough for almost any small
       local model, yet large enough for modern APIs.
    """
    model = model_name.lower().strip()

    # 1. exact
    if model in _CONTEXT_LIMITS:
        return _CONTEXT_LIMITS[model]

    # 2. longest-prefix — try every known prefix, keep the longest match
    best_limit: int | None = None
    best_len = 0
    for prefix, limit in _CONTEXT_LIMITS.items():
        if model.startswith(prefix) and len(prefix) > best_len:
            best_limit = limit
            best_len = len(prefix)
    if best_limit is not None:
        return best_limit

    # 3. env override
    env = os.getenv("LLM_CONTEXT_LIMIT")
    if env:
        try:
            return int(env)
        except ValueError:
            logger.warning("LLM_CONTEXT_LIMIT=%r is not an integer, ignoring", env)

    # 4. safe default — *small-model compatible* by design
    logger.warning(
        "Unknown model %r — falling back to 32 k context limit. "
        "Set LLM_CONTEXT_LIMIT env var to override.",
        model_name,
    )
    return 32_768


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Approximate token count for *text* — conservative, zero-deps.

    Uses ``len(text.encode('utf-8')) // 3`` as the baseline.  This naturally
    adapts to any script (Latin 1 byte/char, CJK 3 bytes/char) and is a safe
    upper-bound for every major tokenizer family.
    """
    if not text:
        return 0
    return max(1, len(text.encode("utf-8")) // 3)


def _text_from_content(content: object) -> str:
    """Flatten a message ``content`` field (str or Anthropic block list) to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype in ("tool_use", "tool_result"):
                    # Tool blocks are small JSON-ish objects — estimate by serialising
                    try:
                        parts.append(json.dumps(block, ensure_ascii=False))
                    except (TypeError, ValueError):
                        parts.append(str(block))
                else:
                    parts.append(str(block))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def count_message_tokens(message: dict) -> int:
    """Tokens for a single chat message, including format overhead."""
    content = _text_from_content(message.get("content", ""))
    return count_tokens(content) + 4  # role + separators + JSON framing


def count_messages_tokens(messages: list[dict]) -> int:
    """Tokens for an entire message array, including array-level overhead."""
    if not messages:
        return 0
    return sum(count_message_tokens(m) for m in messages) + 2  # list brackets
