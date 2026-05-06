"""Prompt Injection Classifier —— 轻量级提示注入检测。

Anthropic Computer Use 启发的安全防护层:
- 基于正则表达式 + 关键词的轻量级分类
- 在 tool hook 之前自动扫描用户输入和工具参数
- 检测到可疑内容时返回结构化结果,供上层决策
- 可配置敏感度级别(low/medium/high)

检测模式:
1. 指令覆盖(ignore previous/future instructions)
2. 角色覆盖(you are now/new role)
3. 越狱关键词(DAN, Do Anything Now)
4. 系统提示注入(system prompt/instruction override)
5. XML 标签注入(<system>, <instructions>)
6. 上下文重置(forget everything/reset)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ---- 检测模式 ----

# 高风险:直接指令覆盖
_HIGH_RISK_PATTERNS: list[tuple[str, int]] = [
    # 指令覆盖
    (r"ignore\s+(all\s+)?(previous|above|prior|earlier)\s+(instructions?|commands?|prompts?|directions?)", 100),
    (r"ignore\s+(all\s+)?(following|below|subsequent)\s+(instructions?|commands?|prompts?|directions?)", 100),
    (r"disregard\s+(all\s+)?(previous|prior)\s+(instructions?|commands?|prompts?)", 100),
    (r"override\s+(all\s+)?(previous|prior)\s+(instructions?|commands?|prompts?)", 100),
    (r"forget\s+(all\s+)?(previous|prior)\s+(instructions?|commands?|prompts?)", 100),
    # 系统提示
    (r"system\s+(prompt|instruction|message)\s*[:=]", 90),
    (r"your\s+(system|core|base)\s+(prompt|instruction)", 90),
    # 角色覆盖
    (r"you\s+are\s+now\s+(a\s+)?", 80),
    (r"your\s+new\s+(role|identity|purpose)\s+is", 80),
    (r"from\s+now\s+on\s+you\s+are", 80),
    # DAN / 越狱
    (r"\bDAN\b.*Do\s+Anything\s+Now", 100),
    (r"jailbreak\b", 80),
    (r"\bmode:\s*(developer|god|admin|root)\b", 70),
]

# 中风险:可疑但可能是合法内容
_MEDIUM_RISK_PATTERNS: list[tuple[str, int]] = [
    (r"<\s*(system|instructions?|prompt)\s*>", 60),
    (r"</\s*(system|instructions?|prompt)\s*>", 60),
    (r"new\s+(instructions?|commands?|rules?)", 50),
    (r"act\s+as\s+(a\s+)?", 40),
    (r"pretend\s+to\s+be\s+(a\s+)?", 40),
    (r"simulate\s+(being\s+)?(a\s+)?", 40),
    (r"hypothetically", 30),
    (r"for\s+educational\s+purposes", 30),
    (r"\b(dev|developer|debug)\s+mode\b", 50),
]

# 低风险:上下文操作
_LOW_RISK_PATTERNS: list[tuple[str, int]] = [
    (r"reset\s+(context|memory|history)", 40),
    (r"clear\s+(context|memory|history)", 40),
    (r"start\s+over\b", 30),
    (r"let's\s+begin\s+anew", 30),
]

_ALL_PATTERNS = _HIGH_RISK_PATTERNS + _MEDIUM_RISK_PATTERNS + _LOW_RISK_PATTERNS


# ---- 数据模型 ----

@dataclass
class InjectionMatch:
    """单次匹配结果。"""

    pattern: str  # 匹配到的正则表达式
    score: int    # 风险分数
    text: str     # 匹配的文本片段
    start: int    # 起始位置
    end: int      # 结束位置


@dataclass
class InjectionScanResult:
    """扫描结果。"""

    is_injection: bool
    confidence: float  # 0.0-1.0
    score: int         # 总分
    threshold: int     # 当前阈值
    matches: list[InjectionMatch] = field(default_factory=list)
    scanned_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_injection": self.is_injection,
            "confidence": round(self.confidence, 3),
            "score": self.score,
            "threshold": self.threshold,
            "matches": [
                {
                    "pattern": m.pattern,
                    "score": m.score,
                    "text": m.text[:100],  # 截断
                }
                for m in self.matches
            ],
        }


# ---- 分类器 ----

class PromptInjectionClassifier:
    """轻量级提示注入分类器。

    用法:
        classifier = PromptInjectionClassifier(sensitivity="medium")
        result = classifier.scan_text(user_input)
        if result.is_injection:
            # 拒绝或要求确认
            ...
    """

    _THRESHOLDS = {
        "low": 150,      # 只拦截最明显的
        "medium": 80,    # 平衡(默认)
        "high": 40,      # 严格,可能误报
    }

    def __init__(self, sensitivity: str = "medium") -> None:
        self._threshold = self._THRESHOLDS.get(sensitivity, 80)
        self._patterns = [
            (re.compile(p, re.IGNORECASE), score)
            for p, score in _ALL_PATTERNS
        ]

    @property
    def threshold(self) -> int:
        return self._threshold

    def scan_text(self, text: str | None) -> InjectionScanResult:
        """扫描单段文本。"""
        if not text:
            return InjectionScanResult(
                is_injection=False, confidence=0.0, score=0, threshold=self._threshold
            )

        matches: list[InjectionMatch] = []
        total_score = 0

        for pattern, score in self._patterns:
            for m in pattern.finditer(text):
                match_obj = InjectionMatch(
                    pattern=pattern.pattern,
                    score=score,
                    text=m.group(0),
                    start=m.start(),
                    end=m.end(),
                )
                matches.append(match_obj)
                total_score += score

        # 去重:同一位置的多个模式只取最高分
        matches = self._deduplicate(matches)
        total_score = sum(m.score for m in matches)

        is_injection = total_score >= self._threshold
        max_possible = sum(s for _, s in _ALL_PATTERNS)
        confidence = min(total_score / max_possible * 3, 1.0) if max_possible > 0 else 0.0

        return InjectionScanResult(
            is_injection=is_injection,
            confidence=confidence,
            score=total_score,
            threshold=self._threshold,
            matches=matches,
            scanned_text=text[:500],  # 保留前 500 字符用于日志
        )

    def scan_arguments(self, arguments: dict[str, Any]) -> InjectionScanResult:
        """扫描工具参数字典(递归检查所有字符串值)。"""
        texts: list[str] = []
        self._collect_strings(arguments, texts)
        combined = "\n".join(texts)
        return self.scan_text(combined)

    def _collect_strings(self, obj: Any, out: list[str]) -> None:
        """递归收集对象中的所有字符串。"""
        if isinstance(obj, str):
            out.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values():
                self._collect_strings(v, out)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_strings(item, out)

    def _deduplicate(self, matches: list[InjectionMatch]) -> list[InjectionMatch]:
        """去重:重叠的匹配只保留最高分的。"""
        if not matches:
            return []

        # 按起始位置排序
        sorted_matches = sorted(matches, key=lambda m: (m.start, -m.score))
        kept: list[InjectionMatch] = []

        for m in sorted_matches:
            # 检查是否与已保留的重叠
            overlap = False
            for k in kept:
                if not (m.end <= k.start or m.start >= k.end):
                    # 重叠,跳过(因为已按分数降序排列,保留的是更高分的)
                    overlap = True
                    break
            if not overlap:
                kept.append(m)

        return kept


# ---- 与 Tool Hooks 集成 ----

def build_injection_guard(
    classifier: PromptInjectionClassifier | None = None,
) -> "PreToolHook":
    """构造一个 pre-tool hook,在工具调用前扫描参数中的提示注入。

    使用方式:
        from .hooks import ToolHooks
        from .prompt_injection import build_injection_guard, PromptInjectionClassifier

        classifier = PromptInjectionClassifier(sensitivity="medium")
        hooks = ToolHooks()
        hooks.register_pre(build_injection_guard(classifier))
    """
    from .hooks import HookResult, PreToolHook

    _classifier = classifier or PromptInjectionClassifier()

    async def _guard(invocation) -> HookResult:
        result = _classifier.scan_arguments(invocation.arguments)
        if result.is_injection:
            logger.warning(
                "prompt injection detected in tool %s (score=%d, confidence=%.2f): %s",
                invocation.name,
                result.score,
                result.confidence,
                result.to_dict(),
            )
            return HookResult(
                decision="deny",
                reason=f"prompt injection detected (score={result.score}): "
                + ", ".join(m.pattern for m in result.matches[:3]),
            )
        return HookResult(decision="allow")

    return _guard
