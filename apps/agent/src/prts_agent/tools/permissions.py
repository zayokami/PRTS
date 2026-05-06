"""基于规则的工具权限系统。

支持工具名白名单/黑名单、参数模式匹配、危险操作确认。
支持从配置文件加载规则。
"""

import fnmatch
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .hooks import HookResult, PreToolHook, ToolInvocation

logger = logging.getLogger(__name__)


class ToolPermissionEngine:
    """规则引擎：按优先级匹配规则，第一个匹配的规则决定结果。"""

    def __init__(self) -> None:
        self._rules: list[_Rule] = []

    def allow(self, pattern: str, reason: str = "") -> None:
        """允许匹配 pattern 的工具。"""
        self._rules.append(_Rule("allow", pattern, reason))

    def deny(self, pattern: str, reason: str) -> None:
        """拒绝匹配 pattern 的工具。"""
        self._rules.insert(0, _Rule("deny", pattern, reason))

    def require_confirm(self, pattern: str, reason: str) -> None:
        """需要用户确认。对于 PRTS，可转为日志警告。"""
        self._rules.append(_Rule("confirm", pattern, reason))

    async def check(self, invocation: ToolInvocation) -> HookResult:
        for rule in self._rules:
            if fnmatch.fnmatch(invocation.name, rule.pattern):
                if rule.action == "allow":
                    return HookResult(decision="allow")
                elif rule.action == "deny":
                    return HookResult(
                        decision="deny",
                        reason=f"{rule.reason} (matched pattern: {rule.pattern})",
                    )
                elif rule.action == "confirm":
                    logger.warning(
                        "potentially dangerous tool %s invoked: %s",
                        invocation.name,
                        rule.reason,
                    )
                    return HookResult(decision="allow")
        return HookResult(decision="allow")

    def to_dict(self) -> list[dict[str, str]]:
        """序列化规则为字典列表。"""
        return [
            {"action": r.action, "pattern": r.pattern, "reason": r.reason}
            for r in self._rules
        ]

    @classmethod
    def from_dict(cls, rules: list[dict[str, str]]) -> "ToolPermissionEngine":
        """从字典列表加载规则。"""
        engine = cls()
        for r in rules:
            action = r["action"]
            pattern = r["pattern"]
            reason = r.get("reason", "")
            if action == "allow":
                engine.allow(pattern, reason)
            elif action == "deny":
                engine.deny(pattern, reason)
            elif action == "confirm":
                engine.require_confirm(pattern, reason)
            else:
                logger.warning("unknown permission action: %s", action)
        return engine

    def save(self, path: Path) -> None:
        """保存规则到 JSON 文件。"""
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "ToolPermissionEngine":
        """从 JSON 文件加载规则。"""
        if not path.exists():
            logger.info("permission config not found at %s, using defaults", path)
            return build_default_permission_engine()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class _Rule:
    action: str  # "allow" | "deny" | "confirm"
    pattern: str  # glob pattern
    reason: str


def build_default_permission_engine() -> ToolPermissionEngine:
    """默认安全策略。"""
    engine = ToolPermissionEngine()

    # 禁止直接操作系统命令
    engine.deny("bash__exec", "arbitrary code execution not allowed")
    engine.deny("shell__run", "arbitrary code execution not allowed")

    # 文件系统写操作需要确认
    engine.require_confirm(
        "filesystem__write*",
        "filesystem write operation"
    )
    engine.require_confirm(
        "filesystem__delete*",
        "filesystem delete operation"
    )

    # 默认允许读操作
    engine.allow("filesystem__read*", "read-only operations are safe")
    engine.allow("*", "default allow")

    return engine
