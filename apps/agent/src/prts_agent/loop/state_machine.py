"""Agent State Machine —— 显式状态跟踪。

OpenAI Agents SDK 启发的状态机模式:
- idle → running → awaiting_tool → running → completed/error
- 每个状态有明确的进入/退出时间戳
- 支持暂停/恢复(未来扩展)
- 可观测的状态转换日志

状态图:
    idle
      ↓ (converse 开始)
    running ←──────┐
      ↓ (tool_call)  │ (tool 结果返回)
    awaiting_tool ───┘
      ↓ (无 tool / 完成)
    completed / error
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """Agent 执行状态。"""

    IDLE = auto()           # 空闲,未在执行
    RUNNING = auto()        # LLM 推理中
    AWAITING_TOOL = auto()  # 等待工具执行
    COMPLETED = auto()      # 正常完成
    ERROR = auto()          # 发生错误
    CANCELLED = auto()      # 被取消


@dataclass
class StateTransition:
    """单次状态转换记录。"""

    from_state: AgentState
    to_state: AgentState
    timestamp: float
    reason: str | None = None
    iteration: int = 0


@dataclass
class AgentStateSnapshot:
    """状态快照(供外部查询)。"""

    state: str
    state_since: float  # 进入当前状态的时间戳
    total_duration_ms: float
    iteration: int
    transition_count: int
    transitions: list[dict[str, Any]]


class AgentStateMachine:
    """Agent 状态机。

    使用方式:
        sm = AgentStateMachine()
        sm.transition_to(AgentState.RUNNING, "converse started")
        ...
        sm.transition_to(AgentState.AWAITING_TOOL, "tool_call received")
        ...
        sm.transition_to(AgentState.COMPLETED, "all done")
    """

    def __init__(self) -> None:
        self._state = AgentState.IDLE
        self._state_since = time.monotonic()
        self._start_time = self._state_since
        self._iteration = 0
        self._transitions: list[StateTransition] = []

    @property
    def state(self) -> AgentState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name.lower()

    @property
    def iteration(self) -> int:
        return self._iteration

    @property
    def current_duration_ms(self) -> float:
        return (time.monotonic() - self._state_since) * 1000

    @property
    def total_duration_ms(self) -> float:
        return (time.monotonic() - self._start_time) * 1000

    def transition_to(
        self, new_state: AgentState, reason: str | None = None
    ) -> None:
        """转换到新状态。"""
        if new_state == self._state:
            return

        old_state = self._state
        now = time.monotonic()

        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            timestamp=now,
            reason=reason,
            iteration=self._iteration,
        )
        self._transitions.append(transition)

        logger.debug(
            "agent state: %s → %s (reason=%s, iter=%d, duration=%.1fms)",
            old_state.name,
            new_state.name,
            reason,
            self._iteration,
            (now - self._state_since) * 1000,
        )

        self._state = new_state
        self._state_since = now

        # 特殊处理
        if new_state == AgentState.RUNNING and old_state == AgentState.IDLE:
            # 新一轮开始,重置计时
            self._start_time = now
            self._iteration = 0
        elif new_state == AgentState.RUNNING and old_state == AgentState.AWAITING_TOOL:
            # 工具结果返回,进入下一轮
            self._iteration += 1

    def start(self) -> None:
        """开始新的 converse 会话。"""
        self._start_time = time.monotonic()
        self._iteration = 0
        self._transitions.clear()
        self.transition_to(AgentState.RUNNING, "converse started")

    def complete(self, reason: str = "done") -> None:
        """标记为完成。"""
        self.transition_to(AgentState.COMPLETED, reason)

    def fail(self, reason: str) -> None:
        """标记为失败。"""
        self.transition_to(AgentState.ERROR, reason)

    def cancel(self) -> None:
        """标记为取消。"""
        self.transition_to(AgentState.CANCELLED, "cancelled by user")

    def snapshot(self) -> AgentStateSnapshot:
        """获取当前状态快照。"""
        return AgentStateSnapshot(
            state=self.state_name,
            state_since=self._state_since,
            total_duration_ms=self.total_duration_ms,
            iteration=self._iteration,
            transition_count=len(self._transitions),
            transitions=[
                {
                    "from": t.from_state.name,
                    "to": t.to_state.name,
                    "reason": t.reason,
                    "iteration": t.iteration,
                    "at": t.timestamp,
                }
                for t in self._transitions[-10:]  # 只保留最近 10 次
            ],
        )

    def to_dict(self) -> dict[str, Any]:
        """转成 dict(供 health endpoint 使用)。"""
        snap = self.snapshot()
        return {
            "state": snap.state,
            "state_since": snap.state_since,
            "total_duration_ms": round(snap.total_duration_ms, 1),
            "iteration": snap.iteration,
            "transition_count": snap.transition_count,
        }
