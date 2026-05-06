"""Multi-Agent Orchestration —— 轻量级多 Agent 编排原型。

OpenAI Orchestration 启发的模式:
- AgentRegistry: 注册多个 specialist Agent
- Orchestrator: 根据用户输入选择最合适的 Agent
- Handoff: Agent 之间转移控制权,附带上下文
- 消息传递: Agent 间通信协议

设计原则:
- 轻量级:不引入复杂的分布式架构
- 透明:Agent 选择过程对用户可见
- 可回退:始终有一个 fallback Agent

典型用法:
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent(code_agent)
    orchestrator.register_agent(research_agent)
    orchestrator.register_agent(fallback_agent, is_default=True)

    result = await orchestrator.route(session_id, user_input)
    # result.agent_name 显示是哪个 Agent 处理的
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AgentDefinition:
    """Agent 定义。

    每个 Agent 是一个 specialist,有独立的:
    - 名称和描述(用于路由决策)
    - 系统提示(塑造 Agent 角色)
    - 工具子集(只暴露相关工具)
    - 可选的路由关键词(加速匹配)
    """

    name: str
    description: str
    system_prompt: str
    tool_names: list[str] = field(default_factory=list)
    routing_keywords: list[str] = field(default_factory=list)
    # 可选:自定义路由函数(覆盖默认的关键词匹配)
    custom_router: Callable[[str], Awaitable[bool]] | None = None


@dataclass
class HandoffContext:
    """Agent 间交接时的上下文。

    当 Agent A 决定把任务交给 Agent B 时,
    通过 HandoffContext 传递必要的上下文信息。
    """

    from_agent: str
    to_agent: str
    reason: str
    session_id: str
    summary: str  # 当前任务的摘要
    key_facts: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)


@dataclass
class RoutingResult:
    """路由结果。"""

    agent_name: str
    confidence: float  # 0.0-1.0
    reason: str
    handoff_from: str | None = None  # 如果不是首次分配


class AgentRegistry:
    """Agent 登记表。"""

    def __init__(self) -> None:
        self._agents: dict[str, AgentDefinition] = {}
        self._default_agent: str | None = None

    def register(self, agent: AgentDefinition, *, is_default: bool = False) -> None:
        """注册一个 Agent。"""
        if agent.name in self._agents:
            logger.warning("agent %r already registered, overwriting", agent.name)
        self._agents[agent.name] = agent
        if is_default:
            self._default_agent = agent.name
            logger.info("default agent set to %r", agent.name)

    def get(self, name: str) -> AgentDefinition | None:
        return self._agents.get(name)

    def all(self) -> list[AgentDefinition]:
        return list(self._agents.values())

    @property
    def default_agent(self) -> str | None:
        return self._default_agent

    def unregister(self, name: str) -> bool:
        """注销 Agent。返回是否成功。"""
        if name not in self._agents:
            return False
        del self._agents[name]
        if self._default_agent == name:
            self._default_agent = None
        return True


class AgentOrchestrator:
    """Agent 编排器 —— 根据用户输入选择最合适的 Agent。

    路由策略(按优先级):
    1. 关键词匹配:Agent 的 routing_keywords 与用户输入匹配
    2. LLM 决策:让 LLM 根据 Agent 描述做选择
    3. Fallback:使用 default_agent
    """

    def __init__(self, registry: AgentRegistry | None = None) -> None:
        self._registry = registry or AgentRegistry()
        self._history: dict[str, list[RoutingResult]] = {}  # session_id -> 路由历史

    @property
    def registry(self) -> AgentRegistry:
        return self._registry

    async def route(
        self,
        session_id: str,
        user_input: str,
        *,
        current_agent: str | None = None,
    ) -> RoutingResult:
        """为用户输入选择最合适的 Agent。

        参数:
            session_id: 会话 ID
            user_input: 用户输入
            current_agent: 当前 Agent(如果是 handoff 场景)

        返回:
            RoutingResult,包含选中的 Agent 名称和置信度
        """
        agents = self._registry.all()
        if not agents:
            raise ValueError("no agents registered")

        # 1. 关键词匹配
        best_match = self._keyword_route(user_input, agents)
        if best_match and best_match.confidence >= 0.3:
            logger.debug(
                "keyword routing: %s (confidence=%.2f, reason=%s)",
                best_match.agent_name,
                best_match.confidence,
                best_match.reason,
            )
            self._record_route(session_id, best_match)
            return best_match

        # 2. 描述匹配(简单版)
        best_match = self._description_route(user_input, agents)
        if best_match and best_match.confidence >= 0.3:
            logger.debug(
                "description routing: %s (confidence=%.2f)",
                best_match.agent_name,
                best_match.confidence,
            )
            self._record_route(session_id, best_match)
            return best_match

        # 3. Fallback
        default = self._registry.default_agent
        if default:
            result = RoutingResult(
                agent_name=default,
                confidence=0.3,
                reason="fallback to default agent",
            )
            logger.debug("fallback routing: %s", default)
            self._record_route(session_id, result)
            return result

        # 4. 最后一个 resort:选第一个注册的
        first = agents[0].name
        result = RoutingResult(
            agent_name=first,
            confidence=0.1,
            reason="fallback to first registered agent",
        )
        self._record_route(session_id, result)
        return result

    async def handoff(
        self,
        session_id: str,
        from_agent: str,
        to_agent: str,
        reason: str,
        context: HandoffContext | None = None,
    ) -> RoutingResult:
        """手动触发 Agent 间交接。

        使用场景:
        - 当前 Agent 发现任务超出自己的能力范围
        - 用户明确请求另一个 Agent 的服务
        - 工作流需要多 Agent 协作
        """
        if not self._registry.get(to_agent):
            raise ValueError(f"target agent {to_agent!r} not registered")

        result = RoutingResult(
            agent_name=to_agent,
            confidence=1.0,
            reason=f"handoff from {from_agent}: {reason}",
            handoff_from=from_agent,
        )

        logger.info(
            "agent handoff: %s → %s (session=%s, reason=%s)",
            from_agent,
            to_agent,
            session_id,
            reason,
        )

        self._record_route(session_id, result)
        return result

    def get_route_history(self, session_id: str) -> list[RoutingResult]:
        """获取某会话的路由历史。"""
        return list(self._history.get(session_id, []))

    def _keyword_route(
        self, user_input: str, agents: list[AgentDefinition]
    ) -> RoutingResult | None:
        """基于关键词的路由。"""
        text = user_input.lower()
        best_agent: str | None = None
        best_score = 0

        for agent in agents:
            if not agent.routing_keywords:
                continue
            score = sum(1 for kw in agent.routing_keywords if kw.lower() in text)
            if score > best_score:
                best_score = score
                best_agent = agent.name

        if best_agent and best_score > 0:
            max_possible = max(len(a.routing_keywords) for a in agents if a.routing_keywords)
            confidence = min(best_score / max(max_possible, 1), 1.0)
            return RoutingResult(
                agent_name=best_agent,
                confidence=confidence,
                reason=f"keyword match (score={best_score})",
            )
        return None

    def _description_route(
        self, user_input: str, agents: list[AgentDefinition]
    ) -> RoutingResult | None:
        """基于描述相似度的路由(简化版)。"""
        text = user_input.lower()
        words = set(text.split())

        best_agent: str | None = None
        best_score = 0.0

        for agent in agents:
            desc_words = set(agent.description.lower().split())
            if not desc_words:
                continue
            overlap = len(words & desc_words)
            score = overlap / len(desc_words)
            if score > best_score:
                best_score = score
                best_agent = agent.name

        if best_agent and best_score > 0:
            return RoutingResult(
                agent_name=best_agent,
                confidence=best_score,
                reason="description similarity",
            )
        return None

    def _record_route(self, session_id: str, result: RoutingResult) -> None:
        if session_id not in self._history:
            self._history[session_id] = []
        self._history[session_id].append(result)
        # 防止内存泄漏:只保留最近 50 条
        if len(self._history[session_id]) > 50:
            self._history[session_id] = self._history[session_id][-50:]


# ---- 便捷工厂函数 ----

def create_default_orchestrator() -> AgentOrchestrator:
    """创建默认编排器,含一个通用 fallback Agent。"""
    registry = AgentRegistry()

    registry.register(
        AgentDefinition(
            name="general",
            description="通用助手,处理日常对话、问答、简单任务",
            system_prompt="你是一个有用的 AI 助手。",
            routing_keywords=["帮助", "问题", "怎么", "什么", "为什么", "如何"],
        ),
        is_default=True,
    )

    return AgentOrchestrator(registry)
