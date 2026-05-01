"""P3 smoke test —— 不需要真 LLM key,用 fake LLM 跑通 AgentLoop。

覆盖:
- workspace/skills/*.py 被加载,@skill 注册到 ToolRegistry
- LLM 决定调 ``add`` 工具 → AgentLoop 执行 → tool_result 写入 DB →
  下一轮 LLM 看到结果后用文本回应
- ``client.notify`` 在 skill 内部触发 → notify 事件流出到 SSE
- assistant 消息(含 tool_calls 元数据) + tool 消息 都写进 SQLite

跑法(项目根)::

    python scripts/smoke_p3.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "apps" / "agent" / "src"))
sys.path.insert(0, str(REPO / "packages" / "prts-sdk" / "src"))

# 直接 import workspace/skills/*.py 时不要在 seed 目录里写 .pyc
sys.dont_write_bytecode = True

import prts.runtime as prts_runtime  # noqa: E402

from prts_agent.llm.base import (  # noqa: E402
    ChatMessage,
    EndEvent,
    LlmClient,
    StreamEvent,
    TextEvent,
    ToolCallEvent,
)
from prts_agent.loop import AgentLoop  # noqa: E402
from prts_agent.memory.sqlite import SqliteStore  # noqa: E402
from prts_agent.runtime import AgentRuntimeBridge  # noqa: E402
from prts_agent.skills import load_user_skills  # noqa: E402
from prts_agent.tools import ToolRegistry  # noqa: E402


GREEN = "\x1b[32m"
RED = "\x1b[31m"
RESET = "\x1b[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}OK{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"{RED}FAIL{RESET} {msg}")
    raise SystemExit(1)


class ScriptedLlm(LlmClient):
    """按"调用次数"切换响应:第 N 次调用 stream_chat 走 turns[N]。"""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self._turns = turns
        self._idx = 0
        self.calls: list[list[ChatMessage]] = []
        self.tools_seen: list[list[dict[str, Any]] | None] = []

    def stream_chat(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self.calls.append([dict(m) for m in messages])
        self.tools_seen.append(tools)
        idx = self._idx
        self._idx += 1
        if idx >= len(self._turns):
            raise AssertionError(f"ScriptedLlm 用完了,被调用 {idx + 1} 次但只有 {len(self._turns)} 轮")
        events = self._turns[idx]

        async def gen() -> AsyncIterator[StreamEvent]:
            for evt in events:
                yield evt

        return gen()


async def run() -> None:
    workspace = REPO / "workspace"
    assert (workspace / "skills" / "add.py").exists(), "workspace/skills/add.py 必须存在"

    # 1) 起 SQLite
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    db_path = Path(tmp.name)
    try:
        store = SqliteStore(db_path)
        await store.ensure_schema()
        ok(f"SQLite schema applied: {db_path.name}")

        # 2) ToolRegistry + skill 加载
        tools = ToolRegistry()

        # 给 SDK 注入一个临时 runtime(skills 加载阶段还没真 LLM,我们用占位)
        class StubRuntime:
            async def notify(self, *args: Any, **kwargs: Any) -> None: ...
            async def invoke_skill(self, name: str, arguments: dict[str, Any]) -> Any:
                return await tools.invoke(name, arguments)
            async def invoke_tool(self, name: str, arguments: dict[str, Any]) -> Any:
                return await tools.invoke(name, arguments)
            async def chat(self, *args: Any, **kwargs: Any) -> str:
                return "stub"
            async def read_workspace(self, *args: Any, **kwargs: Any) -> str: return ""
            async def write_workspace(self, *args: Any, **kwargs: Any) -> None: ...
            async def list_workspace(self, *args: Any, **kwargs: Any) -> list[str]: return []
            async def history(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]: return []

        # 真实 bridge 后续会替换,但 load_user_skills 不需要 runtime
        loaded = load_user_skills(workspace, tools)
        if loaded.errors:
            for err in loaded.errors:
                fail(f"skill 加载错误: {err.file} :: {err.message}")
        if "add" not in tools.names():
            fail(f"add 工具未注册;实际有 {tools.names()}")
        if "ping" not in tools.names():
            fail(f"ping 工具未注册;实际有 {tools.names()}")
        ok(f"skill loader: {len(loaded.skills)} skills, {len(loaded.tasks)} tasks, names={tools.names()}")

        # 验 schema introspection
        add_def = tools.get("add")
        assert add_def is not None
        schema = add_def.input_schema
        if schema.get("type") != "object":
            fail(f"add 顶层 schema 不是 object: {schema}")
        props = schema.get("properties", {})
        if "a" not in props or props["a"].get("type") != "integer":
            fail(f"add.a schema 错: {props}")
        if "b" not in props or props["b"].get("type") != "integer":
            fail(f"add.b schema 错: {props}")
        if "a" not in schema.get("required", []) or "b" not in schema.get("required", []):
            fail(f"add required 缺字段: {schema}")
        ok("add 工具 JSON Schema 正确(integer × integer,required 完整)")

        ping_def = tools.get("ping")
        assert ping_def is not None
        ping_schema = ping_def.input_schema
        if "message" in ping_schema.get("required", []):
            fail("ping.message 默认值,不应在 required 里")
        ok("ping 工具默认参数不出现在 required 中")

        # 3) ToolRegistry 直接调用
        result = await tools.invoke("add", {"a": 2, "b": 3})
        if result != 5:
            fail(f"add(2,3) = {result},预期 5")
        ok("ToolRegistry.invoke add(2,3) == 5")

        # 4) 准备脚本化 LLM
        scripted = ScriptedLlm(
            turns=[
                # 第一轮:"我来算一下" + 调 add(2,3)
                [
                    TextEvent(type="text", delta="我帮博士算一下,"),
                    TextEvent(type="text", delta="2 + 3 =\n"),
                    ToolCallEvent(
                        type="tool_call",
                        id="call_add_001",
                        name="add",
                        arguments={"a": 2, "b": 3},
                    ),
                    EndEvent(type="end", stop_reason="tool_use"),
                ],
                # 第二轮:看到结果后回答
                [
                    TextEvent(type="text", delta="结果是 5"),
                    EndEvent(type="end", stop_reason="stop"),
                ],
            ]
        )

        # 5) Runtime bridge + AgentLoop
        bridge = AgentRuntimeBridge(
            workspace_dir=workspace, store=store, tools=tools, llm_client=scripted
        )
        prts_runtime.set_runtime(bridge)
        loop = AgentLoop(store=store, llm=scripted, tools=tools)

        # 6) 跑一轮 converse
        events: list[dict[str, Any]] = []
        async for evt in loop.converse(
            session_id="smoke-p3-add",
            user_content="算一下 2 加 3",
            system_prompt="(test system prompt)",
            channel="test",
        ):
            events.append(evt)

        types_seen = [e["event"] for e in events]
        for required in ["token", "tool_call", "tool_result", "done"]:
            if required not in types_seen:
                fail(f"事件流缺少 {required}: {types_seen}")
        ok(f"converse 事件流: {types_seen}")

        # 校工具调用事件
        tc_evt = next(e for e in events if e["event"] == "tool_call")
        if tc_evt["data"]["name"] != "add" or tc_evt["data"]["arguments"] != {"a": 2, "b": 3}:
            fail(f"tool_call 数据错: {tc_evt}")
        tr_evt = next(e for e in events if e["event"] == "tool_result")
        if tr_evt["data"]["result"] != 5:
            fail(f"tool_result 数据错: {tr_evt}")
        ok("tool_call/tool_result 事件载荷正确")

        # 校 token
        text_joined = "".join(
            e["data"]["text"] for e in events if e["event"] == "token"
        )
        if "结果是 5" not in text_joined:
            fail(f"最终文本不含 '结果是 5': {text_joined!r}")
        ok(f"token 流文本拼接: {text_joined!r}")

        # 校 LLM 拿到的 messages
        if len(scripted.calls) != 2:
            fail(f"LLM 应被调用 2 次,实际 {len(scripted.calls)}")
        # 第二次调用应当包含 assistant(含 tool_calls) + tool 消息
        second = scripted.calls[1]
        roles = [m["role"] for m in second]
        if "tool" not in roles:
            fail(f"第二轮 LLM 没拿到 tool 消息: {roles}")
        assist = next(m for m in second if m["role"] == "assistant")
        if "tool_calls" not in assist:
            fail(f"第二轮 LLM 拿到的 assistant 消息没有 tool_calls: {assist}")
        ok(f"第二轮 LLM 输入 roles={roles},包含 assistant.tool_calls + tool")

        # 校 tools 透传(OpenAI 风格)
        for t in scripted.tools_seen:
            if not t:
                fail("LLM 调用未传 tools")
            names = [item["function"]["name"] for item in t]
            if "add" not in names:
                fail(f"LLM 看到的工具列表里没有 add: {names}")
        ok("LLM 调用透传了 tools(OpenAI 风格,含 add)")

        # 7) 校 SQLite 状态
        rows = await store.history("smoke-p3-add")
        roles_seq = [r.role for r in rows]
        # 期望:user → assistant(tool_calls,content="我帮博士算一下,2 + 3 =\n") → tool(5) → assistant("结果是 5")
        if roles_seq != ["user", "assistant", "tool", "assistant"]:
            fail(f"DB 消息顺序错: {roles_seq}")
        assist1 = rows[1]
        if not assist1.meta.get("tool_calls"):
            fail(f"第一条 assistant 应有 tool_calls meta: {assist1}")
        if assist1.meta["tool_calls"][0]["name"] != "add":
            fail(f"tool_calls 工具名错: {assist1.meta}")
        tool_row = rows[2]
        if tool_row.meta.get("tool_call_id") != "call_add_001":
            fail(f"tool 行 meta.tool_call_id 错: {tool_row.meta}")
        if json.loads(tool_row.content) != 5:
            fail(f"tool 行 content 不是 5: {tool_row.content!r}")
        ok("SQLite 写入符合预期(user → assistant+tool_calls → tool → assistant)")

        # 8) 测 notify 流程:用 ping 工具触发 client.notify
        scripted2 = ScriptedLlm(
            turns=[
                [
                    ToolCallEvent(
                        type="tool_call",
                        id="call_ping_001",
                        name="ping",
                        arguments={"message": "smoke 通"},
                    ),
                    EndEvent(type="end", stop_reason="tool_use"),
                ],
                [
                    TextEvent(type="text", delta="已发出 notify。"),
                    EndEvent(type="end", stop_reason="stop"),
                ],
            ]
        )
        bridge2 = AgentRuntimeBridge(
            workspace_dir=workspace, store=store, tools=tools, llm_client=scripted2
        )
        prts_runtime.set_runtime(bridge2)
        loop2 = AgentLoop(store=store, llm=scripted2, tools=tools)

        events2: list[dict[str, Any]] = []
        async for evt in loop2.converse(
            session_id="smoke-p3-notify",
            user_content="ping 一下",
            system_prompt="",
            channel="test",
        ):
            events2.append(evt)
        types2 = [e["event"] for e in events2]
        if "notify" not in types2:
            fail(f"事件流缺 notify: {types2}")
        notify_evt = next(e for e in events2 if e["event"] == "notify")
        if notify_evt["data"]["message"] != "smoke 通":
            fail(f"notify 内容不对: {notify_evt}")
        ok(f"notify 流程: events={types2},内容='{notify_evt['data']['message']}'")

        # 9) 测 history API(经 bridge)拿到带 meta 的行
        hist = await bridge.history("smoke-p3-add", 100)
        if not any(r.get("role") == "tool" for r in hist):
            fail("bridge.history 缺 tool 行")
        ok(f"bridge.history 含 {len(hist)} 行,role 集合 {[r['role'] for r in hist]}")

        # 10) 测 unknown tool
        scripted3 = ScriptedLlm(
            turns=[
                [
                    ToolCallEvent(
                        type="tool_call",
                        id="call_x",
                        name="not_exist",
                        arguments={},
                    ),
                    EndEvent(type="end", stop_reason="tool_use"),
                ],
                [
                    TextEvent(type="text", delta="工具不存在,放弃。"),
                    EndEvent(type="end", stop_reason="stop"),
                ],
            ]
        )
        loop3 = AgentLoop(store=store, llm=scripted3, tools=tools)
        events3: list[dict[str, Any]] = []
        async for evt in loop3.converse(
            session_id="smoke-p3-bad",
            user_content="x",
            system_prompt="",
        ):
            events3.append(evt)
        bad_tr = next(e for e in events3 if e["event"] == "tool_result")
        if bad_tr["data"].get("error") is None:
            fail(f"未知工具应当返回 error: {bad_tr}")
        ok(f"未知工具被优雅拒绝: error={bad_tr['data']['error']}")

        # 11) 直接测 tools 适配 (anthropic format)
        ant = tools.to_anthropic_tools()
        if not any(t["name"] == "add" and "input_schema" in t for t in ant):
            fail("anthropic_tools 不含 add")
        ok("ToolRegistry.to_anthropic_tools 形态正确")

        print()
        print(f"{GREEN}P3 smoke all passed{RESET}")
    finally:
        prts_runtime.set_runtime(None)
        try:
            os.unlink(db_path)
        except OSError:
            pass


if __name__ == "__main__":
    asyncio.run(run())
