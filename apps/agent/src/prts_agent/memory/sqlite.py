"""SQLite 持久化:会话 / 消息 / 工具调用 / 事件。

P3 起 messages 表新增 ``meta`` 列(JSON 文本),用来装:
- 助手发出工具调用时:``{"tool_calls": [{id,name,arguments}, ...]}``
- 工具结果消息:``{"tool_call_id":..., "tool_name":..., "is_error": bool}``

老库会被自动迁移(``ALTER TABLE``);幂等。

并发:开 WAL + busy_timeout=5s,允许多个进程同时读、单进程写时不互相 lock。
单 Agent worker 场景 99% 是单写多读,WAL 已经够。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import aiosqlite

logger = logging.getLogger(__name__)

Role = Literal["system", "user", "assistant", "tool"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    channel     TEXT NOT NULL,
    user_ref    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    meta        TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id  INTEGER NOT NULL REFERENCES messages(id),
    tool_name   TEXT NOT NULL,
    arguments   TEXT NOT NULL,
    result      TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
"""

# 连接级 PRAGMA。WAL 让读写能并行,busy_timeout 让另一个进程在写锁时等而非
# 立刻报 SQLITE_BUSY;synchronous=NORMAL 是 WAL 推荐值,牺牲一点 fsync 严格性
# 换吞吐(掉电时最多丢最近一两条 WAL,会话历史这点损失可接受)。
_PRAGMAS = (
    "PRAGMA journal_mode = WAL",
    "PRAGMA synchronous = NORMAL",
    "PRAGMA busy_timeout = 5000",
    "PRAGMA foreign_keys = ON",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoredMessage:
    role: Role
    content: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PendingMessage:
    """append_messages 的入参 —— 同一个事务里把多条按顺序写下去。"""
    role: Role
    content: str
    meta: dict[str, Any] | None = None


class SqliteStore:
    """异步包装的 SQLite 存储。所有方法每次开新连接,简单可靠;
    单用户场景 QPS 极低,不值得维护连接池。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def _set_pragmas(self, conn: aiosqlite.Connection) -> None:
        for pragma in _PRAGMAS:
            await conn.execute(pragma)

    async def ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as conn:
            await self._set_pragmas(conn)
            await conn.executescript(SCHEMA)
            # P2 → P3 迁移:老库 messages 没有 meta 列。
            # 多进程同时启动时,两个 Agent 都做这步会让其中一个撞 "duplicate column"。
            # 重新查 PRAGMA 后再判断 + 用 try 兜底,确保多进程下幂等。
            cursor = await conn.execute("PRAGMA table_info(messages)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "meta" not in cols:
                logger.info("migrating messages table: ADD COLUMN meta")
                try:
                    await conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
                except aiosqlite.OperationalError as exc:
                    # 多进程并发迁移 — 已经被对面进程加上了
                    if "duplicate column" in str(exc).lower():
                        logger.info("meta column already added by concurrent process")
                    else:
                        raise
            await conn.commit()

    async def ensure_session(self, session_id: str, channel: str = "web", user_ref: str | None = None) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
            await self._set_pragmas(conn)
            await conn.execute(
                """
                INSERT INTO sessions (id, channel, user_ref, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, channel, user_ref, _now(), _now()),
            )
            await conn.commit()

    async def append_message(
        self,
        session_id: str,
        role: Role,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> int:
        ids = await self.append_messages(session_id, [PendingMessage(role=role, content=content, meta=meta)])
        return ids[0]

    async def append_messages(
        self,
        session_id: str,
        messages: Iterable[PendingMessage],
    ) -> list[int]:
        """把多条消息按顺序写在 **同一个事务** 里。

        Agent loop 写完 assistant(tool_calls) 后会立刻执行 tool 并写 tool 行,
        如果中途 crash,半成品 history 会让下一轮 LLM 拿到 "tool_calls 但缺
        tool_result" 的状态,很多 provider 会直接报错。一次事务 commit 至少
        让"刚把 assistant 写下去就死了"和"assistant + 所有 tool 都到位了"
        互斥,降低半成品概率。
        """
        msgs = list(messages)
        if not msgs:
            return []

        ids: list[int] = []
        async with aiosqlite.connect(self._db_path) as conn:
            await self._set_pragmas(conn)
            try:
                for m in msgs:
                    meta_text = json.dumps(m.meta, ensure_ascii=False, default=str) if m.meta else None
                    cursor = await conn.execute(
                        "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                        (session_id, m.role, m.content, meta_text, _now()),
                    )
                    assert cursor.lastrowid is not None
                    ids.append(cursor.lastrowid)
                await conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE id = ?",
                    (_now(), session_id),
                )
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
        return ids

    async def history(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[StoredMessage]:
        """按 id 升序返回历史。``limit`` 给定时返回 **最近** N 条(仍按时间正序)。

        旧实现用 ``ORDER BY id ASC LIMIT N`` 只会拿到最早 N 条,长会话后
        新消息会从 LLM 看不到 —— 这是个静默截断,P3 必须修。
        """
        async with aiosqlite.connect(self._db_path) as conn:
            await self._set_pragmas(conn)
            if limit is None:
                cursor = await conn.execute(
                    "SELECT role, content, meta, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                )
                rows = await cursor.fetchall()
            else:
                # 取最近 N 条,然后再翻回时间正序
                cursor = await conn.execute(
                    "SELECT role, content, meta, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                )
                tail = await cursor.fetchall()
                rows = list(reversed(tail))

        out: list[StoredMessage] = []
        for role, content, meta_text, created_at in rows:
            meta = json.loads(meta_text) if meta_text else {}
            out.append(StoredMessage(role=role, content=content, created_at=created_at, meta=meta))
        return out


def init_store(workspace_dir: Path | None = None) -> SqliteStore:
    """构造 ``SqliteStore``。

    路径优先级:
    1. ``$PRTS_DB_PATH``(用户显式覆盖,绝对/相对都可)
    2. ``<workspace_dir>/db/prts.db``(workspace 提供时;让 DB 跟着 workspace 走,
       开发者切换 ``PRTS_WORKSPACE_DIR`` 不会让 prod 库被踢到一旁)
    3. ``./db/prts.db``(都没给时的最后兜底,基本只在测试 / 临时跑)

    曾经的实现只有路径 (3),意味着从不同 CWD 启动 ``prts-agent`` 会落在不同
    位置,用户的 sessions/messages 会"消失"。把 workspace 作为默认锚点更符合
    "本地优先"的设计。
    """
    env = os.getenv("PRTS_DB_PATH")
    if env:
        db_path = Path(env).expanduser()
    elif workspace_dir is not None:
        db_path = workspace_dir / "db" / "prts.db"
    else:
        db_path = Path("./db/prts.db")
    return SqliteStore(db_path)
