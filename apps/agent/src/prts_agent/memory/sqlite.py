"""SQLite 持久化:会话 / 消息 / 工具调用 / 事件。

P3 起 messages 表新增 ``meta`` 列(JSON 文本),用来装:
- 助手发出工具调用时:``{"tool_calls": [{id,name,arguments}, ...]}``
- 工具结果消息:``{"tool_call_id":..., "tool_name":..., "is_error": bool}``

老库会被自动迁移(``ALTER TABLE``);幂等。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class StoredMessage:
    role: Role
    content: str
    created_at: str
    meta: dict[str, Any] = field(default_factory=dict)


class SqliteStore:
    """异步包装的 SQLite 存储。所有方法每次开新连接,简单可靠;
    单用户场景 QPS 极低,不值得维护连接池。"""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    async def ensure_schema(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self._db_path) as conn:
            await conn.executescript(SCHEMA)
            # P2 → P3 迁移:老库 messages 没有 meta 列。
            cursor = await conn.execute("PRAGMA table_info(messages)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "meta" not in cols:
                logger.info("migrating messages table: ADD COLUMN meta")
                await conn.execute("ALTER TABLE messages ADD COLUMN meta TEXT")
            await conn.commit()

    async def ensure_session(self, session_id: str, channel: str = "web", user_ref: str | None = None) -> None:
        async with aiosqlite.connect(self._db_path) as conn:
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
        meta_text = json.dumps(meta, ensure_ascii=False) if meta else None
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(
                "INSERT INTO messages (session_id, role, content, meta, created_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, meta_text, _now()),
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (_now(), session_id),
            )
            await conn.commit()
            assert cursor.lastrowid is not None
            return cursor.lastrowid

    async def history(self, session_id: str, limit: int | None = None) -> list[StoredMessage]:
        sql = (
            "SELECT role, content, meta, created_at FROM messages "
            "WHERE session_id = ? ORDER BY id ASC"
        )
        params: tuple[Any, ...] = (session_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (session_id, limit)
        async with aiosqlite.connect(self._db_path) as conn:
            cursor = await conn.execute(sql, params)
            rows = await cursor.fetchall()
        out: list[StoredMessage] = []
        for role, content, meta_text, created_at in rows:
            meta = json.loads(meta_text) if meta_text else {}
            out.append(StoredMessage(role=role, content=content, created_at=created_at, meta=meta))
        return out


def init_store() -> SqliteStore:
    db_path = Path(os.getenv("PRTS_DB_PATH", "./db/prts.db")).expanduser()
    return SqliteStore(db_path)
