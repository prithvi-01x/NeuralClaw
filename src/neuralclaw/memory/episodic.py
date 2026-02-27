"""
memory/episodic.py — Episodic Memory

SQLite-backed store for structured records of agent activity.
Think of it as the agent's diary — a log of what it did, when, and what happened.

Tables:
  - episodes    : completed task records (goal, steps, outcome)
  - tool_calls  : every tool invocation with params and results
  - reflections : lessons learned, notes the agent stores about itself

Usage:
    episodic = EpisodicMemory("./data/sqlite/episodes.db")
    await episodic.init()
    episode_id = await episodic.record_episode(goal="Research WebGPU", ...)
    history = await episodic.get_recent_episodes(n=5)
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)

# ── Schema DDL ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    goal        TEXT NOT NULL,
    outcome     TEXT,           -- 'success' | 'failure' | 'partial' | 'cancelled'
    summary     TEXT,           -- LLM-generated summary of what happened
    steps_json  TEXT,           -- JSON list of step descriptions
    tool_count  INTEGER DEFAULT 0,
    turn_count  INTEGER DEFAULT 0,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    metadata    TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS tool_calls (
    id              TEXT PRIMARY KEY,
    episode_id      TEXT,
    session_id      TEXT NOT NULL,
    tool_name       TEXT NOT NULL,
    arguments_json  TEXT,
    result_content  TEXT,
    is_error        INTEGER DEFAULT 0,
    risk_level      TEXT,
    duration_ms     REAL,
    timestamp       REAL NOT NULL,
    FOREIGN KEY (episode_id) REFERENCES episodes(id)
);

CREATE TABLE IF NOT EXISTS reflections (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    content     TEXT NOT NULL,   -- the lesson/note
    context     TEXT,            -- what triggered this reflection
    timestamp   REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
CREATE INDEX IF NOT EXISTS idx_episodes_started ON episodes(started_at);
CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_reflections_session ON reflections(session_id);
"""


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Episode:
    id: str
    session_id: str
    goal: str
    outcome: Optional[str] = None
    summary: Optional[str] = None
    steps: list[str] = field(default_factory=list)
    tool_count: int = 0
    turn_count: int = 0
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.ended_at:
            return self.ended_at - self.started_at
        return None


@dataclass
class ToolCallRecord:
    id: str
    session_id: str
    tool_name: str
    timestamp: float
    episode_id: Optional[str] = None
    arguments: dict[str, Any] = field(default_factory=dict)
    result_content: Optional[str] = None
    is_error: bool = False
    risk_level: Optional[str] = None
    duration_ms: Optional[float] = None


@dataclass
class Reflection:
    id: str
    session_id: str
    content: str
    timestamp: float
    context: Optional[str] = None


# ── Main class ────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Async SQLite-backed episodic memory store.

    Records what the agent did and when, enabling:
    - "What was I working on yesterday?"
    - "Which tools have I used most?"
    - "What went wrong last time I tried X?"
    """

    def __init__(self, db_path: str = "./data/sqlite/episodes.db"):
        self.db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Create the database file and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        log.info("episodic_memory.initialized", db_path=self.db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    def _require_db(self) -> None:
        """Raise clearly if the database connection is not open."""
        if self._db is None:
            raise RuntimeError(
                "EpisodicMemory is not initialised (or has been closed). "
                "Call `await episodic.init()` before use."
            )

    # ── Episodes ──────────────────────────────────────────────────────────────

    async def start_episode(
        self,
        session_id: str,
        goal: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Create a new episode record and return its ID."""
        self._require_db()
        episode_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO episodes
               (id, session_id, goal, started_at, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (episode_id, session_id, goal, time.time(), json.dumps(metadata or {})),
        )
        await self._db.commit()
        log.debug("episodic.episode_started", episode_id=episode_id, goal=goal[:80])
        return episode_id

    async def finish_episode(
        self,
        episode_id: str,
        outcome: str,
        summary: Optional[str] = None,
        steps: Optional[list[str]] = None,
        tool_count: int = 0,
        turn_count: int = 0,
    ) -> None:
        """Update an episode with its outcome and summary."""
        self._require_db()
        await self._db.execute(
            """UPDATE episodes SET
               outcome=?, summary=?, steps_json=?,
               tool_count=?, turn_count=?, ended_at=?
               WHERE id=?""",
            (
                outcome,
                summary,
                json.dumps(steps or []),
                tool_count,
                turn_count,
                time.time(),
                episode_id,
            ),
        )
        await self._db.commit()
        log.debug("episodic.episode_finished", episode_id=episode_id, outcome=outcome)

    async def get_recent_episodes(
        self,
        n: int = 10,
        session_id: Optional[str] = None,
    ) -> list[Episode]:
        """Return the most recent episodes, optionally filtered by session."""
        self._require_db()
        if session_id:
            cursor = await self._db.execute(
                "SELECT * FROM episodes WHERE session_id=? ORDER BY started_at DESC LIMIT ?",
                (session_id, n),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM episodes ORDER BY started_at DESC LIMIT ?",
                (n,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    async def search_episodes(self, query: str, n: int = 5) -> list[Episode]:
        """Full-text search over episode goals and summaries."""
        self._require_db()
        # Escape SQLite LIKE special characters so user input isn't treated as wildcards
        escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        like = f"%{escaped}%"
        cursor = await self._db.execute(
            """SELECT * FROM episodes
               WHERE goal LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\'
               ORDER BY started_at DESC LIMIT ?""",
            (like, like, n),
        )
        rows = await cursor.fetchall()
        return [self._row_to_episode(r) for r in rows]

    # ── Tool calls ────────────────────────────────────────────────────────────

    async def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result_content: Optional[str] = None,
        is_error: bool = False,
        risk_level: Optional[str] = None,
        duration_ms: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """Log a single tool invocation. Returns the record ID."""
        self._require_db()
        record_id = str(uuid.uuid4())
        await self._db.execute(
            """INSERT INTO tool_calls
               (id, episode_id, session_id, tool_name, arguments_json,
                result_content, is_error, risk_level, duration_ms, timestamp)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record_id,
                episode_id,
                session_id,
                tool_name,
                json.dumps(arguments),
                result_content,
                int(is_error),
                risk_level,
                duration_ms,
                time.time(),
            ),
        )
        await self._db.commit()
        return record_id

    async def get_tool_call_stats(self) -> list[dict[str, Any]]:
        """Return tool usage counts, sorted by most used."""
        self._require_db()
        cursor = await self._db.execute(
            """SELECT tool_name,
                      COUNT(*) as total,
                      SUM(is_error) as errors,
                      AVG(duration_ms) as avg_duration_ms
               FROM tool_calls
               GROUP BY tool_name
               ORDER BY total DESC"""
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_recent_tool_calls(
        self,
        n: int = 20,
        tool_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> list[ToolCallRecord]:
        self._require_db()
        if tool_name and session_id:
            cursor = await self._db.execute(
                "SELECT * FROM tool_calls WHERE tool_name=? AND session_id=? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, session_id, n),
            )
        elif tool_name:
            cursor = await self._db.execute(
                "SELECT * FROM tool_calls WHERE tool_name=? ORDER BY timestamp DESC LIMIT ?",
                (tool_name, n),
            )
        elif session_id:
            cursor = await self._db.execute(
                "SELECT * FROM tool_calls WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
                (session_id, n),
            )
        else:
            cursor = await self._db.execute(
                "SELECT * FROM tool_calls ORDER BY timestamp DESC LIMIT ?",
                (n,),
            )
        rows = await cursor.fetchall()
        return [self._row_to_tool_call(r) for r in rows]

    # ── Reflections ───────────────────────────────────────────────────────────

    async def add_reflection(
        self,
        session_id: str,
        content: str,
        context: Optional[str] = None,
    ) -> str:
        """Store a reflection/lesson-learned."""
        self._require_db()
        reflection_id = str(uuid.uuid4())
        await self._db.execute(
            "INSERT INTO reflections (id, session_id, content, context, timestamp) VALUES (?, ?, ?, ?, ?)",
            (reflection_id, session_id, content, context, time.time()),
        )
        await self._db.commit()
        return reflection_id

    async def get_recent_reflections(self, n: int = 10) -> list[Reflection]:
        self._require_db()
        cursor = await self._db.execute(
            "SELECT * FROM reflections ORDER BY timestamp DESC LIMIT ?", (n,)
        )
        rows = await cursor.fetchall()
        return [self._row_to_reflection(r) for r in rows]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _row_to_episode(self, row) -> Episode:
        return Episode(
            id=row["id"],
            session_id=row["session_id"],
            goal=row["goal"],
            outcome=row["outcome"],
            summary=row["summary"],
            steps=json.loads(row["steps_json"] or "[]"),
            tool_count=row["tool_count"] or 0,
            turn_count=row["turn_count"] or 0,
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    def _row_to_tool_call(self, row) -> ToolCallRecord:
        return ToolCallRecord(
            id=row["id"],
            episode_id=row["episode_id"],
            session_id=row["session_id"],
            tool_name=row["tool_name"],
            arguments=json.loads(row["arguments_json"] or "{}"),
            result_content=row["result_content"],
            is_error=bool(row["is_error"]),
            risk_level=row["risk_level"],
            duration_ms=row["duration_ms"],
            timestamp=row["timestamp"],
        )

    def _row_to_reflection(self, row) -> Reflection:
        return Reflection(
            id=row["id"],
            session_id=row["session_id"],
            content=row["content"],
            context=row["context"],
            timestamp=row["timestamp"],
        )