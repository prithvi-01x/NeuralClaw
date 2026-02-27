"""
gateway/session_store.py — Centralized Session Store

Owns all Session objects. Maps session_id → Session.
Uses asyncio.Lock for safe concurrent access in the async gateway.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from agent.session import Session
from skills.types import TrustLevel
from observability.logger import get_logger

log = get_logger(__name__)


class SessionStore:
    """
    Async-safe session store for the gateway.

    Sessions are created on-demand or explicitly via create().
    The store is shared across all WebSocket connections.
    """

    def __init__(self, default_trust: TrustLevel = TrustLevel.LOW):
        self._sessions: dict[str, Session] = {}
        self._lock = asyncio.Lock()
        self._default_trust = default_trust

    async def create(
        self,
        user_id: str = "gateway",
        trust_level: Optional[TrustLevel] = None,
    ) -> Session:
        """Create a new session and return it."""
        trust = trust_level or self._default_trust
        session = Session.create(user_id=user_id, trust_level=trust)
        async with self._lock:
            self._sessions[session.id] = session
        log.info("session_store.created", session_id=session.id, trust=trust.value)
        return session

    async def get(self, session_id: str) -> Optional[Session]:
        """Get a session by ID, or None if not found."""
        async with self._lock:
            return self._sessions.get(session_id)

    async def get_or_create(
        self,
        session_id: Optional[str] = None,
        user_id: str = "gateway",
    ) -> Session:
        """Get an existing session or create a new one."""
        if session_id:
            session = await self.get(session_id)
            if session:
                return session
        return await self.create(user_id=user_id)

    async def remove(self, session_id: str) -> bool:
        """Remove a session. Returns True if it existed."""
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def list_sessions(self) -> list[str]:
        """Return all active session IDs."""
        async with self._lock:
            return list(self._sessions.keys())

    async def get_count(self) -> int:
        """Return the number of active sessions."""
        async with self._lock:
            return len(self._sessions)

    @property
    def count(self) -> int:
        """Synchronous count — use only from non-async contexts (e.g. tests)."""
        return len(self._sessions)
