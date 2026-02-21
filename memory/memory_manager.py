"""
memory/memory_manager.py — Memory Manager

The single entry point for all memory operations in the agent.
Coordinates short-term, long-term (ChromaDB), and episodic (SQLite) memory.

The orchestrator only ever calls this — it never touches the underlying
stores directly.

Usage:
    mm = MemoryManager.from_settings(settings)
    await mm.init()

    # Store information
    await mm.store("Python asyncio is used for async programming", collection="knowledge")

    # Retrieve relevant context
    results = await mm.search("async python", n=3)

    # Record tool usage
    await mm.record_tool_call(session_id, "file_read", {...}, result="content")

    # Commit a completed task to episodic memory
    await mm.commit_episode(session_id, goal="Research WebGPU", outcome="success", ...)
"""

from __future__ import annotations

import time
from typing import Any, Optional

from memory.embedder import Embedder
from memory.episodic import Episode, EpisodicMemory, Reflection, ToolCallRecord
from memory.long_term import LongTermMemory, MemoryEntry
from memory.short_term import ShortTermMemory
from observability.logger import get_logger

log = get_logger(__name__)


class MemoryManager:
    """
    Unified interface over all NeuralClaw memory subsystems.

    Lifecycle:
        mm = MemoryManager(...)
        await mm.init()          # connect to DBs, load embedder
        ...                      # use during agent operation
        await mm.close()         # flush and close connections
    """

    def __init__(
        self,
        chroma_persist_dir: str = "./data/chroma",
        sqlite_path: str = "./data/sqlite/episodes.db",
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        max_short_term_turns: int = 20,
        relevance_threshold: float = 0.85,
    ):
        self.embedder = Embedder(model_name=embedding_model)
        self.long_term = LongTermMemory(
            persist_dir=chroma_persist_dir,
            embedder=self.embedder,
            relevance_threshold=relevance_threshold,
        )
        self.episodic = EpisodicMemory(db_path=sqlite_path)
        self._max_short_term_turns = max_short_term_turns
        self._sessions: dict[str, ShortTermMemory] = {}
        self._initialized = False

    async def init(self, load_embedder: bool = True) -> None:
        """
        Initialize all memory subsystems.
        Call once at agent startup.

        Args:
            load_embedder: If True, preload the embedding model (recommended).
                           Set False in tests to skip the slow model load.
        """
        if self._initialized:
            return

        log.info("memory_manager.initializing")

        if load_embedder:
            await self.embedder.load()

        await self.long_term.init()
        await self.episodic.init()
        self._initialized = True

        log.info("memory_manager.ready")

    async def close(self) -> None:
        """Flush and close all database connections."""
        await self.episodic.close()
        log.info("memory_manager.closed")

    # ── Session management ────────────────────────────────────────────────────

    def get_session(self, session_id: str, user_id: str = "local") -> ShortTermMemory:
        """
        Get or create short-term memory for a session.
        Sessions are in-process only — not persisted.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = ShortTermMemory(
                session_id=session_id,
                user_id=user_id,
                max_turns=self._max_short_term_turns,
            )
            log.debug("memory_manager.session_created", session_id=session_id)
        return self._sessions[session_id]

    def clear_session(self, session_id: str) -> None:
        """Remove a session's short-term memory."""
        if session_id in self._sessions:
            del self._sessions[session_id]

    # ── Long-term memory operations ───────────────────────────────────────────

    async def store(
        self,
        text: str,
        collection: str = "knowledge",
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """Store text in long-term vector memory. Returns document ID."""
        self._require_init()
        return await self.long_term.store(text, collection, metadata, doc_id)

    async def search(
        self,
        query: str,
        collection: str = "knowledge",
        n: int = 5,
        where: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """Semantic search over long-term memory."""
        self._require_init()
        return await self.long_term.search(query, collection, n, where)

    async def search_all(
        self,
        query: str,
        n_per_collection: int = 3,
    ) -> dict[str, list[MemoryEntry]]:
        """Search across all collections and return combined results."""
        self._require_init()
        from memory.long_term import COLLECTIONS
        results = {}
        for col in COLLECTIONS:
            try:
                hits = await self.long_term.search(query, col, n_per_collection)
                if hits:
                    results[col] = hits
            except Exception as e:
                log.warning("memory_manager.search_collection_failed", collection=col, error=str(e))
        return results

    async def store_conversation_summary(
        self,
        session_id: str,
        summary: str,
        turn_count: int = 0,
    ) -> str:
        """Store a conversation summary in long-term memory."""
        return await self.store(
            summary,
            collection="conversations",
            metadata={"session_id": session_id, "turn_count": turn_count},
        )

    async def store_tool_result(
        self,
        tool_name: str,
        result: str,
        query_context: str = "",
    ) -> str:
        """Store a significant tool result for future recall."""
        text = f"Tool '{tool_name}' result: {result}"
        if query_context:
            text = f"Context: {query_context}\n{text}"
        return await self.store(
            text,
            collection="tool_results",
            metadata={"tool_name": tool_name},
        )

    # ── Episodic memory operations ────────────────────────────────────────────

    async def start_episode(self, session_id: str, goal: str) -> str:
        """Start tracking a new task episode. Returns episode ID."""
        self._require_init()
        return await self.episodic.start_episode(session_id, goal)

    async def commit_episode(
        self,
        episode_id: str,
        outcome: str,
        summary: Optional[str] = None,
        steps: Optional[list[str]] = None,
        tool_count: int = 0,
        turn_count: int = 0,
    ) -> None:
        """Finish and commit a task episode to episodic memory."""
        self._require_init()
        await self.episodic.finish_episode(
            episode_id, outcome, summary, steps, tool_count, turn_count
        )
        # Also store summary in long-term vector memory for semantic search
        if summary:
            await self.store(
                summary,
                collection="plans",
                metadata={"episode_id": episode_id, "outcome": outcome},
            )

    async def record_tool_call(
        self,
        session_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: Optional[str] = None,
        is_error: bool = False,
        risk_level: Optional[str] = None,
        duration_ms: Optional[float] = None,
        episode_id: Optional[str] = None,
    ) -> str:
        """Record a tool invocation in episodic memory."""
        self._require_init()
        return await self.episodic.record_tool_call(
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            result_content=result,
            is_error=is_error,
            risk_level=risk_level,
            duration_ms=duration_ms,
            episode_id=episode_id,
        )

    async def get_recent_episodes(self, n: int = 5) -> list[Episode]:
        self._require_init()
        return await self.episodic.get_recent_episodes(n)

    async def search_episodes(self, query: str, n: int = 5) -> list[Episode]:
        self._require_init()
        return await self.episodic.search_episodes(query, n)

    async def add_reflection(
        self, session_id: str, content: str, context: Optional[str] = None
    ) -> str:
        self._require_init()
        return await self.episodic.add_reflection(session_id, content, context)

    # ── Context building helper ───────────────────────────────────────────────

    async def build_memory_context(
        self,
        query: str,
        session_id: str,
        n_long_term: int = 5,
    ) -> str:
        """
        Build a text block of relevant memories to inject into the system prompt.

        Searches long-term memory for context relevant to the current query,
        then formats it as a readable block the LLM can reference.
        """
        self._require_init()
        results = await self.search_all(query, n_per_collection=2)
        if not results:
            return ""

        lines = ["## Relevant Memory\n"]
        for collection, entries in results.items():
            for entry in entries:
                if entry.relevance_score > 0.3:
                    lines.append(f"[{collection}] {entry.text[:300]}")

        if len(lines) == 1:
            return ""

        return "\n".join(lines)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, settings) -> "MemoryManager":
        """Create a MemoryManager from the NeuralClaw Settings object."""
        return cls(
            chroma_persist_dir=settings.memory.get("chroma_persist_dir", "./data/chroma"),
            sqlite_path=settings.memory.get("sqlite_path", "./data/sqlite/episodes.db"),
            embedding_model=settings.memory.get("embedding_model", "BAAI/bge-small-en-v1.5"),
            max_short_term_turns=settings.memory.get("max_short_term_turns", 20),
            relevance_threshold=settings.memory.get("relevance_threshold", 0.85),
        )

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_init(self) -> None:
        if not self._initialized:
            raise RuntimeError(
                "MemoryManager not initialized. Call `await mm.init()` first."
            )