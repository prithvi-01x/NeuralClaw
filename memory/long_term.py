"""
memory/long_term.py — Long-Term Memory (ChromaDB)

Persists agent knowledge, facts, and conversation summaries across sessions
using ChromaDB as a local vector store.

Fix applied:
  - Telemetry disabled via chromadb.Settings(anonymized_telemetry=False)
    to eliminate the "capture() takes 1 positional argument but 3 were given"
    error spam caused by a posthog version mismatch in the chromadb package.
"""

from __future__ import annotations

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from memory.embedder import Embedder
from observability.logger import get_logger

log = get_logger(__name__)

DEFAULT_COLLECTION = "knowledge"

COLLECTIONS = [
    "knowledge",       # general facts and research
    "conversations",   # summarised past interactions
    "reflections",     # agent self-reflections and lessons
    "user_prefs",      # user preferences and context
]


@dataclass
class MemoryEntry:
    id: str
    text: str
    collection: str
    metadata: dict = field(default_factory=dict)
    relevance_score: float = 0.0


class LongTermMemory:
    """
    Async wrapper around ChromaDB for persistent vector memory.

    Collections:
        knowledge     — general facts learned during tasks
        conversations — summaries of past sessions
        reflections   — lessons from completed autonomous tasks
        user_prefs    — stable user preferences / context

    Usage:
        lt = LongTermMemory(persist_dir="./data/chroma", embedder=embedder)
        await lt.init()
        doc_id = await lt.store("Python asyncio overview", collection="knowledge")
        results = await lt.search("async programming", n=5)
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        embedder: Optional[Embedder] = None,
        relevance_threshold: float = 0.5,
    ):
        self.persist_dir = persist_dir
        self._embedder = embedder or Embedder()
        self._relevance_threshold = relevance_threshold
        self._client = None
        self._collections: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chromadb")
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        """Initialize ChromaDB client and create all collections."""
        async with self._lock:
            if self._client is not None:
                return
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._executor, self._setup_chroma)
            log.info("long_term_memory.initialized", persist_dir=self.persist_dir)

    async def store(
        self,
        text: str,
        collection: str = DEFAULT_COLLECTION,
        metadata: Optional[dict] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Store a text document with its embedding in the specified collection.

        Returns:
            The document ID.
        """
        await self._ensure_init()
        doc_id = doc_id or f"doc_{uuid.uuid4().hex}"
        embedding = await self._embedder.embed(text)
        meta = metadata or {}

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._chroma_add,
            collection,
            doc_id,
            text,
            embedding,
            meta,
        )
        return doc_id

    async def search(
        self,
        query: str,
        collection: str = DEFAULT_COLLECTION,
        n: int = 5,
    ) -> list[MemoryEntry]:
        """
        Semantic search within a single collection.

        Returns:
            List of MemoryEntry objects sorted by relevance (highest first).
            Entries below relevance_threshold are filtered out.
        """
        await self._ensure_init()
        query_embedding = await self._embedder.embed(query)

        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(
            self._executor,
            self._chroma_query,
            collection,
            query_embedding,
            n,
        )

        entries = []
        if raw and raw.get("ids"):
            for i, doc_id in enumerate(raw["ids"][0]):
                distance = raw["distances"][0][i] if raw.get("distances") else 1.0
                # ChromaDB cosine distance: 0 = identical, 2 = opposite
                # Convert to similarity score in [0, 1]
                score = max(0.0, 1.0 - (distance / 2.0))
                if score < self._relevance_threshold:
                    continue
                entries.append(MemoryEntry(
                    id=doc_id,
                    text=raw["documents"][0][i] if raw.get("documents") else "",
                    collection=collection,
                    metadata=raw["metadatas"][0][i] if raw.get("metadatas") else {},
                    relevance_score=score,
                ))
        return sorted(entries, key=lambda e: e.relevance_score, reverse=True)

    async def search_all(
        self,
        query: str,
        n_per_collection: int = 3,
    ) -> dict[str, list[MemoryEntry]]:
        """Search across all collections and return results grouped by collection."""
        tasks = {
            col: self.search(query, collection=col, n=n_per_collection)
            for col in COLLECTIONS
        }
        results = {}
        for col, coro in tasks.items():
            entries = await coro
            if entries:
                results[col] = entries
        return results

    async def delete(self, doc_id: str, collection: str = DEFAULT_COLLECTION) -> None:
        """Delete a document by ID."""
        await self._ensure_init()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._chroma_delete,
            collection,
            doc_id,
        )

    async def clear_collection(self, collection: str) -> None:
        """Delete all documents in a collection."""
        await self._ensure_init()
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self._executor,
            self._chroma_clear,
            collection,
        )

    async def count(self, collection: str = DEFAULT_COLLECTION) -> int:
        """Return the number of documents in a collection."""
        await self._ensure_init()
        if collection not in self._collections:
            return 0
        return self._collections[collection].count()

    async def close(self) -> None:
        """Shut down the thread-pool executor cleanly."""
        self._executor.shutdown(wait=True)
        log.debug("long_term_memory.closed")

    # ── Private / sync helpers (run in thread-pool) ───────────────────────────

    def _setup_chroma(self) -> None:
        import chromadb
        from chromadb.config import Settings

        # Disable telemetry to prevent "capture() takes 1 positional argument
        # but 3 were given" error spam caused by a posthog version mismatch.
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        for name in COLLECTIONS:
            self._collections[name] = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )

    def _chroma_add(
        self,
        collection: str,
        doc_id: str,
        text: str,
        embedding: list[float],
        metadata: dict,
    ) -> None:
        col = self._collections.get(collection)
        if col is None:
            raise ValueError(f"Unknown collection: {collection!r}")
        col.upsert(
            ids=[doc_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )

    def _chroma_query(
        self,
        collection: str,
        embedding: list[float],
        n: int,
    ) -> dict:
        col = self._collections.get(collection)
        if col is None:
            return {}
        count = col.count()
        if count == 0:
            return {}
        return col.query(
            query_embeddings=[embedding],
            n_results=min(n, count),
            include=["documents", "metadatas", "distances"],
        )

    def _chroma_delete(self, collection: str, doc_id: str) -> None:
        col = self._collections.get(collection)
        if col:
            col.delete(ids=[doc_id])

    def _chroma_clear(self, collection: str) -> None:
        col = self._collections.get(collection)
        if col:
            col.delete(where={"_id": {"$ne": ""}})

    async def _ensure_init(self) -> None:
        if self._client is None:
            await self.init()