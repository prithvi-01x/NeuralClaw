"""
memory/long_term.py — Long-Term Vector Memory

Persistent semantic memory backed by ChromaDB.
Stores text with embeddings for fuzzy/semantic search across sessions.

Collections:
  - conversations : summaries of past dialogue sessions
  - knowledge     : facts, snippets, learned context
  - tool_results  : significant tool outputs worth remembering
  - plans         : completed or failed task plans

Usage:
    lt = LongTermMemory(persist_dir="./data/chroma", embedder=embedder)
    await lt.store("Python is great for async I/O", collection="knowledge")
    results = await lt.search("async programming", collection="knowledge", n=3)
"""

from __future__ import annotations

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional

from observability.logger import get_logger

log = get_logger(__name__)

# Valid collection names
COLLECTIONS = frozenset(["conversations", "knowledge", "tool_results", "plans"])
DEFAULT_COLLECTION = "knowledge"
_MAX_RESULTS = 20


@dataclass
class MemoryEntry:
    """A single entry retrieved from long-term memory."""
    id: str
    text: str
    collection: str
    distance: float          # 0.0 = identical, 2.0 = completely different
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    @property
    def relevance_score(self) -> float:
        """Convert distance to a 0-1 relevance score (1 = most relevant)."""
        return max(0.0, 1.0 - (self.distance / 2.0))


class LongTermMemory:
    """
    ChromaDB-backed vector store for semantic memory.

    Uses local embeddings (sentence-transformers) — no external API calls.
    All ChromaDB operations are synchronous; we wrap them in thread pool
    so they don't block the async event loop.
    """

    def __init__(
        self,
        persist_dir: str = "./data/chroma",
        embedder=None,
        relevance_threshold: float = 0.85,
    ):
        self.persist_dir = persist_dir
        self.embedder = embedder
        self.relevance_threshold = relevance_threshold
        self._client = None
        self._collections: dict[str, Any] = {}
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="chromadb")
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        """Initialize ChromaDB client and create all collections."""
        async with self._lock:
            if self._client is not None:
                return
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self._executor, self._setup_chroma)
            log.info("long_term_memory.initialized", persist_dir=self.persist_dir)

    async def store(
        self,
        text: str,
        collection: str = DEFAULT_COLLECTION,
        metadata: Optional[dict[str, Any]] = None,
        doc_id: Optional[str] = None,
    ) -> str:
        """
        Embed and store a text entry in the specified collection.

        Args:
            text:       Text content to store.
            collection: One of: conversations, knowledge, tool_results, plans.
            metadata:   Optional key-value metadata (filterable).
            doc_id:     Optional ID — auto-generated if not provided.

        Returns:
            The document ID of the stored entry.
        """
        await self._ensure_init()
        collection = self._validate_collection(collection)
        doc_id = doc_id or str(uuid.uuid4())

        meta = {
            "timestamp": time.time(),
            "collection": collection,
            **(metadata or {}),
        }

        # Embed in async context
        if self.embedder:
            embedding = await self.embedder.embed(text)
        else:
            embedding = None

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._chroma_add,
            collection,
            doc_id,
            text,
            meta,
            embedding,
        )

        log.debug(
            "long_term_memory.stored",
            collection=collection,
            doc_id=doc_id,
            text_length=len(text),
        )
        return doc_id

    async def search(
        self,
        query: str,
        collection: str = DEFAULT_COLLECTION,
        n: int = 5,
        where: Optional[dict] = None,
    ) -> list[MemoryEntry]:
        """
        Semantic search over a collection.

        Args:
            query:      Natural language query.
            collection: Collection to search.
            n:          Max results to return.
            where:      Optional ChromaDB metadata filter.

        Returns:
            List of MemoryEntry sorted by relevance (most relevant first).
        """
        await self._ensure_init()
        collection = self._validate_collection(collection)
        n = min(n, _MAX_RESULTS)

        if self.embedder:
            query_embedding = await self.embedder.embed(query)
        else:
            query_embedding = None

        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            self._executor,
            self._chroma_query,
            collection,
            query,
            query_embedding,
            n,
            where,
        )

        return self._parse_results(raw, collection)

    async def delete(self, doc_id: str, collection: str = DEFAULT_COLLECTION) -> bool:
        """Delete a document from a collection by ID."""
        await self._ensure_init()
        collection = self._validate_collection(collection)
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                self._executor,
                self._chroma_delete,
                collection,
                doc_id,
            )
            return True
        except Exception as e:
            log.warning("long_term_memory.delete_failed", doc_id=doc_id, error=str(e))
            return False

    async def count(self, collection: str = DEFAULT_COLLECTION) -> int:
        """Return the number of documents in a collection."""
        await self._ensure_init()
        collection = self._validate_collection(collection)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor,
            lambda: self._collections[collection].count(),
        )

    async def clear_collection(self, collection: str) -> None:
        """Delete all documents in a collection."""
        await self._ensure_init()
        collection = self._validate_collection(collection)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self._executor,
            self._chroma_clear,
            collection,
        )
        log.info("long_term_memory.collection_cleared", collection=collection)

    # ── Private: ChromaDB operations (run in thread pool) ─────────────────────

    def _setup_chroma(self) -> None:
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
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
        metadata: dict,
        embedding: Optional[list[float]],
    ) -> None:
        col = self._collections[collection]
        kwargs: dict[str, Any] = {
            "ids": [doc_id],
            "documents": [text],
            "metadatas": [metadata],
        }
        if embedding:
            kwargs["embeddings"] = [embedding]
        col.add(**kwargs)

    def _chroma_query(
        self,
        collection: str,
        query_text: str,
        query_embedding: Optional[list[float]],
        n: int,
        where: Optional[dict],
    ) -> dict:
        col = self._collections[collection]
        kwargs: dict[str, Any] = {"n_results": min(n, max(col.count(), 1))}
        if query_embedding:
            kwargs["query_embeddings"] = [query_embedding]
        else:
            kwargs["query_texts"] = [query_text]
        if where:
            kwargs["where"] = where
        return col.query(**kwargs)

    def _chroma_delete(self, collection: str, doc_id: str) -> None:
        self._collections[collection].delete(ids=[doc_id])

    def _chroma_clear(self, collection: str) -> None:
        col = self._collections[collection]
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)

    def _parse_results(self, raw: dict, collection: str) -> list[MemoryEntry]:
        entries = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return entries

        ids = raw["ids"][0]
        docs = raw.get("documents", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]

        for i, doc_id in enumerate(ids):
            entries.append(MemoryEntry(
                id=doc_id,
                text=docs[i] if i < len(docs) else "",
                collection=collection,
                distance=distances[i] if i < len(distances) else 1.0,
                metadata=metadatas[i] if i < len(metadatas) else {},
            ))

        return entries

    async def _ensure_init(self) -> None:
        if self._client is None:
            await self.init()

    def _validate_collection(self, collection: str) -> str:
        if collection not in COLLECTIONS:
            raise ValueError(
                f"Unknown collection: '{collection}'. Valid: {sorted(COLLECTIONS)}"
            )
        return collection