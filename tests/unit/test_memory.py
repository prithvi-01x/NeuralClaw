"""
tests/unit/test_memory.py — Memory System Unit Tests

Tests all memory components with mocks — no real ChromaDB,
SQLite, or sentence-transformers model required.

Run with:
    pytest tests/unit/test_memory.py -v
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from brain.types import Message, Role
from memory.short_term import (
    ConversationBuffer,
    SessionState,
    ShortTermMemory,
    ToolResultEntry,
    ToolResultCache,
)
from memory.long_term import MemoryEntry, LongTermMemory
from memory.episodic import EpisodicMemory, Episode
from memory.memory_manager import MemoryManager


# ─────────────────────────────────────────────────────────────────────────────
# ConversationBuffer tests
# ─────────────────────────────────────────────────────────────────────────────


class TestConversationBuffer:
    def test_add_and_retrieve(self):
        buf = ConversationBuffer(max_turns=10)
        buf.add(Message.user("Hello"))
        buf.add(Message.assistant("Hi!"))
        msgs = buf.get_messages(include_system=False)
        assert len(msgs) == 2
        assert msgs[0].content == "Hello"
        assert msgs[1].content == "Hi!"

    def test_system_prompt_prepended(self):
        buf = ConversationBuffer()
        buf.set_system_prompt("You are helpful")
        buf.add(Message.user("Hello"))
        msgs = buf.get_messages(include_system=True)
        assert msgs[0].role == Role.SYSTEM
        assert msgs[0].content == "You are helpful"
        assert msgs[1].role == Role.USER

    def test_system_message_in_add_sets_prompt(self):
        buf = ConversationBuffer()
        buf.add(Message.system("System prompt"))
        # System messages go to the prompt, not the message list
        assert buf.message_count == 0
        msgs = buf.get_messages(include_system=True)
        assert msgs[0].role == Role.SYSTEM

    def test_eviction_at_max_turns(self):
        buf = ConversationBuffer(max_turns=2)
        for i in range(4):
            buf.add(Message.user(f"User {i}"))
            buf.add(Message.assistant(f"Assistant {i}"))
        # Should only keep the last 2 turns (4 messages)
        assert buf.message_count <= 4

    def test_clear_keeps_system_prompt(self):
        buf = ConversationBuffer()
        buf.set_system_prompt("Keep me")
        buf.add(Message.user("Delete me"))
        buf.clear()
        assert buf.message_count == 0
        msgs = buf.get_messages(include_system=True)
        assert msgs[0].content == "Keep me"

    def test_get_recent(self):
        buf = ConversationBuffer()
        for i in range(10):
            buf.add(Message.user(f"msg {i}"))
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "msg 9"

    def test_turn_count(self):
        buf = ConversationBuffer()
        buf.add(Message.user("Q1"))
        buf.add(Message.assistant("A1"))
        buf.add(Message.user("Q2"))
        buf.add(Message.assistant("A2"))
        assert buf.turn_count == 2

    def test_to_text(self):
        buf = ConversationBuffer()
        buf.add(Message.user("Hello"))
        buf.add(Message.assistant("Hi!"))
        text = buf.to_text()
        assert "USER: Hello" in text
        assert "ASSISTANT: Hi!" in text

    def test_add_user_assistant_shortcuts(self):
        buf = ConversationBuffer()
        buf.add_user("Hello")
        buf.add_assistant("Hi")
        assert buf.message_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# ToolResultCache tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolResultCache:
    def test_add_and_retrieve(self):
        cache = ToolResultCache(max_results=5)
        cache.add(ToolResultEntry("id1", "search", "results"))
        recent = cache.get_recent(5)
        assert len(recent) == 1
        assert recent[0].tool_call_id == "id1"

    def test_max_size_enforced(self):
        cache = ToolResultCache(max_results=3)
        for i in range(5):
            cache.add(ToolResultEntry(f"id{i}", "tool", f"result{i}"))
        assert len(cache) == 3

    def test_get_by_id(self):
        cache = ToolResultCache()
        cache.add(ToolResultEntry("find_me", "search", "content"))
        entry = cache.get_by_id("find_me")
        assert entry is not None
        assert entry.content == "content"

    def test_get_by_id_missing(self):
        cache = ToolResultCache()
        assert cache.get_by_id("missing") is None

    def test_clear(self):
        cache = ToolResultCache()
        cache.add(ToolResultEntry("id1", "tool", "result"))
        cache.clear()
        assert len(cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# ShortTermMemory tests
# ─────────────────────────────────────────────────────────────────────────────


class TestShortTermMemory:
    def test_creation(self):
        stm = ShortTermMemory("sess_123", user_id="user_1")
        assert stm.state.session_id == "sess_123"
        assert stm.state.user_id == "user_1"

    def test_add_message_increments_turn_count(self):
        stm = ShortTermMemory("sess_1")
        stm.add_message(Message.user("Hello"))
        assert stm.state.turn_count == 1
        stm.add_message(Message.assistant("Hi"))
        assert stm.state.turn_count == 1  # Only user messages count

    def test_add_tool_result(self):
        stm = ShortTermMemory("sess_1")
        stm.add_tool_result("call_1", "web_search", "search results")
        assert len(stm.tool_results) == 1

    def test_get_context_messages_with_system_prompt(self):
        stm = ShortTermMemory("sess_1")
        stm.add_message(Message.user("Hello"))
        msgs = stm.get_context_messages(system_prompt="You are helpful")
        assert msgs[0].role == Role.SYSTEM
        assert msgs[1].role == Role.USER

    def test_clear_conversation(self):
        stm = ShortTermMemory("sess_1")
        stm.add_message(Message.user("Hello"))
        stm.clear_conversation()
        msgs = stm.get_context_messages()
        # Only system message (if any) remains
        assert all(m.role == Role.SYSTEM for m in msgs)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryEntry tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryEntry:
    def test_relevance_score_perfect(self):
        entry = MemoryEntry(id="1", text="t", collection="knowledge", distance=0.0)
        assert entry.relevance_score == 1.0

    def test_relevance_score_half(self):
        entry = MemoryEntry(id="1", text="t", collection="knowledge", distance=1.0)
        assert entry.relevance_score == 0.5

    def test_relevance_score_none(self):
        entry = MemoryEntry(id="1", text="t", collection="knowledge", distance=2.0)
        assert entry.relevance_score == 0.0

    def test_relevance_score_clamped(self):
        entry = MemoryEntry(id="1", text="t", collection="knowledge", distance=3.0)
        assert entry.relevance_score == 0.0  # clamped, not negative


# ─────────────────────────────────────────────────────────────────────────────
# LongTermMemory tests (mocked ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────


class TestLongTermMemory:
    @pytest.fixture
    def mock_embedder(self):
        embedder = MagicMock()
        embedder.embed = AsyncMock(return_value=[0.1] * 384)
        embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384])
        return embedder

    @pytest.fixture
    def lt_memory(self, mock_embedder):
        lt = LongTermMemory(persist_dir="/tmp/test_chroma", embedder=mock_embedder)
        # Mock the ChromaDB collection
        mock_col = MagicMock()
        mock_col.count.return_value = 1
        mock_col.add.return_value = None
        mock_col.query.return_value = {
            "ids": [["doc_1"]],
            "documents": [["Python async programming is great"]],
            "distances": [[0.1]],
            "metadatas": [[{"timestamp": time.time(), "collection": "knowledge"}]],
        }
        lt._collections = {
            "knowledge": mock_col,
            "conversations": mock_col,
            "tool_results": mock_col,
            "plans": mock_col,
        }
        lt._client = MagicMock()  # mark as initialized
        return lt

    @pytest.mark.asyncio
    async def test_store(self, lt_memory):
        doc_id = await lt_memory.store("Python async patterns", collection="knowledge")
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0

    @pytest.mark.asyncio
    async def test_search_returns_entries(self, lt_memory):
        results = await lt_memory.search("async python", collection="knowledge", n=3)
        assert len(results) >= 1
        assert results[0].text == "Python async programming is great"
        assert results[0].distance == 0.1

    @pytest.mark.asyncio
    async def test_search_relevance_score(self, lt_memory):
        results = await lt_memory.search("test query", collection="knowledge")
        assert results[0].relevance_score > 0.9  # distance=0.1 → score≈0.95

    @pytest.mark.asyncio
    async def test_invalid_collection_raises(self, lt_memory):
        with pytest.raises(ValueError, match="Unknown collection"):
            await lt_memory.store("text", collection="invalid_collection")

    @pytest.mark.asyncio
    async def test_count(self, lt_memory):
        count = await lt_memory.count("knowledge")
        assert count == 1


# ─────────────────────────────────────────────────────────────────────────────
# EpisodicMemory tests (real SQLite in-memory)
# ─────────────────────────────────────────────────────────────────────────────


class TestEpisodicMemory:
    @pytest.fixture
    async def episodic(self, tmp_path):
        db_path = str(tmp_path / "test_episodes.db")
        mem = EpisodicMemory(db_path=db_path)
        await mem.init()
        yield mem
        await mem.close()

    @pytest.mark.asyncio
    async def test_start_and_finish_episode(self, episodic):
        episode_id = await episodic.start_episode("sess_1", "Research Python async")
        assert isinstance(episode_id, str)

        await episodic.finish_episode(
            episode_id,
            outcome="success",
            summary="Found that asyncio is the standard library",
            steps=["searched web", "read docs"],
            tool_count=2,
            turn_count=3,
        )

        episodes = await episodic.get_recent_episodes(n=1)
        assert len(episodes) == 1
        assert episodes[0].goal == "Research Python async"
        assert episodes[0].outcome == "success"
        assert episodes[0].tool_count == 2
        assert len(episodes[0].steps) == 2

    @pytest.mark.asyncio
    async def test_record_tool_call(self, episodic):
        record_id = await episodic.record_tool_call(
            session_id="sess_1",
            tool_name="web_search",
            arguments={"query": "python async"},
            result_content="Found results",
            duration_ms=150.5,
        )
        assert isinstance(record_id, str)

        calls = await episodic.get_recent_tool_calls(n=5)
        assert len(calls) == 1
        assert calls[0].tool_name == "web_search"
        assert calls[0].duration_ms == 150.5

    @pytest.mark.asyncio
    async def test_search_episodes(self, episodic):
        await episodic.start_episode("sess_1", "Research WebGPU adoption")
        await episodic.start_episode("sess_1", "Fix Python bug in main.py")

        results = await episodic.search_episodes("WebGPU")
        assert len(results) == 1
        assert "WebGPU" in results[0].goal

    @pytest.mark.asyncio
    async def test_add_and_get_reflections(self, episodic):
        await episodic.add_reflection(
            "sess_1",
            "Always check file permissions before writing",
            context="file_write failed with PermissionError",
        )
        reflections = await episodic.get_recent_reflections(n=5)
        assert len(reflections) == 1
        assert "file permissions" in reflections[0].content

    @pytest.mark.asyncio
    async def test_tool_call_stats(self, episodic):
        await episodic.record_tool_call("s1", "web_search", {"query": "test"})
        await episodic.record_tool_call("s1", "web_search", {"query": "test2"})
        await episodic.record_tool_call("s1", "file_read", {"path": "test.txt"})

        stats = await episodic.get_tool_call_stats()
        assert stats[0]["tool_name"] == "web_search"
        assert stats[0]["total"] == 2

    @pytest.mark.asyncio
    async def test_episode_duration(self, episodic):
        episode_id = await episodic.start_episode("sess_1", "Quick task")
        await asyncio.sleep(0.01)
        await episodic.finish_episode(episode_id, outcome="success")

        episodes = await episodic.get_recent_episodes(n=1)
        assert episodes[0].duration_seconds is not None
        assert episodes[0].duration_seconds >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMemoryManager:
    @pytest.fixture
    async def mm(self, tmp_path):
        manager = MemoryManager(
            chroma_persist_dir=str(tmp_path / "chroma"),
            sqlite_path=str(tmp_path / "episodes.db"),
        )
        # Skip embedder load and ChromaDB for unit tests
        manager._initialized = True
        manager.episodic._db = None

        # Mock long-term memory
        manager.long_term.store = AsyncMock(return_value="doc_id_123")
        manager.long_term.search = AsyncMock(return_value=[
            MemoryEntry(id="1", text="Relevant memory", collection="knowledge", distance=0.2)
        ])
        manager.long_term._client = MagicMock()

        # Use real SQLite for episodic (fast enough)
        manager.episodic = EpisodicMemory(db_path=str(tmp_path / "test.db"))
        await manager.episodic.init()

        yield manager
        await manager.episodic.close()

    def test_get_session_creates_new(self, mm):
        session = mm.get_session("new_session_1")
        assert session.state.session_id == "new_session_1"

    def test_get_session_returns_same(self, mm):
        s1 = mm.get_session("sess_abc")
        s2 = mm.get_session("sess_abc")
        assert s1 is s2

    def test_clear_session(self, mm):
        mm.get_session("sess_to_clear")
        mm.clear_session("sess_to_clear")
        # Getting it again creates a fresh one
        fresh = mm.get_session("sess_to_clear")
        assert fresh.state.turn_count == 0

    @pytest.mark.asyncio
    async def test_store_calls_long_term(self, mm):
        doc_id = await mm.store("Test text", collection="knowledge")
        assert doc_id == "doc_id_123"
        mm.long_term.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_returns_entries(self, mm):
        results = await mm.search("test query")
        assert len(results) == 1
        assert results[0].text == "Relevant memory"

    @pytest.mark.asyncio
    async def test_episode_lifecycle(self, mm):
        episode_id = await mm.start_episode("sess_1", "Test goal")
        assert isinstance(episode_id, str)

        await mm.commit_episode(
            episode_id,
            outcome="success",
            summary="Completed the test goal",
            tool_count=1,
        )

        episodes = await mm.get_recent_episodes(n=1)
        assert episodes[0].outcome == "success"

    @pytest.mark.asyncio
    async def test_requires_init(self):
        from exceptions import MemoryError as MemorySubsystemError
        mm = MemoryManager()
        with pytest.raises(MemorySubsystemError, match="not initialized"):
            await mm.store("text")

    @pytest.mark.asyncio
    async def test_record_tool_call(self, mm):
        record_id = await mm.record_tool_call(
            session_id="sess_1",
            tool_name="file_read",
            arguments={"path": "~/test.txt"},
            result="file contents here",
        )
        assert isinstance(record_id, str)

    @pytest.mark.asyncio
    async def test_build_memory_context(self, mm):
        context = await mm.build_memory_context("async python", "sess_1")
        # Returns empty string or a string with memory content
        assert isinstance(context, str)