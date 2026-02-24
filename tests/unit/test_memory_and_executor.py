"""
tests/unit/test_memory_and_executor.py

Tests for:
  - memory/short_term.py    (ConversationBuffer, ToolResultCache, ShortTermMemory)
  - memory/long_term.py     (MemoryEntry, LongTermMemory logic)
  - memory/memory_manager.py (MemoryManager)
  - agent/executor.py       (Executor, _fire_and_forget)
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call


# ─────────────────────────────────────────────────────────────────────────────
# ConversationBuffer
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationBuffer:
    def _make_buf(self, max_turns: int = 5):
        from memory.short_term import ConversationBuffer
        return ConversationBuffer(max_turns=max_turns)

    def _user(self, text: str):
        from brain.types import Message
        return Message.user(text)

    def _assistant(self, text: str):
        from brain.types import Message
        return Message.assistant(text)

    def _system(self, text: str):
        from brain.types import Message
        return Message.system(text)

    def test_add_user_increments_turn_count(self):
        buf = self._make_buf()
        buf.add(self._user("hello"))
        assert buf.turn_count == 1

    def test_system_message_stored_as_prompt_not_in_buffer(self):
        buf = self._make_buf()
        buf.add(self._system("You are an agent"))
        assert buf.message_count == 0
        assert buf._system_prompt == "You are an agent"

    def test_get_messages_includes_system(self):
        buf = self._make_buf()
        buf.set_system_prompt("sys")
        buf.add(self._user("hi"))
        msgs = buf.get_messages(include_system=True)
        assert msgs[0].content == "sys"
        assert len(msgs) == 2

    def test_get_messages_excludes_system_when_asked(self):
        buf = self._make_buf()
        buf.set_system_prompt("sys")
        buf.add(self._user("hi"))
        msgs = buf.get_messages(include_system=False)
        assert all(m.content != "sys" for m in msgs)

    def test_evicts_oldest_when_over_capacity(self):
        buf = self._make_buf(max_turns=2)
        buf.add(self._user("turn 1"))
        buf.add(self._assistant("reply 1"))
        buf.add(self._user("turn 2"))
        buf.add(self._assistant("reply 2"))
        buf.add(self._user("turn 3"))  # should evict turn 1
        assert buf.turn_count == 2
        contents = [m.content for m in buf._messages]
        assert "turn 1" not in contents

    def test_buffer_starts_with_user_after_eviction(self):
        buf = self._make_buf(max_turns=2)
        buf.add(self._assistant("orphan"))
        buf.add(self._user("user 1"))
        buf.add(self._assistant("reply 1"))
        buf.add(self._user("user 2"))
        buf.add(self._assistant("reply 2"))
        buf.add(self._user("user 3"))
        from brain.types import Role
        assert buf._messages[0].role == Role.USER

    def test_get_recent_returns_last_n(self):
        buf = self._make_buf()
        for i in range(5):
            buf.add(self._user(f"msg {i}"))
        recent = buf.get_recent(3)
        assert len(recent) == 3
        assert recent[-1].content == "msg 4"

    def test_get_recent_all_when_n_exceeds_size(self):
        buf = self._make_buf()
        buf.add(self._user("only"))
        assert len(buf.get_recent(100)) == 1

    def test_clear_removes_messages_and_summary(self):
        buf = self._make_buf()
        buf.add(self._user("hi"))
        buf._compact_summary = "some summary"
        buf.clear()
        assert buf.message_count == 0
        assert buf._compact_summary is None

    def test_clear_preserves_system_prompt(self):
        buf = self._make_buf()
        buf.set_system_prompt("keep me")
        buf.add(self._user("hi"))
        buf.clear()
        assert buf._system_prompt == "keep me"

    def test_has_summary_false_initially(self):
        buf = self._make_buf()
        assert buf.has_summary is False

    def test_compact_sets_summary_and_trims(self):
        buf = self._make_buf()
        for i in range(6):
            buf.add(self._user(f"u{i}"))
            buf.add(self._assistant(f"a{i}"))
        buf.compact("Summary text", keep_recent=2)
        assert buf.has_summary is True
        assert buf._compact_summary == "Summary text"
        assert buf.turn_count == 2

    def test_compact_on_empty_does_nothing(self):
        buf = self._make_buf()
        buf.compact("summary", keep_recent=2)
        assert buf.has_summary is False

    def test_get_messages_includes_compact_summary(self):
        buf = self._make_buf()
        buf._compact_summary = "Earlier stuff happened"
        buf.add(self._user("recent"))
        msgs = buf.get_messages()
        contents = " ".join(m.content or "" for m in msgs)
        assert "Earlier stuff happened" in contents

    def test_to_text_renders_messages(self):
        buf = self._make_buf()
        buf.add(self._user("hello"))
        buf.add(self._assistant("world"))
        text = buf.to_text()
        assert "USER: hello" in text
        assert "ASSISTANT: world" in text

    def test_len_returns_message_count(self):
        buf = self._make_buf()
        buf.add(self._user("a"))
        buf.add(self._user("b"))
        assert len(buf) == 2

    def test_repr(self):
        buf = self._make_buf()
        buf.add(self._user("hi"))
        r = repr(buf)
        assert "ConversationBuffer" in r


# ─────────────────────────────────────────────────────────────────────────────
# ToolResultCache
# ─────────────────────────────────────────────────────────────────────────────

class TestToolResultCache:
    def _make_cache(self, max_results: int = 5):
        from memory.short_term import ToolResultCache
        return ToolResultCache(max_results=max_results)

    def _entry(self, tool_call_id: str, name: str = "tool", content: str = "ok"):
        from memory.short_term import ToolResultEntry
        return ToolResultEntry(tool_call_id=tool_call_id, tool_name=name, content=content)

    def test_add_and_get_recent(self):
        cache = self._make_cache()
        cache.add(self._entry("id1"))
        results = cache.get_recent(5)
        assert len(results) == 1

    def test_respects_max_results(self):
        cache = self._make_cache(max_results=3)
        for i in range(5):
            cache.add(self._entry(f"id{i}"))
        assert len(cache) == 3

    def test_get_by_id_found(self):
        cache = self._make_cache()
        cache.add(self._entry("abc", content="result"))
        entry = cache.get_by_id("abc")
        assert entry is not None
        assert entry.content == "result"

    def test_get_by_id_not_found_returns_none(self):
        cache = self._make_cache()
        assert cache.get_by_id("nonexistent") is None

    def test_get_recent_n_less_than_size(self):
        cache = self._make_cache()
        for i in range(5):
            cache.add(self._entry(f"id{i}"))
        results = cache.get_recent(2)
        assert len(results) == 2

    def test_clear_empties_cache(self):
        cache = self._make_cache()
        cache.add(self._entry("id1"))
        cache.clear()
        assert len(cache) == 0


# ─────────────────────────────────────────────────────────────────────────────
# ShortTermMemory
# ─────────────────────────────────────────────────────────────────────────────

class TestShortTermMemory:
    def _make_stm(self, session_id: str = "sess-1"):
        from memory.short_term import ShortTermMemory
        return ShortTermMemory(session_id=session_id, user_id="user1", max_turns=10)

    def test_add_user_message_increments_state_turn_count(self):
        stm = self._make_stm()
        from brain.types import Message
        stm.add_message(Message.user("hello"))
        assert stm.state.turn_count == 1

    def test_add_assistant_message_no_turn_increment(self):
        stm = self._make_stm()
        from brain.types import Message
        stm.add_message(Message.assistant("hi"))
        assert stm.state.turn_count == 0

    def test_add_tool_result(self):
        stm = self._make_stm()
        stm.add_tool_result("call-1", "web_search", "results here")
        assert len(stm.tool_results) == 1

    def test_add_tool_result_error_flag(self):
        stm = self._make_stm()
        stm.add_tool_result("call-1", "terminal", "error msg", is_error=True)
        entry = stm.tool_results.get_by_id("call-1")
        assert entry.is_error is True

    def test_get_context_messages_with_system_prompt(self):
        stm = self._make_stm()
        from brain.types import Message
        stm.add_message(Message.user("hello"))
        msgs = stm.get_context_messages(system_prompt="You are helpful")
        assert any("helpful" in (m.content or "") for m in msgs)

    def test_clear_conversation(self):
        stm = self._make_stm()
        from brain.types import Message
        stm.add_message(Message.user("hello"))
        stm.clear_conversation()
        assert stm.conversation.message_count == 0

    def test_repr(self):
        stm = self._make_stm("my-session")
        r = repr(stm)
        assert "my-session" in r

    @pytest.mark.asyncio
    async def test_compact_raises_when_not_enough_turns(self):
        stm = self._make_stm()
        from brain.types import Message
        stm.add_message(Message.user("only one"))
        with pytest.raises(RuntimeError, match="Nothing to compact"):
            await stm.compact(AsyncMock(), MagicMock(), keep_recent=4)

    @pytest.mark.asyncio
    async def test_compact_calls_llm_and_applies_summary(self):
        stm = self._make_stm()
        from brain.types import Message
        for i in range(6):
            stm.add_message(Message.user(f"msg {i}"))
            stm.add_message(Message.assistant(f"reply {i}"))

        llm = AsyncMock()
        resp = MagicMock()
        resp.content = "A nice summary"
        llm.generate = AsyncMock(return_value=resp)
        config = MagicMock()
        config.model = "test-model"

        summary = await stm.compact(llm, config, keep_recent=2)
        assert summary == "A nice summary"
        assert stm.conversation.has_summary is True

    @pytest.mark.asyncio
    async def test_compact_raises_on_empty_llm_response(self):
        stm = self._make_stm()
        from brain.types import Message
        for i in range(6):
            stm.add_message(Message.user(f"msg {i}"))

        llm = AsyncMock()
        resp = MagicMock()
        resp.content = ""
        llm.generate = AsyncMock(return_value=resp)
        config = MagicMock()
        config.model = "test-model"

        with pytest.raises(RuntimeError, match="empty summary"):
            await stm.compact(llm, config, keep_recent=2)


# ─────────────────────────────────────────────────────────────────────────────
# MemoryEntry
# ─────────────────────────────────────────────────────────────────────────────

class TestMemoryEntry:
    def test_relevance_score_from_distance(self):
        from memory.long_term import MemoryEntry
        entry = MemoryEntry(id="1", text="hello", collection="knowledge", distance=0.0)
        assert entry.relevance_score == 1.0

    def test_relevance_score_max_distance(self):
        from memory.long_term import MemoryEntry
        entry = MemoryEntry(id="1", text="hello", collection="knowledge", distance=2.0)
        assert entry.relevance_score == 0.0

    def test_relevance_score_mid_distance(self):
        from memory.long_term import MemoryEntry
        entry = MemoryEntry(id="1", text="hello", collection="knowledge", distance=1.0)
        assert entry.relevance_score == 0.5

    def test_relevance_score_clamped_at_zero(self):
        from memory.long_term import MemoryEntry
        entry = MemoryEntry(id="1", text="hello", collection="knowledge", distance=3.0)
        assert entry.relevance_score == 0.0

    def test_no_distance_gives_zero_relevance(self):
        from memory.long_term import MemoryEntry
        entry = MemoryEntry(id="1", text="hello", collection="knowledge")
        assert entry.relevance_score == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# LongTermMemory (with mocked ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

def _make_lt_memory(relevance_threshold: float = 0.5):
    from memory.long_term import LongTermMemory
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
    lt = LongTermMemory(persist_dir="/tmp/test_chroma", embedder=embedder, relevance_threshold=relevance_threshold)
    # Pre-inject a fake client and collections so we don't need real ChromaDB
    lt._client = MagicMock()
    return lt


def _make_chroma_collection(docs=None, distances=None):
    col = MagicMock()
    col.count = MagicMock(return_value=len(docs or []))
    col.query = MagicMock(return_value={
        "ids": [["doc1"]],
        "documents": [docs or ["text"]],
        "metadatas": [[{"key": "val"}]],
        "distances": [distances or [0.4]],
    })
    col.upsert = MagicMock()
    col.delete = MagicMock()
    return col


class TestLongTermMemory:
    @pytest.mark.asyncio
    async def test_store_calls_chroma_add(self):
        lt = _make_lt_memory()
        col = _make_chroma_collection()
        lt._collections = {"knowledge": col}
        doc_id = await lt.store("some text", collection="knowledge")
        assert col.upsert.called
        assert isinstance(doc_id, str)

    @pytest.mark.asyncio
    async def test_store_uses_provided_doc_id(self):
        lt = _make_lt_memory()
        col = _make_chroma_collection()
        lt._collections = {"knowledge": col}
        doc_id = await lt.store("text", doc_id="my-id")
        assert doc_id == "my-id"

    @pytest.mark.asyncio
    async def test_store_unknown_collection_raises(self):
        lt = _make_lt_memory()
        lt._collections = {}
        with pytest.raises(ValueError, match="Unknown collection"):
            await lt.store("text", collection="nonexistent")

    @pytest.mark.asyncio
    async def test_search_returns_entries_above_threshold(self):
        lt = _make_lt_memory(relevance_threshold=0.5)
        col = _make_chroma_collection(docs=["result text"], distances=[0.4])  # score = 0.8
        lt._collections = {"knowledge": col}
        entries = await lt.search("query", collection="knowledge")
        assert len(entries) == 1
        assert entries[0].text == "result text"

    @pytest.mark.asyncio
    async def test_search_filters_below_threshold(self):
        lt = _make_lt_memory(relevance_threshold=0.9)  # very high threshold
        col = _make_chroma_collection(docs=["result"], distances=[1.5])  # score = 0.25
        lt._collections = {"knowledge": col}
        entries = await lt.search("query", collection="knowledge")
        assert len(entries) == 0

    @pytest.mark.asyncio
    async def test_search_empty_collection_returns_empty(self):
        lt = _make_lt_memory()
        col = MagicMock()
        col.count = MagicMock(return_value=0)
        lt._collections = {"knowledge": col}
        entries = await lt.search("query", collection="knowledge")
        assert entries == []

    @pytest.mark.asyncio
    async def test_search_unknown_collection_returns_empty(self):
        lt = _make_lt_memory()
        lt._collections = {}
        entries = await lt.search("query", collection="knowledge")
        assert entries == []

    @pytest.mark.asyncio
    async def test_search_results_sorted_by_relevance(self):
        lt = _make_lt_memory(relevance_threshold=0.0)
        col = MagicMock()
        col.count = MagicMock(return_value=2)
        col.query = MagicMock(return_value={
            "ids": [["id1", "id2"]],
            "documents": [["less relevant", "more relevant"]],
            "metadatas": [[{}, {}]],
            "distances": [[1.2, 0.2]],  # scores: 0.4 and 0.9
        })
        lt._collections = {"knowledge": col}
        entries = await lt.search("query", collection="knowledge")
        assert entries[0].relevance_score > entries[1].relevance_score

    @pytest.mark.asyncio
    async def test_delete_calls_chroma_delete(self):
        lt = _make_lt_memory()
        col = MagicMock()
        lt._collections = {"knowledge": col}
        await lt.delete("doc-123", collection="knowledge")
        col.delete.assert_called_once_with(ids=["doc-123"])

    @pytest.mark.asyncio
    async def test_delete_unknown_collection_silently_passes(self):
        lt = _make_lt_memory()
        lt._collections = {}
        # Should not raise
        await lt.delete("doc-123", collection="knowledge")

    @pytest.mark.asyncio
    async def test_count_returns_zero_for_unknown_collection(self):
        lt = _make_lt_memory()
        lt._collections = {}
        assert await lt.count("knowledge") == 0

    @pytest.mark.asyncio
    async def test_count_returns_collection_count(self):
        lt = _make_lt_memory()
        col = MagicMock()
        col.count = MagicMock(return_value=42)
        lt._collections = {"knowledge": col}
        assert await lt.count("knowledge") == 42

    @pytest.mark.asyncio
    async def test_search_all_aggregates_across_collections(self):
        lt = _make_lt_memory(relevance_threshold=0.0)
        # Give each collection one result
        for name in ["knowledge", "conversations", "reflections", "user_prefs"]:
            col = MagicMock()
            col.count = MagicMock(return_value=1)
            col.query = MagicMock(return_value={
                "ids": [["id1"]],
                "documents": [["text"]],
                "metadatas": [[{}]],
                "distances": [[0.2]],
            })
            lt._collections[name] = col
        results = await lt.search_all("query")
        assert len(results) == 4


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

def _make_memory_manager():
    from memory.memory_manager import MemoryManager
    from memory.task import TaskMemoryStore
    mm = MemoryManager.__new__(MemoryManager)
    mm._initialized = True
    mm._relevance_threshold = 0.55
    mm._max_short_term_turns = 20
    mm._sessions = {}
    mm._task_store = TaskMemoryStore()  # required after hardening
    mm.embedder = MagicMock()
    mm.embedder.close = MagicMock()
    mm.long_term = AsyncMock()
    mm.episodic = AsyncMock()
    return mm


class TestMemoryManagerInit:
    @pytest.mark.asyncio
    async def test_init_idempotent(self):
        mm = _make_memory_manager()
        mm._initialized = True
        mm.embedder.load = AsyncMock()
        mm.long_term.init = AsyncMock()
        mm.episodic.init = AsyncMock()
        await mm.init()
        # init() should return immediately when already initialized
        mm.long_term.init.assert_not_called()

    @pytest.mark.asyncio
    async def test_requires_init_raises_when_not_initialized(self):
        mm = _make_memory_manager()
        mm._initialized = False
        from exceptions import MemoryError as MemorySubsystemError
        with pytest.raises(MemorySubsystemError, match="not initialized"):
            mm._require_init()


class TestMemoryManagerSession:
    def test_get_session_creates_new(self):
        mm = _make_memory_manager()
        stm = mm.get_session("sess-1", "user1")
        assert stm is not None
        assert "sess-1" in mm._sessions

    def test_get_session_returns_same_object(self):
        mm = _make_memory_manager()
        stm1 = mm.get_session("sess-1")
        stm2 = mm.get_session("sess-1")
        assert stm1 is stm2

    def test_clear_session_removes_from_dict(self):
        mm = _make_memory_manager()
        mm.get_session("sess-1")
        mm.clear_session("sess-1")
        assert "sess-1" not in mm._sessions

    def test_clear_nonexistent_session_is_safe(self):
        mm = _make_memory_manager()
        mm.clear_session("no-such-session")  # should not raise


class TestMemoryManagerOperations:
    @pytest.mark.asyncio
    async def test_store_delegates_to_long_term(self):
        mm = _make_memory_manager()
        mm.long_term.store = AsyncMock(return_value="doc-id")
        result = await mm.store("text", collection="knowledge")
        assert result == "doc-id"
        mm.long_term.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_delegates_to_long_term(self):
        mm = _make_memory_manager()
        mm.long_term.search = AsyncMock(return_value=[])
        result = await mm.search("query")
        assert result == []

    @pytest.mark.asyncio
    async def test_store_conversation_summary(self):
        mm = _make_memory_manager()
        mm.long_term.store = AsyncMock(return_value="doc-id")
        result = await mm.store_conversation_summary("sess-1", "summary text", turn_count=5)
        assert result == "doc-id"
        all_args = mm.long_term.store.call_args
        pos_args, kw_args = all_args[0], all_args[1]
        collection = kw_args.get("collection") or (pos_args[1] if len(pos_args) > 1 else None)
        metadata = kw_args.get("metadata") or (pos_args[2] if len(pos_args) > 2 else None)
        assert collection == "conversations"
        assert metadata["session_id"] == "sess-1"

    @pytest.mark.asyncio
    async def test_build_memory_context_returns_empty_when_no_results(self):
        mm = _make_memory_manager()
        mm.long_term.search = AsyncMock(return_value=[])
        result = await mm.build_memory_context("query", "sess-1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_build_memory_context_returns_formatted_block(self):
        from memory.long_term import MemoryEntry
        mm = _make_memory_manager()
        entry = MemoryEntry(id="1", text="important fact", collection="knowledge", distance=0.1)
        mm.long_term.search = AsyncMock(return_value=[entry])
        result = await mm.build_memory_context("query", "sess-1")
        assert "important fact" in result

    @pytest.mark.asyncio
    async def test_search_all_skips_failed_collections(self):
        from memory.long_term import COLLECTIONS
        mm = _make_memory_manager()
        # First collection raises, rest return empty
        call_count = [0]
        async def _search(query, col, n):
            call_count[0] += 1
            if call_count[0] == 1:
                raise ValueError("collection failed")
            return []
        mm.long_term.search = _search
        result = await mm.search_all("query")
        # Should not raise and should return empty dict (all empty)
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_commit_episode_also_stores_summary(self):
        mm = _make_memory_manager()
        mm.long_term.store = AsyncMock(return_value="doc-id")
        mm.episodic.finish_episode = AsyncMock()
        await mm.commit_episode("ep-1", outcome="success", summary="It worked")
        mm.long_term.store.assert_called_once()
        mm.episodic.finish_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_commit_episode_no_summary_skips_store(self):
        mm = _make_memory_manager()
        mm.long_term.store = AsyncMock()
        mm.episodic.finish_episode = AsyncMock()
        await mm.commit_episode("ep-1", outcome="success", summary=None)
        mm.long_term.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_resets_initialized_flag(self):
        mm = _make_memory_manager()
        mm.long_term.close = AsyncMock()
        mm.episodic.close = AsyncMock()
        await mm.close()
        assert mm._initialized is False


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

def _make_btc(name: str = "web_search", args: dict = None, call_id: str = "call-1"):
    from brain.types import ToolCall
    tc = MagicMock(spec=ToolCall)
    tc.name = name
    tc.id = call_id
    tc.arguments = args or {"query": "test"}
    return tc


def _make_tool_result(is_error: bool = False):
    from skills.types import SkillResult, RiskLevel
    if is_error:
        return SkillResult.fail(
            skill_name="web_search",
            skill_call_id="call-1",
            error="tool error",
            error_type="TestError",
        )
    return SkillResult.ok(
        skill_name="web_search",
        skill_call_id="call-1",
        output="tool output",
        duration_ms=42.0,
    )


def _make_session(trust_val: str = "low", active_plan=None):
    from skills.types import TrustLevel
    session = MagicMock()
    session.id = "sess-test"
    session.trust_level = TrustLevel(trust_val) if trust_val in ("low", "medium", "high") else MagicMock(value=trust_val)
    session.active_plan = active_plan
    session.granted_capabilities = frozenset()
    session.record_tool_call = MagicMock()
    return session


def _make_executor(bus_result=None, reasoner_proceed=True):
    from agent.executor import Executor
    registry = MagicMock()
    registry.get_schema = MagicMock(return_value=None)  # no schema = skip reasoner

    bus = AsyncMock()
    bus.dispatch = AsyncMock(return_value=bus_result or _make_tool_result())

    reasoner = AsyncMock()
    verdict = MagicMock()
    verdict.proceed = reasoner_proceed
    verdict.concern = "blocked reason"
    verdict.reasoning = "reasoning"
    reasoner.evaluate_tool_call = AsyncMock(return_value=verdict)

    memory = AsyncMock()
    memory.record_tool_call = AsyncMock(return_value="rec-id")

    return Executor(registry, bus, reasoner, memory), registry, bus, reasoner, memory


class TestExecutorDispatch:
    @pytest.mark.asyncio
    async def test_basic_dispatch_returns_result(self):
        executor, _, bus, _, _ = _make_executor()
        session = _make_session()
        result = await executor.dispatch(_make_btc(), session)
        assert result is not None
        bus.dispatch.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_tool_call_on_session(self):
        executor, _, _, _, _ = _make_executor()
        session = _make_session()
        await executor.dispatch(_make_btc(), session)
        session.record_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoner_not_called_for_low_risk(self):
        executor, registry, _, reasoner, _ = _make_executor()
        # schema with LOW risk
        from skills.types import RiskLevel
        schema = MagicMock()
        schema.risk_level = RiskLevel.LOW
        registry.get_schema = MagicMock(return_value=schema)
        session = _make_session()
        await executor.dispatch(_make_btc(), session)
        reasoner.evaluate_tool_call.assert_not_called()

    @pytest.mark.asyncio
    async def test_reasoner_called_for_high_risk(self):
        executor, registry, _, reasoner, _ = _make_executor(reasoner_proceed=True)
        from skills.types import RiskLevel
        schema = MagicMock()
        schema.risk_level = RiskLevel.HIGH
        registry.get_schema = MagicMock(return_value=schema)
        session = _make_session()
        await executor.dispatch(_make_btc(), session)
        reasoner.evaluate_tool_call.assert_called_once()

    @pytest.mark.asyncio
    async def test_reasoner_block_returns_error_result(self):
        executor, registry, bus, _, _ = _make_executor(reasoner_proceed=False)
        from skills.types import RiskLevel
        schema = MagicMock()
        schema.risk_level = RiskLevel.HIGH
        registry.get_schema = MagicMock(return_value=schema)
        session = _make_session()
        result = await executor.dispatch(_make_btc(), session)
        bus.dispatch.assert_not_called()
        assert result.is_error

    @pytest.mark.asyncio
    async def test_invalid_trust_level_falls_back_to_low(self):
        executor, _, bus, _, _ = _make_executor()
        session = _make_session(trust_val="INVALID_TRUST")
        # Should not raise — falls back to LOW
        await executor.dispatch(_make_btc(), session)
        assert bus.dispatch.called

    @pytest.mark.asyncio
    async def test_reasoner_uses_plan_goal_when_available(self):
        executor, registry, _, reasoner, _ = _make_executor(reasoner_proceed=True)
        from skills.types import RiskLevel
        schema = MagicMock()
        schema.risk_level = RiskLevel.HIGH
        registry.get_schema = MagicMock(return_value=schema)

        plan = MagicMock()
        plan.goal = "World domination"
        plan.current_step = MagicMock()
        plan.current_step.description = "Step one"
        session = _make_session(active_plan=plan)
        await executor.dispatch(_make_btc(), session)
        call_kwargs = reasoner.evaluate_tool_call.call_args[1]
        assert call_kwargs["goal"] == "World domination"

    @pytest.mark.asyncio
    async def test_on_response_callback_called_on_confirmation(self):
        """Confirmation path: callback should be invoked with confirmation request."""
        from agent.executor import Executor
        from skills.types import SafetyDecision, SafetyStatus, RiskLevel

        registry = MagicMock()
        registry.get_schema = MagicMock(return_value=None)

        # Bus triggers on_confirm_needed via the native dispatch signature
        async def _bus_dispatch(skill_call, trust_level, on_confirm_needed=None, granted_capabilities=frozenset()):
            if on_confirm_needed is not None:
                from skills.types import ConfirmationRequest, RiskLevel
                confirm_req = ConfirmationRequest(
                    skill_name="terminal",
                    skill_call_id="call-123",
                    risk_level=RiskLevel.HIGH,
                    reason="risky",
                    arguments={},
                )
                with patch("agent.executor.asyncio.wait_for", return_value=True):
                    await on_confirm_needed(confirm_req)
            return _make_tool_result()

        bus = AsyncMock()
        bus.dispatch = _bus_dispatch

        memory = AsyncMock()
        memory.record_tool_call = AsyncMock(return_value="x")
        reasoner = AsyncMock()
        executor = Executor(registry, bus, reasoner, memory)

        on_response = MagicMock()
        session = _make_session()

        # Resolve the future immediately so the wait_for doesn't hang
        fut = asyncio.get_event_loop().create_future()
        fut.set_result(True)
        session.register_confirmation = MagicMock(return_value=fut)
        session.cancel_confirmation = MagicMock()

        result = await executor.dispatch(_make_btc(), session, on_response=on_response)
        assert on_response.called


class TestFireAndForget:
    @pytest.mark.asyncio
    async def test_creates_task(self):
        from agent.utils import fire_and_forget as _fire_and_forget

        async def _noop():
            pass

        task = _fire_and_forget(_noop())
        assert isinstance(task, asyncio.Task)
        await task  # let it finish

    @pytest.mark.asyncio
    async def test_logs_exception_on_failure(self):
        from agent.utils import fire_and_forget as _fire_and_forget

        async def _fail():
            raise RuntimeError("oops")

        with patch("agent.utils.log") as mock_log:
            task = _fire_and_forget(_fail(), label="test_task")
            await asyncio.sleep(0.01)  # allow task to complete
            mock_log.warning.assert_called()