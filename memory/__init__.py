"""
memory/__init__.py — NeuralClaw Memory System

Public interface for the memory module.

Usage:
    from memory import MemoryManager

    mm = MemoryManager.from_settings(settings)
    await mm.init()

    session = mm.get_session("session_123")
    await mm.store("Learned fact", collection="knowledge")
    results = await mm.search("related query")
"""

from memory.memory_manager import MemoryManager
from memory.short_term import ShortTermMemory, ConversationBuffer, SessionState
from memory.long_term import LongTermMemory, MemoryEntry, COLLECTIONS
from memory.episodic import EpisodicMemory, Episode, ToolCallRecord, Reflection
from memory.embedder import Embedder, EmbedderError

__all__ = [
    "MemoryManager",
    "ShortTermMemory",
    "ConversationBuffer",
    "SessionState",
    "LongTermMemory",
    "MemoryEntry",
    "COLLECTIONS",
    "EpisodicMemory",
    "Episode",
    "ToolCallRecord",
    "Reflection",
    "Embedder",
    "EmbedderError",
]