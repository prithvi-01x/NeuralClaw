"""
memory/short_term.py — Short-Term Memory

In-process conversation buffer. Stores the last N turns of dialogue
and recent tool results. Lives for the duration of a session.

No persistence — cleared when the process restarts.
The MemoryManager handles saving summaries to long-term memory.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from brain.types import Message, Role
from observability.logger import get_logger

log = get_logger(__name__)


@dataclass
class ToolResultEntry:
    """A cached tool result in short-term memory."""
    tool_call_id: str
    tool_name: str
    content: str
    timestamp: float = field(default_factory=time.time)
    is_error: bool = False


@dataclass
class SessionState:
    """Mutable state for the current agent session."""
    session_id: str
    user_id: str = "local"
    active_goal: Optional[str] = None
    active_plan: Optional[dict[str, Any]] = None
    trust_level: str = "low"
    turn_count: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationBuffer:
    """
    Circular buffer of the last N conversation messages.

    Automatically drops oldest messages when capacity is reached.
    System messages are always preserved.
    """

    def __init__(self, max_turns: int = 20):
        self.max_turns = max_turns
        self._messages: list[Message] = []
        self._system_prompt: Optional[str] = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set the persistent system prompt (not counted in max_turns)."""
        self._system_prompt = prompt

    def add(self, message: Message) -> None:
        """Add a message to the buffer, evicting oldest if at capacity."""
        if message.role == Role.SYSTEM:
            self._system_prompt = message.content
            return

        self._messages.append(message)

        # Evict oldest non-system messages if over capacity
        # Count pairs: each user+assistant = 1 turn
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)

    def add_user(self, content: str) -> None:
        self.add(Message.user(content))

    def add_assistant(self, content: str) -> None:
        self.add(Message.assistant(content))

    def get_messages(self, include_system: bool = True) -> list[Message]:
        """
        Return the full message list ready to send to the LLM.
        System prompt is prepended if set.
        """
        messages = []
        if include_system and self._system_prompt:
            messages.append(Message.system(self._system_prompt))
        messages.extend(self._messages)
        return messages

    def get_recent(self, n: int) -> list[Message]:
        """Return the last n messages (excluding system prompt)."""
        return self._messages[-n:] if n < len(self._messages) else list(self._messages)

    def clear(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        self._messages.clear()

    def to_text(self) -> str:
        """Render the conversation as plain text (for summarisation)."""
        lines = []
        for msg in self._messages:
            role = msg.role.value.upper()
            content = msg.content or ""
            if msg.tool_calls:
                content += f" [called tools: {', '.join(tc.name for tc in msg.tool_calls)}]"
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @property
    def turn_count(self) -> int:
        """Number of complete user+assistant turns."""
        user_msgs = sum(1 for m in self._messages if m.role == Role.USER)
        return user_msgs

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"<ConversationBuffer turns={self.turn_count} messages={self.message_count}>"


class ToolResultCache:
    """
    Cache of recent tool results for short-term recall.
    The LLM can reference recent tool outputs without re-running them.
    """

    def __init__(self, max_results: int = 10):
        self._cache: deque[ToolResultEntry] = deque(maxlen=max_results)

    def add(self, entry: ToolResultEntry) -> None:
        self._cache.append(entry)

    def get_recent(self, n: int = 5) -> list[ToolResultEntry]:
        results = list(self._cache)
        return results[-n:] if n < len(results) else results

    def get_by_id(self, tool_call_id: str) -> Optional[ToolResultEntry]:
        for entry in self._cache:
            if entry.tool_call_id == tool_call_id:
                return entry
        return None

    def clear(self) -> None:
        self._cache.clear()

    def __len__(self) -> int:
        return len(self._cache)


class ShortTermMemory:
    """
    Unified short-term memory for a session.

    Contains:
    - ConversationBuffer: recent messages
    - ToolResultCache: recent tool outputs
    - SessionState: current session metadata
    """

    def __init__(
        self,
        session_id: str,
        user_id: str = "local",
        max_turns: int = 20,
        max_tool_results: int = 10,
    ):
        self.state = SessionState(session_id=session_id, user_id=user_id)
        self.conversation = ConversationBuffer(max_turns=max_turns)
        self.tool_results = ToolResultCache(max_results=max_tool_results)

    def add_message(self, message: Message) -> None:
        self.conversation.add(message)
        if message.role == Role.USER:
            self.state.turn_count += 1

    def add_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        content: str,
        is_error: bool = False,
    ) -> None:
        self.tool_results.add(ToolResultEntry(
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            content=content,
            is_error=is_error,
        ))

    def get_context_messages(self, system_prompt: Optional[str] = None) -> list[Message]:
        """Return messages ready to send to the LLM."""
        if system_prompt:
            self.conversation.set_system_prompt(system_prompt)
        return self.conversation.get_messages(include_system=True)

    def clear_conversation(self) -> None:
        self.conversation.clear()
        log.info("short_term.cleared", session_id=self.state.session_id)

    def __repr__(self) -> str:
        return (
            f"<ShortTermMemory session={self.state.session_id} "
            f"turns={self.state.turn_count}>"
        )