"""
memory/short_term.py — Short-Term Memory

In-process conversation buffer. Stores the last N turns of dialogue
and recent tool results. Lives for the duration of a session.

No persistence — cleared when the process restarts.
The MemoryManager handles saving summaries to long-term memory.

New in this version:
  - ConversationBuffer.compact(): replaces old messages with an LLM-generated
    summary, keeping only the most recent `keep_recent` turns intact.
    Returns the summary text so the caller can display/store it.
  - ShortTermMemory.compact(): async wrapper that calls the LLM to produce
    the summary, then calls ConversationBuffer.compact().
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
        # Set when compact() has been called; injected as a system message
        # before the kept recent turns so the LLM knows what happened earlier.
        self._compact_summary: Optional[str] = None

    def set_system_prompt(self, prompt: str) -> None:
        """Set the persistent system prompt (not counted in max_turns)."""
        self._system_prompt = prompt

    def add(self, message: Message) -> None:
        """Add a message to the buffer, evicting oldest turns if at capacity."""
        if message.role == Role.SYSTEM:
            self._system_prompt = message.content
            return

        self._messages.append(message)

        # Evict oldest turns until we're within budget.
        # Count by USER messages (one per turn) so that tool-result and
        # assistant-tool-call messages don't cause premature eviction.
        while True:
            user_count = sum(1 for m in self._messages if m.role == Role.USER)
            if user_count <= self.max_turns:
                break
            self._messages.pop(0)

        # Drop only leading orphaned plain-assistant messages (no tool_calls).
        # We must NOT drop Role.TOOL messages or Role.ASSISTANT messages that
        # carry tool_calls — those are paired with their tool result and removing
        # one half leaves the buffer in an invalid state that causes provider
        # API errors (OpenAI: "tool role must follow assistant with tool_calls").
        while (
            self._messages
            and self._messages[0].role == Role.ASSISTANT
            and not getattr(self._messages[0], "tool_calls", None)
        ):
            self._messages.pop(0)

    def add_user(self, content: str) -> None:
        self.add(Message.user(content))

    def add_assistant(self, content: str) -> None:
        self.add(Message.assistant(content))

    def get_messages(self, include_system: bool = True) -> list[Message]:
        """
        Return the full message list ready to send to the LLM.
        System prompt is prepended if set.
        If a compaction summary exists it is injected after the system prompt
        so the LLM knows what happened before the kept recent turns.
        """
        messages = []
        if include_system and self._system_prompt:
            messages.append(Message.system(self._system_prompt))
        if self._compact_summary:
            messages.append(Message.system(
                f"## Earlier Conversation Summary\n{self._compact_summary}\n"
                f"_(Older messages have been compacted. The recent turns below are exact.)_"
            ))
        messages.extend(self._messages)
        return messages

    def get_recent(self, n: int) -> list[Message]:
        """Return the last n messages (excluding system prompt)."""
        return self._messages[-n:] if n < len(self._messages) else list(self._messages)

    def compact(self, summary: str, keep_recent: int = 4) -> None:
        """
        Replace old messages with a summary, keeping `keep_recent` most recent turns.

        After compaction:
          - self._compact_summary holds the summary (injected by get_messages())
          - self._messages holds only the last `keep_recent` user+assistant pairs
          - Tool call/result messages from old turns are dropped (already in summary)

        Args:
            summary:     LLM-generated plain-text summary of the discarded messages.
            keep_recent: Number of most-recent USER turns to keep verbatim.
        """
        if not self._messages:
            return

        # Collect messages to keep: walk from the end, collect turns
        # A "turn" is a USER message plus everything up to the next USER message.
        kept: list[Message] = []
        user_turns_kept = 0
        for msg in reversed(self._messages):
            kept.insert(0, msg)
            if msg.role == Role.USER:
                user_turns_kept += 1
                if user_turns_kept >= keep_recent:
                    break

        discarded = len(self._messages) - len(kept)
        self._messages = kept
        self._compact_summary = summary

        log.info(
            "conversation.compacted",
            discarded_messages=discarded,
            kept_messages=len(kept),
            summary_chars=len(summary),
        )

    def clear(self) -> None:
        """Clear conversation history and compact summary (keeps system prompt)."""
        self._messages.clear()
        self._compact_summary = None

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
    def has_summary(self) -> bool:
        """True if this buffer has been compacted at least once."""
        return self._compact_summary is not None

    @property
    def turn_count(self) -> int:
        """Number of complete user+assistant turns in the buffer."""
        return sum(1 for m in self._messages if m.role == Role.USER)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"<ConversationBuffer turns={self.turn_count} messages={self.message_count}>"


class ToolResultCache:
    """Cache of recent tool results for short-term recall."""

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

    async def compact(
        self,
        llm_client,          # BaseLLMClient
        llm_config,          # LLMConfig
        keep_recent: int = 4,
    ) -> str:
        """
        Summarise and compact the conversation buffer.

        Asks the LLM to write a concise summary of all messages currently in
        the buffer, then replaces all but the last `keep_recent` user turns
        with that summary.

        Returns:
            The summary text (so the caller can display it to the user).

        Raises:
            RuntimeError if there's nothing to compact (fewer than keep_recent+1 turns).
        """
        from brain.types import LLMConfig

        current_turns = self.conversation.turn_count
        if current_turns <= keep_recent:
            raise RuntimeError(
                f"Nothing to compact — only {current_turns} turns in buffer "
                f"(need more than {keep_recent} to compact)."
            )

        # Build a summary request using the raw conversation text.
        # We intentionally don't include the system prompt here — it's about
        # the conversation content, not the agent's instructions.
        conversation_text = self.conversation.to_text()
        if not conversation_text.strip():
            raise RuntimeError("Conversation buffer is empty.")

        from brain.types import Message as BrainMessage

        summary_messages = [
            BrainMessage.system(
                "You are a precise summarisation assistant. "
                "Your job is to summarise a conversation accurately and concisely."
            ),
            BrainMessage.user(
                f"Summarise the following conversation in 3–6 sentences. "
                f"Capture the key topics discussed, decisions made, and important facts. "
                f"Write in third person (e.g. 'The user asked about...', 'The agent explained...'). "
                f"Do not add opinions or commentary.\n\n"
                f"CONVERSATION:\n{conversation_text}"
            ),
        ]

        # Use a lower-cost config for the summary call
        from brain.types import LLMConfig as LC
        summary_config = LC(
            model=llm_config.model,
            temperature=0.2,           # deterministic summary
            max_tokens=400,
        )

        response = await llm_client.generate(
            messages=summary_messages,
            config=summary_config,
            tools=None,
        )
        summary = (response.content or "").strip()

        if not summary:
            raise RuntimeError("LLM returned an empty summary.")

        # Apply compaction to the buffer
        self.conversation.compact(summary=summary, keep_recent=keep_recent)

        log.info(
            "short_term.compacted",
            session_id=self.state.session_id,
            turns_before=current_turns,
            turns_kept=keep_recent,
            summary_chars=len(summary),
        )
        return summary

    def clear_conversation(self) -> None:
        self.conversation.clear()
        log.info("short_term.cleared", session_id=self.state.session_id)

    def __repr__(self) -> str:
        return (
            f"<ShortTermMemory session={self.state.session_id} "
            f"turns={self.state.turn_count}>"
        )