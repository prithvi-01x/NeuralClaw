"""
agent/context_builder.py — LLM Context Builder

Assembles the full message list sent to the LLM each turn:
    System prompt → Long-term memory block → Conversation history → User message

Handles token budget awareness via character-count approximation.
"""

from __future__ import annotations

import time
from typing import Optional

from neuralclaw.exceptions import MemoryError as NeuralClawMemoryError, MemoryStoreError
from neuralclaw.brain.types import Message, Role, ToolSchema
from neuralclaw.memory.memory_manager import MemoryManager
from neuralclaw.observability.logger import get_logger
from neuralclaw.agent.workspace import get_workspace_loader

log = get_logger(__name__)

_HISTORY_MAX_CHARS = 20_000
_MEMORY_MAX_CHARS = 6_000

_TRUST_DESCRIPTIONS = {
    "low": (
        "LOW and MEDIUM risk actions execute automatically. "
        "HIGH risk actions pause for your confirmation. "
        "CRITICAL actions are blocked or require explicit confirmation."
    ),
    "medium": (
        "LOW, MEDIUM, and HIGH risk actions execute automatically. "
        "CRITICAL risk actions require your confirmation."
    ),
    "high": "All actions execute without confirmation. Use with care.",
}

_SYSTEM_TEMPLATE = """You are {agent_name}, a local-first autonomous AI agent running on the user's machine.

## Capabilities
You can use tools to browse the web, run terminal commands, read/write files, and search for information.
Always choose the most appropriate tool. Prefer reading before writing, writing before deleting.

## Tool Usage
You have tools available in your tool list. Use them directly — do not ask for permission first.
- If a tool is listed, call it. Never say "I don't have access" if the tool appears in your list.
- For LOW and MEDIUM risk actions: execute immediately without preamble or explanation.
- For HIGH risk actions: state what you are about to do in one sentence, then call the tool.
- Never offer to walk the user through doing something manually if you have a tool for it.
- Never ask "Would you like me to…?" for routine actions. Just do them.

## Guidelines
- Think before acting. For complex tasks, state your plan before executing it.
- Report tool errors honestly - never fabricate results.
- Keep responses concise and use markdown where it aids readability.
- Ask for clarification rather than guessing when the goal is ambiguous.
- IMPORTANT: If tools are present in your tool list, USE THEM DIRECTLY. Never claim you lack network or tool access when the tools are available — just call them.
- IMPORTANT: Installed skills (e.g. Notion, Todoist, etc.) appear in your tool list. Use them immediately when the user's request matches — do not say you cannot access external services.

## Trust Level: {trust_level}
{trust_description}

## Session Capabilities (Active Right Now)
{capabilities_block}

## Current UTC Time
{utc_time}
{plan_block}"""

_CAPABILITIES_GRANTED_TEMPLATE = "Active capabilities: {granted_list}\nThese are UNLOCKED - use the corresponding tools freely without asking permission."

_CAPABILITIES_NONE = "All tools listed in your tool list are available for immediate use — no grants required. Invoke them directly without asking for permission. Only if a tool execution fails with a capability error should you ask the user to run /grant <capability>. Do NOT claim you lack tool access — just call the tool."

_PLAN_TEMPLATE = """
## Active Plan
**Goal:** {goal}
**Progress:** {progress}

**Steps:**
{steps}"""


class ContextBuilder:
    """Assembles the complete LLM message list for each agent turn."""

    def __init__(self, memory_manager: MemoryManager, agent_name: str = "NeuralClaw"):
        self.memory = memory_manager
        self.agent_name = agent_name

    async def build(
        self,
        session,                           # agent.session.Session
        user_message: str,
        tool_schemas: Optional[list[ToolSchema]] = None,
        extra_system: Optional[str] = None,
    ) -> list[Message]:
        """
        Build the full message list for this turn.

        Returns a list ready to pass directly to BaseLLMClient.generate().
        """
        messages: list[Message] = []

        # 1. System prompt
        system_text = self._build_system_prompt(session, extra_system)
        messages.append(Message.system(system_text))

        # 2. Long-term memory injection
        memory_block = await self._build_memory_block(user_message, session.id)
        if memory_block:
            messages.append(Message.system(memory_block))

        # 3. Recent conversation history (token-limited)
        history = session.get_recent_messages(n=40)
        history = self._trim_history(history)
        messages.extend(history)

        # 4. Current user message (avoid duplicating if already in history)
        last = history[-1] if history else None
        if not (last and last.role == Role.USER and last.content == user_message):
            messages.append(Message.user(user_message))

        log.debug(
            "context_builder.built",
            session_id=session.id,
            total_messages=len(messages),
            has_memory=bool(memory_block),
            history_msgs=len(history),
        )
        return messages

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self, session, extra_system: Optional[str]) -> str:
        utc_time = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())
        trust_level = session.trust_level.value
        trust_desc = _TRUST_DESCRIPTIONS.get(trust_level, "")

        plan_block = ""
        if session.active_plan:
            plan = session.active_plan
            step_lines = []
            for step in plan.steps:
                if step.completed:
                    icon = "✅"
                elif step.index == plan.current_step_index:
                    icon = "▶️ "
                else:
                    icon = "⬜"
                line = f"  {icon} {step.index + 1}. {step.description}"
                if step.result_summary:
                    line += f"\n     ↳ {step.result_summary}"
                if step.error:
                    line += f"\n     ↳ ❌ {step.error}"
                step_lines.append(line)

            plan_block = _PLAN_TEMPLATE.format(
                goal=plan.goal,
                progress=plan.progress_summary,
                steps="\n".join(step_lines),
            )

        # Build capabilities block so the LLM knows exactly what is unlocked
        granted = getattr(session, "granted_capabilities", frozenset())
        if granted:
            granted_list = "  " + "\n  ".join(sorted(granted))
            capabilities_block = _CAPABILITIES_GRANTED_TEMPLATE.format(granted_list=granted_list)
        else:
            capabilities_block = _CAPABILITIES_NONE

        prompt = _SYSTEM_TEMPLATE.format(
            agent_name=self.agent_name,
            trust_level=trust_level.upper(),
            trust_description=trust_desc,
            capabilities_block=capabilities_block,
            utc_time=utc_time,
            plan_block=plan_block,
        ).strip()

        if extra_system:
            prompt += f"\n\n{extra_system}"

        # Inject workspace files (SOUL.md, USER.md, MEMORY.md) if they exist.
        # Cached by mtime — zero overhead if files haven't changed.
        workspace_block = get_workspace_loader().build_context_block()
        if workspace_block:
            prompt += f"\n\n{workspace_block}"

        return prompt

    # ── Long-term memory ──────────────────────────────────────────────────────

    async def _build_memory_block(self, query: str, session_id: str) -> str:
        try:
            context = await self.memory.build_memory_context(
                query=query, session_id=session_id
            )
        except (NeuralClawMemoryError, MemoryStoreError, OSError) as e:
            log.warning("context_builder.memory_search_failed", error=str(e), error_type=type(e).__name__)
            return ""

        if not context:
            return ""

        if len(context) > _MEMORY_MAX_CHARS:
            context = context[:_MEMORY_MAX_CHARS] + "\n[...memory truncated]"

        return f"<long_term_memory>\n{context}\n</long_term_memory>"

    # ── History trimming ──────────────────────────────────────────────────────

    def _trim_history(self, messages: list[Message]) -> list[Message]:
        """Drop oldest messages until the history fits within the char budget."""
        if not messages:
            return []

        total = sum(len(m.content or "") for m in messages)
        if total <= _HISTORY_MAX_CHARS:
            return messages

        trimmed = list(messages)
        while len(trimmed) > 4:
            if sum(len(m.content or "") for m in trimmed) <= _HISTORY_MAX_CHARS:
                break
            trimmed.pop(0)

        dropped = len(messages) - len(trimmed)
        if dropped:
            log.debug("context_builder.history_trimmed", dropped=dropped, kept=len(trimmed))

        return trimmed