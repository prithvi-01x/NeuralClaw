"""
agent/orchestrator.py — Agent Orchestrator

The heart of NeuralClaw. Implements the observe → think → act → reflect loop.

For each user turn the orchestrator:
    1. Builds LLM context  (ContextBuilder)
    2. Calls the LLM       (BaseLLMClient)
    3. Dispatches tool calls (ToolBus) — in parallel where safe
    4. Feeds results back to LLM, repeats until no more tool calls
    5. Persists the turn to memory

Autonomous mode (run_autonomous) additionally uses the Planner to decompose
a goal into steps and drives the loop step-by-step.

Fixes applied:
  - LLMInvalidRequestError is now caught explicitly in _agent_loop and
    surfaces a clear, user-readable error panel instead of a raw exception
    traceback. This is hit when using Bytez (no tool support) with tools
    registered — the agent now gracefully informs the user rather than
    crashing.
  - Autonomous step errors are now yielded as proper ERROR responses so the
    user sees them in the UI, not just in the log.

Usage:
    orc = Orchestrator(llm, config, bus, registry, memory)
    response = await orc.run_turn(session, "What files are in ~/projects?")

    async for update in orc.run_autonomous(session, "Research WebGPU and write a report"):
        print(update.text)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Callable, Optional

from brain.llm_client import BaseLLMClient, LLMInvalidRequestError
from brain.types import (
    LLMConfig, LLMResponse, Message, Role,
    ToolCall as BrainToolCall,
    ToolResult as BrainToolResult,
    ToolSchema as BrainToolSchema,
)
from memory.memory_manager import MemoryManager
from observability.logger import bind_session, clear_session, get_logger
from tools.tool_bus import ToolBus
from tools.tool_registry import ToolRegistry
from tools.types import RiskLevel, ToolCall, ToolResult, TrustLevel

from agent.context_builder import ContextBuilder
from agent.planner import Planner
from agent.reasoner import Reasoner
from agent.response_synthesizer import AgentResponse, ResponseKind, ResponseSynthesizer
from agent.session import ActivePlan, PlanStep, Session

log = get_logger(__name__)

# Max LLM + tool-call iterations per single user turn
_MAX_ITER = 10

# Risk level at which the reasoner runs before dispatching
_REASON_THRESHOLD = RiskLevel.HIGH


class Orchestrator:
    """
    Coordinates the full agent loop for each user turn.

    Inject all dependencies via constructor; use from_settings() for
    convenience when wiring up the application.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        llm_config: LLMConfig,
        tool_bus: ToolBus,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        agent_name: str = "NeuralClaw",
        max_iterations: int = _MAX_ITER,
        on_response: Optional[Callable[[AgentResponse], None]] = None,
    ):
        self._llm = llm_client
        self._config = llm_config
        self._bus = tool_bus
        self._registry = tool_registry
        self._memory = memory_manager
        self._agent_name = agent_name
        self._max_iter = max_iterations
        self._on_response = on_response   # streaming callback for the interface

        self._ctx = ContextBuilder(memory_manager=memory_manager, agent_name=agent_name)
        self._synth = ResponseSynthesizer()
        self._planner = Planner(llm_client=llm_client, llm_config=llm_config)
        self._reasoner = Reasoner(llm_client=llm_client, llm_config=llm_config)

    # ─────────────────────────────────────────────────────────────────────────
    # Public: interactive turn
    # ─────────────────────────────────────────────────────────────────────────

    async def run_turn(self, session: Session, user_message: str) -> AgentResponse:
        """
        Process one user message and return the final AgentResponse.
        Entry point for interactive (/ask) mode.
        """
        bind_session(session.id, session.user_id)
        session.reset_cancel()

        log.info("orchestrator.turn_start", session_id=session.id,
                 user_message=user_message[:120])
        t0 = time.monotonic()

        try:
            session.add_user_message(user_message)
            final = await self._agent_loop(session, user_message)
            session.add_assistant_message(final.text)

            # Persist turn to long-term memory (fire-and-forget)
            asyncio.create_task(self._persist_turn(session, user_message, final.text))

            log.info("orchestrator.turn_done", session_id=session.id,
                     ms=round((time.monotonic() - t0) * 1000),
                     tool_calls=session.tool_call_count)
            return final

        except asyncio.CancelledError:
            log.info("orchestrator.turn_cancelled", session_id=session.id)
            return self._synth.cancelled()
        except Exception as e:
            log.error("orchestrator.turn_error", error=str(e), exc_info=True)
            return self._synth.error(type(e).__name__, detail=str(e))
        finally:
            clear_session()

    # ─────────────────────────────────────────────────────────────────────────
    # Public: autonomous multi-step mode
    # ─────────────────────────────────────────────────────────────────────────

    async def run_autonomous(
        self, session: Session, goal: str
    ) -> AsyncIterator[AgentResponse]:
        """
        Autonomous mode: plan → execute each step → reflect.
        Yields AgentResponse objects so the interface can stream progress.
        """
        bind_session(session.id, session.user_id)
        session.reset_cancel()
        log.info("orchestrator.autonomous_start", goal=goal[:100])

        try:
            # 1. Plan
            available_tools = self._registry.list_names()
            memory_ctx = await self._memory.build_memory_context(goal, session.id)
            plan_result = await self._planner.create_plan(
                goal=goal, available_tools=available_tools, context=memory_ctx
            )

            if not plan_result.steps:
                yield self._synth.error("Planner returned no steps.")
                return

            # 2. Register episode + plan on session
            episode_id = await self._memory.start_episode(session.id, goal)
            session.set_plan(goal, plan_result.steps, episode_id=episode_id)
            yield self._synth.plan_preview(goal, plan_result.steps)

            # 3. Execute each step
            steps_taken: list[str] = []
            plan = session.active_plan

            while plan and not plan.is_complete:
                if session.is_cancelled():
                    yield self._synth.cancelled()
                    break

                step = plan.current_step
                if step is None:
                    break

                log.info("orchestrator.plan_step", index=step.index,
                         description=step.description[:80])

                yield self._synth.tool_progress(
                    tool_name=f"Step {step.index + 1}",
                    step=step.index + 1,
                    total=len(plan.steps),
                    detail=step.description,
                )

                step_prompt = (
                    f"Execute this step:\n{step.description}\n\n"
                    f"Overall goal: {goal}"
                )

                try:
                    step_response = await self._agent_loop(session, step_prompt)
                    step.result_summary = step_response.text[:200]
                    steps_taken.append(
                        f"Step {step.index + 1}: {step.description} "
                        f"→ {step_response.text[:100]}"
                    )
                    plan.advance()

                    if len(step_response.text.strip()) > 80:
                        yield step_response

                except Exception as step_err:
                    step.error = str(step_err)
                    log.warning("orchestrator.step_failed", step=step.description[:80],
                                error=str(step_err))

                    # Yield the error so the user sees it in the UI
                    yield self._synth.error(
                        f"Step {step.index + 1} encountered an error",
                        detail=str(step_err),
                    )

                    recovery = await self._planner.create_recovery(
                        goal=goal,
                        failed_step=step.description,
                        error=str(step_err),
                        steps_remaining=[
                            s.description
                            for s in plan.steps[plan.current_step_index + 1:]
                        ],
                    )

                    if recovery.can_recover and recovery.recovery_steps:
                        insert_at = plan.current_step_index + 1
                        for i, rdesc in enumerate(recovery.recovery_steps):
                            new_step = PlanStep(
                                index=insert_at + i,
                                description=f"[Recovery] {rdesc}",
                            )
                            plan.steps.insert(insert_at + i, new_step)
                        # Renumber every step so indices match position
                        for j, s in enumerate(plan.steps):
                            s.index = j
                        if recovery.skip_failed_step:
                            plan.advance()
                        yield AgentResponse(
                            kind=ResponseKind.PROGRESS,
                            text=f"⚠️ Attempting recovery: {recovery.recovery_steps[0]}",
                            is_final=False,
                        )
                    else:
                        yield self._synth.error(
                            f"Step {step.index + 1} failed and could not be recovered.",
                            detail=str(step_err),
                        )
                        break

            # 4. Reflect + commit episode
            if steps_taken and not session.is_cancelled():
                outcome = "success" if (plan and plan.is_complete) else "partial"
                lesson = await self._reasoner.reflect(
                    goal=goal, steps_taken=steps_taken, outcome=outcome
                )
                if lesson:
                    await self._memory.add_reflection(session.id, lesson, context=goal)

                episode_id = plan.episode_id if plan else None
                if episode_id:
                    await self._memory.commit_episode(
                        episode_id=episode_id,
                        outcome=outcome,
                        summary=f"{outcome.capitalize()}: {goal}",
                        steps=steps_taken,
                        tool_count=session.tool_call_count,
                        turn_count=session.turn_count,
                    )

            if plan and plan.is_complete:
                yield AgentResponse(
                    kind=ResponseKind.TEXT,
                    text=(
                        f"✅ **Task complete!**\n_{goal}_\n\n"
                        f"Used {session.tool_call_count} tool calls across "
                        f"{len(steps_taken)} steps."
                    ),
                )

            session.clear_plan()

        except asyncio.CancelledError:
            yield self._synth.cancelled()
        except Exception as e:
            log.error("orchestrator.autonomous_error", error=str(e), exc_info=True)
            yield self._synth.error(type(e).__name__, detail=str(e))
        finally:
            clear_session()

    # ─────────────────────────────────────────────────────────────────────────
    # Inner agent loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _agent_loop(self, session: Session, user_message: str) -> AgentResponse:
        """
        Core observe → think → act loop.
        Iterates until the LLM produces a final text response (no tool calls),
        or until _max_iter is reached.
        """
        # Build tool schemas — only for providers that support tool calling.
        # Providers like Bytez declare supports_tools = False; passing tools
        # to them raises LLMInvalidRequestError before hitting the network,
        # which previously broke every single turn including simple "hey".
        _provider_supports_tools = getattr(self._llm, "supports_tools", True)
        llm_tools: list[BrainToolSchema] = (
            [
                BrainToolSchema(
                    name=s.name,
                    description=s.description,
                    parameters=s.parameters,
                )
                for s in self._registry.list_schemas(enabled_only=True)
            ]
            if _provider_supports_tools
            else []
        )
        if not _provider_supports_tools:
            log.debug("orchestrator.tools_suppressed",
                      reason="provider does not support tool calling")

        # Build full message context
        messages = await self._ctx.build(session=session, user_message=user_message)

        for iteration in range(self._max_iter):
            if session.is_cancelled():
                raise asyncio.CancelledError()

            log.debug("orchestrator.llm_call", iteration=iteration,
                      msg_count=len(messages))

            # ── LLM call ──────────────────────────────────────────────────────
            try:
                llm_resp: LLMResponse = await self._llm.generate(
                    messages=messages,
                    config=self._config,
                    tools=llm_tools or None,
                )
            except LLMInvalidRequestError as e:
                # Defensive: provider raised despite our supports_tools check.
                log.warning("orchestrator.llm_invalid_request", error=str(e))
                return self._synth.error(
                    "Provider limitation",
                    detail=str(e),
                )

            session.record_token_usage(
                llm_resp.usage.input_tokens,
                llm_resp.usage.output_tokens,
            )

            # ── No tool calls → done ──────────────────────────────────────────
            if not llm_resp.has_tool_calls:
                return self._synth.from_llm(llm_resp)

            # ── Tool calls → append assistant message with tool_calls field ───
            messages.append(Message(
                role=Role.ASSISTANT,
                content=llm_resp.content,
                tool_calls=llm_resp.tool_calls,
            ))

            # ── Execute tool calls (parallel where safe) ──────────────────────
            tool_results: list[ToolResult] = await self._execute_tool_calls(
                llm_resp.tool_calls, session
            )

            # ── Append tool results to messages ───────────────────────────────
            for brain_tc, result in zip(llm_resp.tool_calls, tool_results):
                brain_tr = BrainToolResult(
                    tool_call_id=brain_tc.id,
                    name=brain_tc.name,
                    content=result.content,
                    is_error=result.is_error,
                )
                messages.append(Message.tool_response(brain_tr))

                # Stream individual tool result to interface if callback set
                if self._on_response:
                    resp = (self._synth.tool_error(result)
                            if result.is_error
                            else self._synth.tool_success(result))
                    self._on_response(resp)

        log.warning("orchestrator.max_iter_reached", session_id=session.id,
                    iterations=self._max_iter)
        return self._synth.error(
            f"Reached maximum iterations ({self._max_iter}). "
            "Task may be too complex for a single turn."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Tool execution
    # ─────────────────────────────────────────────────────────────────────────

    async def _execute_tool_calls(
        self,
        brain_tool_calls: list[BrainToolCall],
        session: Session,
    ) -> list[ToolResult]:
        """
        Execute a list of tool calls.
        LOW/MEDIUM risk → parallel. HIGH/CRITICAL → sequential.

        Bug 7 fix: results were returned as low_results + high_results, which
        reorders them relative to brain_tool_calls.  The caller zips results
        with llm_resp.tool_calls positionally, so a mixed list (e.g. [low,
        high, low]) caused tool_call_id ↔ result mismatches and fed wrong
        results back to the LLM.  We now collect results into a pre-sized list
        indexed by original position so order is always preserved.
        """
        low: list[tuple[int, BrainToolCall]] = []
        high: list[tuple[int, BrainToolCall]] = []

        for idx, btc in enumerate(brain_tool_calls):
            schema = self._registry.get_schema(btc.name)
            risk = schema.risk_level if schema else RiskLevel.MEDIUM
            (high if risk >= RiskLevel.HIGH else low).append((idx, btc))

        # Pre-allocate results list so we can insert by original index
        results: list[ToolResult | None] = [None] * len(brain_tool_calls)

        if low:
            tasks = [self._dispatch_one(btc, session) for _, btc in low]
            parallel = await asyncio.gather(*tasks, return_exceptions=True)
            for (idx, btc), r in zip(low, parallel):
                if isinstance(r, Exception):
                    results[idx] = ToolResult.error(btc.id, btc.name, str(r))
                else:
                    results[idx] = r

        for idx, btc in high:
            if session.is_cancelled():
                results[idx] = ToolResult.error(btc.id, btc.name, "Cancelled.")
                continue
            results[idx] = await self._dispatch_one(btc, session)

        # All slots must be filled
        unfilled = [i for i, r in enumerate(results) if r is None]
        if unfilled:
            raise RuntimeError(f"BUG: unfilled tool result slots at indices {unfilled}")
        return results  # type: ignore[return-value]

    async def _dispatch_one(self, btc: BrainToolCall, session: Session) -> ToolResult:
        """
        Dispatch a single tool call:
          1. Optional reasoner pre-check for high-risk calls
          2. Wire confirmation callback into session
          3. Call ToolBus.dispatch()
          4. Record to episodic memory
        """
        schema = self._registry.get_schema(btc.name)

        # Optional reasoner pre-check
        if schema and schema.risk_level >= _REASON_THRESHOLD:
            verdict = await self._reasoner.evaluate_tool_call(
                tool_name=btc.name,
                tool_args=btc.arguments,
                goal=session.active_plan.goal if session.active_plan else "user request",
                step=(session.active_plan.current_step.description
                      if session.active_plan and session.active_plan.current_step else ""),
            )
            if not verdict.proceed:
                log.warning("orchestrator.reasoner_blocked", tool=btc.name,
                            concern=verdict.concern)
                return ToolResult.error(
                    btc.id, btc.name,
                    f"Reasoner blocked: {verdict.concern or verdict.reasoning}",
                    risk_level=schema.risk_level,
                )

        # Wire confirmation through session future
        async def _on_confirm(decision) -> bool:
            future = session.register_confirmation(decision.tool_call_id)
            if self._on_response:
                self._on_response(
                    self._synth.confirmation_request(decision, btc.arguments)
                )
            try:
                return await asyncio.wait_for(future, timeout=120.0)
            except asyncio.TimeoutError:
                log.warning("orchestrator.confirm_timeout",
                            tool_call_id=decision.tool_call_id)
                session._pending_confirmations.pop(decision.tool_call_id, None)
                return False

        # Convert brain ToolCall → tools ToolCall for the bus
        tool_call = ToolCall(id=btc.id, name=btc.name, arguments=btc.arguments)

        result = await self._bus.dispatch(
            tool_call,
            trust_level=session.trust_level,
            on_confirm_needed=_on_confirm,
        )

        session.record_tool_call()

        # Record to episodic memory (fire-and-forget)
        asyncio.create_task(self._memory.record_tool_call(
            session_id=session.id,
            tool_name=btc.name,
            arguments=btc.arguments,
            result=result.content[:500],
            is_error=result.is_error,
            risk_level=result.risk_level.value,
            duration_ms=result.duration_ms,
            episode_id=session.active_plan.episode_id if session.active_plan else None,
        ))

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Compaction
    # ─────────────────────────────────────────────────────────────────────────

    async def compact_session(self, session: Session, keep_recent: int = 4) -> str:
        """
        Summarise and compact the session's short-term conversation buffer.

        Calls the LLM to produce a ~5-sentence summary of everything in the
        buffer, then replaces all but the `keep_recent` most-recent user turns
        with that summary.  The summary is injected into subsequent contexts so
        the agent retains awareness of earlier topics.

        Also persists the summary to long-term memory so it survives restarts.

        Returns the summary text for display.
        Raises RuntimeError if there is nothing to compact.
        """
        summary = await session.memory.compact(
            llm_client=self._llm,
            llm_config=self._config,
            keep_recent=keep_recent,
        )

        # Persist to long-term memory (fire-and-forget)
        asyncio.create_task(self._memory.store(
            f"[Compact summary] {summary}",
            collection="conversations",
            metadata={
                "session_id": session.id,
                "user_id":    session.user_id,
                "type":       "compact_summary",
                "turn":       session.turn_count,
            },
        ))

        log.info(
            "orchestrator.session_compacted",
            session_id=session.id,
            turn=session.turn_count,
            keep_recent=keep_recent,
        )
        return summary

    # ─────────────────────────────────────────────────────────────────────────
    # Memory persistence
    # ─────────────────────────────────────────────────────────────────────────

    async def _persist_turn(self, session: Session, user_msg: str, reply: str) -> None:
        try:
            summary = f"User: {user_msg[:200]}\nAssistant: {reply[:300]}"
            await self._memory.store(
                summary,
                collection="conversations",
                metadata={
                    "session_id": session.id,
                    "user_id": session.user_id,
                    "turn": session.turn_count,
                },
            )
        except Exception as e:
            log.warning("orchestrator.persist_failed", error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        settings,
        llm_client: BaseLLMClient,
        tool_bus: ToolBus,
        tool_registry: ToolRegistry,
        memory_manager: MemoryManager,
        on_response: Optional[Callable[[AgentResponse], None]] = None,
    ) -> "Orchestrator":
        """Create an Orchestrator from the NeuralClaw Settings object."""
        llm_config = LLMConfig(
            model=settings.default_llm_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        return cls(
            llm_client=llm_client,
            llm_config=llm_config,
            tool_bus=tool_bus,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            agent_name=settings.agent.name,
            max_iterations=settings.agent.max_iterations_per_turn,
            on_response=on_response,
        )