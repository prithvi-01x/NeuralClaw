"""
agent/orchestrator.py — Agent Orchestrator

The heart of NeuralClaw. Implements the observe → think → act → reflect loop.

For each user turn the orchestrator:
    1. Builds LLM context  (ContextBuilder)
    2. Calls the LLM       (BaseLLMClient)
    3. Dispatches tool calls (Executor → SkillBus) — in parallel where safe
    4. Feeds results back to LLM, repeats until a termination condition is met
    5. Reflects and persists the turn to memory (Reflector)

Phase 3 hardening:
  - run_turn() now returns TurnResult(status, response, steps_taken, duration_ms)
  - _agent_loop() NEVER raises — every code path returns a TurnResult
  - All termination conditions are explicit: SUCCESS, ITER_LIMIT, TIMEOUT,
    BLOCKED, CONTEXT_LIMIT, ERROR
  - Executor and Reflector extracted as separate, independently testable classes

Usage:
    orc = Orchestrator(llm, config, bus, registry, memory)
    turn = await orc.run_turn(session, "What files are in ~/projects?")
    print(turn.status, turn.response.text)

    async for update in orc.run_autonomous(session, "Research WebGPU"):
        print(update.text)
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from brain.llm_client import BaseLLMClient, LLMContextError, LLMInvalidRequestError
from brain.types import (
    LLMConfig, LLMResponse, Message, Role,
    ToolCall as BrainToolCall,
    ToolResult as BrainToolResult,
    ToolSchema as BrainToolSchema,
)
from memory.memory_manager import MemoryManager
from observability.logger import bind_session, clear_session, get_logger
from observability.trace import TraceContext
from skills.bus import SkillBus
from skills.registry import SkillRegistry
from skills.types import RiskLevel, TrustLevel
from skills.md_loader import MarkdownSkillLoader as _MdSkillLoader

from agent.context_builder import ContextBuilder
from agent.executor import Executor
from agent.planner import Planner
from agent.reasoner import Reasoner
from agent.reflector import Reflector
from agent.response_synthesizer import AgentResponse, ResponseKind, ResponseSynthesizer
from agent.session import ActivePlan, PlanStep, Session
from exceptions import NeuralClawError, PlanError, IterationLimitError, LLMError, MemoryError
from brain.capabilities import get_capabilities

log = get_logger(__name__)

# Max LLM + tool-call iterations per single user turn
_MAX_ITER = 10

# Risk level at which the reasoner runs before dispatching
_REASON_THRESHOLD = RiskLevel.HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Fire-and-forget helper
# ─────────────────────────────────────────────────────────────────────────────

from agent.utils import fire_and_forget as _fire_and_forget


# ─────────────────────────────────────────────────────────────────────────────
# TurnStatus and TurnResult  (Phase 3)
# ─────────────────────────────────────────────────────────────────────────────

class TurnStatus(str, Enum):
    """
    Every possible outcome of a single agent turn.
    run_turn() always returns one of these — it never raises.
    """
    SUCCESS       = "success"        # LLM returned a final text response
    ITER_LIMIT    = "iter_limit"     # max_iterations reached without final answer
    TIMEOUT       = "timeout"        # max_turn_timeout_seconds elapsed
    BLOCKED       = "blocked"        # safety kernel blocked a critical step
    CONTEXT_LIMIT = "context_limit"  # LLMContextError: input exceeded model limit
    ERROR         = "error"          # Unhandled exception in skill or LLM


@dataclass(frozen=True)
class TurnResult:
    """
    Typed result from a single agent turn. Always returned — never raises.

    Fields:
        status:      What happened.
        response:    The AgentResponse to display to the user.
        steps_taken: Number of tool calls executed in this turn.
        duration_ms: Wall-clock time for the full turn in milliseconds.
    """
    status: TurnStatus
    response: AgentResponse
    steps_taken: int = 0
    duration_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status == TurnStatus.SUCCESS



# ─────────────────────────────────────────────────────────────────────────────
# Capability helpers (module-level so they can be used inside _agent_loop)
# ─────────────────────────────────────────────────────────────────────────────

def _make_legacy_error(tool_call_id: str, name: str, error: str) -> "SkillResult":
    """Return a SkillResult representing a failed/blocked call."""
    from skills.types import SkillResult
    return SkillResult.fail(
        skill_name=name,
        skill_call_id=tool_call_id,
        error=error,
        error_type="ExecutorError",
        blocked=True,
    )


_TOOL_ERROR_PHRASES = (
    "does not support tools",
    "does not support tool",
    "tool_use is not supported",
    "function calling is not supported",
    "unsupported parameter: tools",
    "tools is not supported",
    "tool calls",
)


def _is_tool_error(err_lower: str) -> bool:
    """Return True if an error message indicates tool calling is unsupported."""
    return any(p in err_lower for p in _TOOL_ERROR_PHRASES)


def _provider_name_from_client(client) -> str:
    """Return the provider name for the given client.

    Prefers the typed ``provider_name`` property introduced in Phase 6.
    Falls back to class-name string matching for any third-party subclass
    that hasn't been updated yet — guarantees backward compatibility.
    """
    from brain.llm_client import ResilientLLMClient
    actual = client
    if isinstance(client, ResilientLLMClient):
        actual = client._primary

    # Fast path — typed property (all first-party clients implement this)
    if hasattr(actual, "provider_name"):
        try:
            name = actual.provider_name
            if isinstance(name, str) and name:
                return name
        except AttributeError:
            pass

    # Fallback — class-name heuristic for third-party/unported subclasses
    cls_name = type(actual).__name__.lower()
    if "openai" in cls_name:
        return "openai"
    if "anthropic" in cls_name:
        return "anthropic"
    if "ollama" in cls_name:
        return "ollama"
    if "gemini" in cls_name:
        return "gemini"
    if "openrouter" in cls_name:
        return "openrouter"
    if "bytez" in cls_name:
        return "bytez"
    return "unknown"


def _register_no_tools(client, model_id: str) -> None:
    """Register model as no-tools in capability registry and update client flag."""
    from brain.capabilities import register_capabilities
    provider = _provider_name_from_client(client)
    register_capabilities(provider, model_id, supports_tools=False)
    # Directly update client instance attribute if possible
    try:
        client.supports_tools = False
    except AttributeError:
        pass
    # Also update inner primary client for ResilientLLMClient
    from brain.llm_client import ResilientLLMClient
    if isinstance(client, ResilientLLMClient):
        try:
            client._primary.supports_tools = False
            client.supports_tools = False
        except AttributeError:
            pass



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
        tool_bus: SkillBus,
        tool_registry: SkillRegistry,
        memory_manager: MemoryManager,
        agent_name: str = "NeuralClaw",
        max_iterations: int = _MAX_ITER,
        max_turn_timeout: int = 300,
        on_response: Optional[Callable[[AgentResponse], None]] = None,
    ):
        self._llm = llm_client
        self._config = llm_config
        self._bus = tool_bus
        self._registry = tool_registry
        self._memory = memory_manager
        self._agent_name = agent_name
        self._max_iter = max_iterations
        self._max_turn_timeout = max_turn_timeout
        self._on_response = on_response

        self._ctx = ContextBuilder(memory_manager=memory_manager, agent_name=agent_name)
        self._synth = ResponseSynthesizer()
        self._planner = Planner(llm_client=llm_client, llm_config=llm_config)
        self._reasoner = Reasoner(llm_client=llm_client, llm_config=llm_config)

        # Phase 3: Executor and Reflector as separate objects
        # We default the explicit confirmation timeout to max_turn_timeout / 2 or so if unset,
        # but in practice this is injected via from_settings. For raw instantiation we default it.
        self._executor = Executor(
            registry=tool_registry,
            bus=tool_bus,
            reasoner=self._reasoner,
            memory_manager=memory_manager,
            confirmation_timeout=max_turn_timeout, # basic default, from_settings has the real one
        )
        self._reflector = Reflector(
            reasoner=self._reasoner,
            memory_manager=memory_manager,
        )

        # Cache for tool schemas — built once from the registry at startup and
        # on model swap. The registry is read-only after init, so rebuilding
        # on every _agent_loop call is wasteful with many skills.
        self._cached_tool_schemas: list[BrainToolSchema] | None = None

        # Cache for markdown skill instructions block — rebuilt only when None
        # (first call or after swap_llm_client invalidates it).
        self._cached_md_instructions: str | None = None
    # ─────────────────────────────────────────────────────────────────────────

    def reset_tool_support(self) -> None:
        """
        Re-enable tool calling for the current model after a stale disable.

        Clears both the capability registry entry and the client-instance flag
        that get set when an LLMInvalidRequestError causes automatic fallback to
        chat-only mode. Use via the CLI's /resetcaps command.

        Safe to call at any time — if tools were never disabled, this is a no-op.
        """
        from brain.capabilities import reset_capabilities
        provider = _provider_name_from_client(self._llm)
        model_id = self._config.model

        reset_capabilities(provider, model_id)

        # Re-enable the client-instance flag
        try:
            self._llm.supports_tools = True
        except AttributeError:
            pass
        # Also propagate to inner primary for ResilientLLMClient
        from brain.llm_client import ResilientLLMClient
        if isinstance(self._llm, ResilientLLMClient):
            try:
                self._llm._primary.supports_tools = True
                self._llm.supports_tools = True
            except AttributeError:
                pass

        # Invalidate tool schema cache so it is rebuilt on next turn
        self._cached_tool_schemas = None

        log.info(
            "orchestrator.tool_support_reset",
            provider=provider,
            model=model_id,
        )

    # ─────────────────────────────────────────────────────────────────────────

    def swap_llm_client(self, new_client: BaseLLMClient, new_model_id: str = "") -> None:
        """
        Replace the active LLM client and model ID without restarting.

        Propagates to Planner and Reasoner so all components use the new model.
        Also refreshes the capability registry so _agent_loop immediately picks
        up the correct supports_tools value for the new model — no restart needed.

        Args:
            new_client:   The new LLM client instance.
            new_model_id: The model ID string for LLMConfig (e.g. "codellama:latest").
        """
        self._llm = new_client
        if new_model_id and hasattr(self, "_config"):
            self._config.model = new_model_id

        # Propagate to sub-components via a safe helper.
        # Using a named helper (instead of raw hasattr chains) means a rename
        # of an internal attribute produces a log warning rather than a silent skip.
        _SENTINEL = object()

        def _set_model(obj, attr: str, value: str) -> None:
            cfg = getattr(obj, attr, _SENTINEL)
            if cfg is _SENTINEL:
                return  # attribute doesn't exist on this object — OK
            if hasattr(cfg, "model"):
                cfg.model = value
            else:
                log.warning(
                    "orchestrator.swap_model_attr_unexpected",
                    obj=type(obj).__name__,
                    attr=attr,
                    cfg_type=type(cfg).__name__,
                )

        if hasattr(self._planner, "_llm"):
            self._planner._llm = new_client
        if new_model_id:
            _set_model(self._planner, "_config", new_model_id)

        if hasattr(self._reasoner, "_llm"):
            self._reasoner._llm = new_client
        if new_model_id:
            _set_model(self._reasoner, "_config", new_model_id)
            _set_model(self._reasoner, "_reflect_config", new_model_id)

        # Refresh OllamaClient's per-instance capability flags for the new model
        if new_model_id and hasattr(new_client, "_refresh_capabilities"):
            try:
                new_client._refresh_capabilities(new_model_id)
            except (AttributeError, RuntimeError, OSError):
                pass

        # Also propagate to inner primary for ResilientLLMClient
        from brain.llm_client import ResilientLLMClient
        if isinstance(new_client, ResilientLLMClient):
            inner = new_client._primary
            if new_model_id and hasattr(inner, "_refresh_capabilities"):
                try:
                    inner._refresh_capabilities(new_model_id)
                    # Mirror the inner's supports_tools flag upward
                    new_client.supports_tools = getattr(inner, "supports_tools", True)
                except (AttributeError, RuntimeError, OSError):
                    pass

        log.info(
            "orchestrator.model_swapped",
            new_model=new_model_id or "same",
            supports_tools=getattr(new_client, "supports_tools", True),
        )
        # Invalidate schema cache — the new model may have different tool support
        self._cached_tool_schemas = None
        # Invalidate MD instructions cache — new model may affect which skills apply
        self._cached_md_instructions = None



    async def run_turn(self, session: Session, user_message: str) -> TurnResult:
        """
        Process one user message and return a TurnResult.

        NEVER raises — every code path returns a TurnResult with an explicit
        TurnStatus. Entry point for interactive (/ask) mode.
        """
        bind_session(session.id, session.user_id)
        _trace = TraceContext.for_session(session.id)
        _trace.new_turn().bind()
        session.reset_cancel()

        log.info("orchestrator.turn_start", session_id=session.id,
                 user_message=user_message[:120])
        t0 = time.monotonic()

        try:
            session.add_user_message(user_message)

            # Wrap in a timeout so the loop can never hang forever
            try:
                turn_result = await asyncio.wait_for(
                    self._agent_loop(session, user_message),
                    timeout=self._max_turn_timeout,
                )
            except asyncio.TimeoutError:
                log.warning("orchestrator.turn_timeout", session_id=session.id,
                            timeout=self._max_turn_timeout)
                response = self._synth.error(
                    f"Turn timed out after {self._max_turn_timeout}s.",
                    detail="The task was too slow to complete. Try a simpler request.",
                )
                turn_result = TurnResult(
                    status=TurnStatus.TIMEOUT,
                    response=response,
                    steps_taken=session.tool_call_count,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            session.add_assistant_message(turn_result.response.text)

            # Persist turn to long-term memory (fire-and-forget)
            _fire_and_forget(
                self._persist_turn(session, user_message, turn_result.response.text),
                label="persist_turn",
            )

            log.info(
                "orchestrator.turn_done",
                session_id=session.id,
                status=turn_result.status.value,
                ms=round(turn_result.duration_ms),
                tool_calls=turn_result.steps_taken,
            )
            return turn_result

        except asyncio.CancelledError:
            log.info("orchestrator.turn_cancelled", session_id=session.id)
            return TurnResult(
                status=TurnStatus.ERROR,
                response=self._synth.cancelled(),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except NeuralClawError as e:
            log.error("orchestrator.turn_error", error=str(e), error_type=type(e).__name__, exc_info=True)
            return TurnResult(
                status=TurnStatus.ERROR,
                response=self._synth.error(type(e).__name__, detail=str(e)),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        except BaseException as e:
            log.error("orchestrator.turn_unexpected_error", error=str(e), error_type=type(e).__name__, exc_info=True)
            return TurnResult(
                status=TurnStatus.ERROR,
                response=self._synth.error(type(e).__name__, detail=str(e)),
                duration_ms=(time.monotonic() - t0) * 1000,
            )
        finally:
            clear_session()

    # ─────────────────────────────────────────────────────────────────────────
    # Public: autonomous multi-step mode
    # ─────────────────────────────────────────────────────────────────────────

    async def run_autonomous(
        self, session: Session, goal: str
    ) -> AsyncGenerator[AgentResponse, None]:
        """
        Autonomous mode: plan → execute each step → reflect.
        Yields AgentResponse objects so the interface can stream progress.
        """
        bind_session(session.id, session.user_id)
        _trace = TraceContext.for_session(session.id)
        _trace.new_turn().bind()
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

            # Phase 4: Create TaskMemory for this plan (tracked inside the store)
            plan_id = session.active_plan.id if session.active_plan else f"plan_{episode_id}"
            self._memory.task_create(plan_id=plan_id, goal=goal)

            yield self._synth.plan_preview(goal, plan_result.steps)

            # 3. Execute each step
            steps_taken: list[str] = []
            plan = session.active_plan
            _MAX_RECOVERY_DEPTH = 3  # max consecutive recovery attempts per step
            _MAX_TOTAL_STEPS = 100   # absolute cap on plan length
            _recovery_depth = 0       # resets to 0 whenever a step succeeds

            # Configured overall timeout for autonomous execution, hardcapped if missing
            timeout = getattr(self._settings.agent, "max_autonomous_timeout_seconds", 3600)

            async def _execute_loop() -> AsyncGenerator[AgentResponse, None]:
                nonlocal _recovery_depth
                while plan and not plan.is_complete:
                    if session.is_cancelled():
                        yield self._synth.cancelled()
                        break

                    if len(plan.steps) > _MAX_TOTAL_STEPS:
                        yield self._synth.error(
                            f"Plan exceeded absolute maximum of {_MAX_TOTAL_STEPS} steps. "
                            "Stopping to prevent infinite recovery loops."
                        )
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
                        # Phase 4: log step start in TaskMemory
                        self._memory.task_log_step(
                            plan_id=plan_id,
                            step_id=f"step_{step.index}",
                            description=step.description,
                        )
                        step_turn = await self.run_turn(session, step_prompt)
                        # Phase 4: record result in TaskMemory
                        self._memory.task_update_result(
                            plan_id=plan_id,
                            step_id=f"step_{step.index}",
                            result_content=step_turn.response.text[:300],
                            is_error=not step_turn.succeeded,
                            duration_ms=step_turn.duration_ms,
                        )
                        step.result_summary = step_turn.response.text[:200]
                        steps_taken.append(
                            f"Step {step.index + 1}: {step.description} "
                            f"→ {step_turn.response.text[:100]}"
                        )
                        plan.advance()
                        _recovery_depth = 0  # successful step resets recovery counter

                        if len(step_turn.response.text.strip()) > 80:
                            yield step_turn.response

                    except NeuralClawError as step_err:
                        step.error = str(step_err)
                        log.warning("orchestrator.step_failed", step=step.description[:80],
                                    error=str(step_err), error_type=type(step_err).__name__)
                    except BaseException as step_err:
                        step.error = str(step_err)
                        log.error("orchestrator.step_unexpected_error", step=step.description[:80],
                                  error=str(step_err), error_type=type(step_err).__name__, exc_info=True)

                        yield self._synth.error(
                            f"Step {step.index + 1} encountered an error",
                            detail=str(step_err),
                        )

                        # Enforce maximum consecutive recovery depth
                        if _recovery_depth >= _MAX_RECOVERY_DEPTH:
                            log.warning(
                                "orchestrator.recovery_depth_exceeded",
                                step=step.description[:80],
                                max_depth=_MAX_RECOVERY_DEPTH,
                            )
                            yield self._synth.error(
                                f"Step {step.index + 1} failed after {_MAX_RECOVERY_DEPTH} "
                                "recovery attempts. Stopping autonomous execution.",
                                detail=str(step_err),
                            )
                            break

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
                            _recovery_depth += 1
                            insert_at = plan.current_step_index + 1
                            # Insert all recovery steps at once, then do a single O(n)
                            # re-index pass (avoids O(n²) re-index inside the loop).
                            new_steps = [
                                PlanStep(
                                    index=insert_at + i,
                                    description=f"[Recovery] {rdesc}",
                                )
                                for i, rdesc in enumerate(recovery.recovery_steps)
                            ]
                            plan.steps[insert_at:insert_at] = new_steps
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

            # Run the loop with an overall wall-clock timeout
            import asyncio
            try:
                # We can't use wait_for directly on an async generator, so we iterate it
                # with a timeout on the *entire* process. We use an async def to allow
                # wait_for to manage the timeout over the loop execution time.
                async def _consume():
                    async for item in _execute_loop():
                        yield item

                # Since async generators don't support wait_for directly, we must wrap each
                # next() call in wait_for, keeping a rolling timeout.
                loop_start = asyncio.get_running_loop().time()
                agen = _execute_loop()
                while True:
                    elapsed = asyncio.get_running_loop().time() - loop_start
                    remaining = timeout - elapsed
                    if remaining <= 0:
                        raise asyncio.TimeoutError()

                    try:
                        item = await asyncio.wait_for(anext(agen), timeout=remaining)
                        yield item
                    except StopAsyncIteration:
                        break
            except asyncio.TimeoutError:
                log.warning("orchestrator.autonomous_timeout", session_id=session.id, timeout=timeout)
                yield self._synth.error(
                    f"Autonomous execution timed out after {timeout} seconds."
                )

            # 4. Reflect + commit episode via Reflector
            if steps_taken and not session.is_cancelled():
                outcome = "success" if (plan and plan.is_complete) else "partial"
                await self._reflector.commit(
                    session=session,
                    steps_taken=steps_taken,
                    goal=goal,
                    outcome=outcome,
                )

            # Phase 4: close or fail TaskMemory
            if plan and plan.is_complete:
                self._memory.task_close(plan_id)
            else:
                self._memory.task_fail(plan_id, reason="Plan did not complete")

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
        except NeuralClawError as e:
            log.error("orchestrator.autonomous_error", error=str(e), error_type=type(e).__name__, exc_info=True)
            yield self._synth.error(type(e).__name__, detail=str(e))
        except BaseException as e:
            log.error("orchestrator.autonomous_unexpected_error", error=str(e), error_type=type(e).__name__, exc_info=True)
            yield self._synth.error(type(e).__name__, detail=str(e))
        finally:
            clear_session()

    # ─────────────────────────────────────────────────────────────────────────
    # Inner agent loop
    # ─────────────────────────────────────────────────────────────────────────

    async def _agent_loop(self, session: Session, user_message: str) -> TurnResult:
        """
        Core observe → think → act loop.

        Returns a TurnResult with an explicit TurnStatus for EVERY possible
        exit path. Never raises — all exceptions are caught and wrapped.

        Termination conditions (exhaustive):
          SUCCESS       — LLM returned text with no tool calls
          ITER_LIMIT    — max_iterations reached without a final answer
          TIMEOUT       — handled by run_turn()'s wait_for wrapper
          BLOCKED       — safety kernel blocked a HIGH/CRITICAL step and
                          the LLM has no remaining viable actions
          CONTEXT_LIMIT — LLMContextError raised (input too long)
          ERROR         — unhandled exception in LLM call or skill dispatch
        """
        t0 = time.monotonic()
        steps_taken = 0

        # ── Determine tool support ────────────────────────────────────────────
        _client_supports = getattr(self._llm, "supports_tools", True)
        _registry_caps = get_capabilities(
            _provider_name_from_client(self._llm),
            self._config.model,
        )
        _supports_tools = _client_supports and _registry_caps.supports_tools

        # Build (or reuse cached) tool schema list — registry is static after
        # startup so we rebuild only when the cache is cold or after a model swap.
        if self._cached_tool_schemas is None:
            self._cached_tool_schemas = [
                BrainToolSchema(
                    name=s.name,
                    description=s.description,
                    parameters=s.parameters,
                )
                for s in self._registry.list_schemas(enabled_only=True)
            ]
        all_tool_schemas = self._cached_tool_schemas
        llm_tools: list[BrainToolSchema] = all_tool_schemas if _supports_tools else []

        if not _supports_tools:
            log.warning(
                "orchestrator.tools_disabled",
                model=self._config.model,
                client_flag=_client_supports,
                registry_flag=_registry_caps.supports_tools,
                reason=(
                    "client.supports_tools=False — set by prior LLMInvalidRequestError fallback"
                    if not _client_supports
                    else "capability registry has supports_tools=False for this model"
                ),
            )

        # ── Build message context ─────────────────────────────────────────────
        # Inject markdown skill instructions into system prompt (cached per instance).
        if self._cached_md_instructions is None:
            self._cached_md_instructions = _MdSkillLoader().get_instructions_block(self._registry)
        _md_extra = self._cached_md_instructions
        messages = await self._ctx.build(
            session=session,
            user_message=user_message,
            extra_system=_md_extra if _md_extra else None,
        )
        _chat_only_notified = not _supports_tools

        for iteration in range(self._max_iter):
            if session.is_cancelled():
                return TurnResult(
                    status=TurnStatus.ERROR,
                    response=self._synth.cancelled(),
                    steps_taken=steps_taken,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            log.debug("orchestrator.llm_call", iteration=iteration,
                      msg_count=len(messages), tools_enabled=bool(llm_tools))

            # ── LLM call ──────────────────────────────────────────────────────
            try:
                llm_resp: LLMResponse = await self._llm.generate(
                    messages=messages,
                    config=self._config,
                    tools=llm_tools or None,
                )
            except LLMContextError as e:
                log.warning("orchestrator.context_limit", error=str(e))
                return TurnResult(
                    status=TurnStatus.CONTEXT_LIMIT,
                    response=self._synth.error(
                        "Context limit reached",
                        detail="The conversation is too long. Use /compact to summarise it.",
                    ),
                    steps_taken=steps_taken,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )
            except LLMInvalidRequestError as e:
                err_lower = str(e).lower()
                if llm_tools and _is_tool_error(err_lower):
                    log.warning("orchestrator.tool_error_fallback",
                                model=self._config.model, error=str(e))
                    llm_tools = []
                    _supports_tools = False
                    _register_no_tools(self._llm, self._config.model)
                    if not _chat_only_notified and self._on_response:
                        self._on_response(self._synth.info(
                            f"⚠️ **{self._config.model}** does not support tools. "
                            "Running in chat-only mode."
                        ))
                        _chat_only_notified = True
                    try:
                        llm_resp = await self._llm.generate(
                            messages=messages,
                            config=self._config,
                            tools=None,
                        )
                    except (NeuralClawError, LLMError, Exception) as retry_err:
                        log.error("orchestrator.llm_retry_failed", error=str(retry_err), error_type=type(retry_err).__name__)
                        return TurnResult(
                            status=TurnStatus.ERROR,
                            response=self._synth.error(
                                "Provider limitation", detail=str(retry_err)
                            ),
                            steps_taken=steps_taken,
                            duration_ms=(time.monotonic() - t0) * 1000,
                        )
                else:
                    log.warning("orchestrator.llm_invalid_request", error=str(e))
                    return TurnResult(
                        status=TurnStatus.ERROR,
                        response=self._synth.error("Provider limitation", detail=str(e)),
                        steps_taken=steps_taken,
                        duration_ms=(time.monotonic() - t0) * 1000,
                    )
            except (NeuralClawError, LLMError, BaseException) as e:
                log.error("orchestrator.llm_error", error=str(e), error_type=type(e).__name__, exc_info=True)
                return TurnResult(
                    status=TurnStatus.ERROR,
                    response=self._synth.error(type(e).__name__, detail=str(e)),
                    steps_taken=steps_taken,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            session.record_token_usage(
                llm_resp.usage.input_tokens,
                llm_resp.usage.output_tokens,
            )

            # ── No tool calls → SUCCESS ───────────────────────────────────────
            if not llm_resp.has_tool_calls:
                return TurnResult(
                    status=TurnStatus.SUCCESS,
                    response=self._synth.from_llm(llm_resp),
                    steps_taken=steps_taken,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            # ── Tool calls → execute ──────────────────────────────────────────
            messages.append(Message(
                role=Role.ASSISTANT,
                content=llm_resp.content,
                tool_calls=llm_resp.tool_calls,
            ))

            tool_results, blocked = await self._execute_tool_calls(
                llm_resp.tool_calls, session
            )
            steps_taken += len(tool_results)

            # If a critical step was hard-blocked → BLOCKED status
            if blocked:
                log.warning("orchestrator.critical_step_blocked",
                            session_id=session.id)
                return TurnResult(
                    status=TurnStatus.BLOCKED,
                    response=self._synth.error(
                        "A required action was blocked by the safety kernel.",
                        detail="Increase trust level or remove the restricted action.",
                    ),
                    steps_taken=steps_taken,
                    duration_ms=(time.monotonic() - t0) * 1000,
                )

            # Append tool results to messages
            for brain_tc, result in zip(llm_resp.tool_calls, tool_results):
                raw_content = result.content
                # Defend against None/empty content — an empty string passed to
                # the provider produces a confusing "I have no output" LLM reply.
                if not raw_content:
                    raw_content = (
                        f"[Tool '{brain_tc.name}' executed successfully but returned no output]"
                        if not result.is_error
                        else f"[Tool '{brain_tc.name}' failed with no error detail]"
                    )
                brain_tr = BrainToolResult(
                    tool_call_id=brain_tc.id,
                    name=brain_tc.name,
                    content=raw_content,
                    is_error=result.is_error,
                )
                messages.append(Message.tool_response(brain_tr))

                if self._on_response:
                    resp = (self._synth.tool_error(result)
                            if result.is_error
                            else self._synth.tool_success(result))
                    self._on_response(resp)

        # ── Exhausted all iterations → ITER_LIMIT ────────────────────────────
        log.warning("orchestrator.max_iter_reached",
                    session_id=session.id, iterations=self._max_iter)
        return TurnResult(
            status=TurnStatus.ITER_LIMIT,
            response=self._synth.error(
                f"Reached maximum iterations ({self._max_iter}).",
                detail="Task may be too complex. Try breaking it into smaller steps.",
            ),
            steps_taken=steps_taken,
            duration_ms=(time.monotonic() - t0) * 1000,
        )
    # ─────────────────────────────────────────────────────────────────────────
    # Tool execution
    # ─────────────────────────────────────────────────────────────────────────

    async def _execute_tool_calls(
        self,
        brain_tool_calls: list[BrainToolCall],
        session: Session,
    ) -> tuple[list, bool]:
        """
        Execute a list of tool calls via the Executor.
        LOW/MEDIUM risk → parallel. HIGH/CRITICAL → sequential.

        Returns:
            (results, blocked) where blocked=True signals a critical step
            was hard-blocked by the safety kernel and the loop should halt.

        Preserves result ordering relative to brain_tool_calls (indexed insert).
        """
        low: list[tuple[int, BrainToolCall]] = []
        high: list[tuple[int, BrainToolCall]] = []

        # Guard: nothing to do
        if not brain_tool_calls:
            return [], False

        for idx, btc in enumerate(brain_tool_calls):
            manifest = self._registry.get_schema(btc.name)
            risk = manifest.risk_level if manifest else RiskLevel.MEDIUM
            (high if risk >= RiskLevel.HIGH else low).append((idx, btc))

        results: list = [None] * len(brain_tool_calls)
        blocked = False

        if low:
            tasks = [
                self._executor.dispatch(btc, session, self._on_response)
                for _, btc in low
            ]
            parallel = await asyncio.gather(*tasks, return_exceptions=True)
            for (idx, btc), r in zip(low, parallel):
                if isinstance(r, Exception):
                    results[idx] = _make_legacy_error(btc.id, btc.name, str(r))
                else:
                    results[idx] = r

        for idx, btc in high:
            if session.is_cancelled():
                results[idx] = _make_legacy_error(btc.id, btc.name, "Cancelled.")
                continue
            result = await self._executor.dispatch(btc, session, self._on_response)
            results[idx] = result

            # Detect hard block — typed check on result.blocked, no string sniffing
            if result.is_error and getattr(result, "blocked", False):
                blocked = True

        unfilled = [i for i, r in enumerate(results) if r is None]
        if unfilled:
            raise RuntimeError(f"BUG: unfilled tool result slots at indices {unfilled}")

        return results, blocked  # type: ignore[return-value]

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
        _fire_and_forget(self._memory.store(
            f"[Compact summary] {summary}",
            collection="conversations",
            metadata={
                "session_id": session.id,
                "user_id":    session.user_id,
                "type":       "compact_summary",
                "turn":       session.turn_count,
            },
        ), label="compact_summary_store")

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
        except (NeuralClawError, MemoryError, OSError) as e:
            log.warning("orchestrator.persist_failed", error=str(e), error_type=type(e).__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # Factory
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        settings,
        llm_client: BaseLLMClient,
        tool_bus: SkillBus,
        tool_registry: SkillRegistry,
        memory_manager: MemoryManager,
        on_response: Optional[Callable[[AgentResponse], None]] = None,
    ) -> "Orchestrator":
        """Create an Orchestrator from the NeuralClaw Settings object."""
        llm_config = LLMConfig(
            model=settings.default_llm_model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
        )
        orch = cls(
            llm_client=llm_client,
            llm_config=llm_config,
            tool_bus=tool_bus,
            tool_registry=tool_registry,
            memory_manager=memory_manager,
            agent_name=settings.agent.name,
            max_iterations=settings.agent.max_iterations_per_turn,
            max_turn_timeout=settings.agent.max_turn_timeout_seconds,
            on_response=on_response,
        )
        
        # Override the defaults in the raw constructor now that we have the full settings
        # Used for run_autonomous
        orch._settings = settings
        orch._executor._confirmation_timeout = settings.agent.confirmation_timeout_seconds
        return orch