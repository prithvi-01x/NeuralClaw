"""
tests/unit/test_agent_modules.py

Tests for low-coverage agent modules:
  - agent/context_builder.py   (was 24%)
  - agent/planner.py           (was 29%)
  - agent/reasoner.py          (was 39%)
  - agent/response_synthesizer.py (was 56%)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / Factories
# ─────────────────────────────────────────────────────────────────────────────

def _mock_llm_response(content: str = "{}") -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.is_complete = True
    resp.model = "test-model"
    resp.usage = MagicMock(input_tokens=10, output_tokens=20)
    resp.finish_reason = MagicMock(value="stop")
    return resp


def _make_llm(content: str = "{}") -> AsyncMock:
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value=_mock_llm_response(content))
    return llm


def _mock_session(
    trust_level_val: str = "low",
    active_plan=None,
    history=None,
):
    from skills.types import TrustLevel
    session = MagicMock()
    session.id = "sess-test"
    session.trust_level = TrustLevel.LOW
    session.active_plan = active_plan
    session.get_recent_messages = MagicMock(return_value=history or [])
    return session


def _mock_memory_manager(context: str = "") -> AsyncMock:
    mm = AsyncMock()
    mm.build_memory_context = AsyncMock(return_value=context)
    return mm


# ─────────────────────────────────────────────────────────────────────────────
# ContextBuilder
# ─────────────────────────────────────────────────────────────────────────────

class TestContextBuilderSystemPrompt:
    def _make_builder(self):
        from agent.context_builder import ContextBuilder
        return ContextBuilder(_mock_memory_manager(), agent_name="TestAgent")

    def test_agent_name_in_prompt(self):
        cb = self._make_builder()
        session = _mock_session()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "TestAgent" in prompt

    def test_trust_level_in_prompt(self):
        cb = self._make_builder()
        session = _mock_session()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "LOW" in prompt

    def test_extra_system_appended(self):
        cb = self._make_builder()
        session = _mock_session()
        prompt = cb._build_system_prompt(session, extra_system="CUSTOM_EXTRA")
        assert "CUSTOM_EXTRA" in prompt

    def test_no_plan_block_when_no_plan(self):
        cb = self._make_builder()
        session = _mock_session(active_plan=None)
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "Active Plan" not in prompt

    def test_plan_block_included_when_plan_present(self):
        from agent.session import ActivePlan, PlanStep
        step = PlanStep(index=0, description="Do something")
        plan = ActivePlan(id="plan1", goal="My Goal", steps=[step])
        session = _mock_session(active_plan=plan)
        cb = self._make_builder()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "My Goal" in prompt
        assert "Do something" in prompt

    def test_completed_step_shows_checkmark(self):
        from agent.session import ActivePlan, PlanStep
        step = PlanStep(index=0, description="Done step", completed=True)
        plan = ActivePlan(id="plan1", goal="Goal", steps=[step], current_step_index=1)
        session = _mock_session(active_plan=plan)
        cb = self._make_builder()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "✅" in prompt

    def test_current_step_shows_arrow(self):
        from agent.session import ActivePlan, PlanStep
        step = PlanStep(index=0, description="Current step")
        plan = ActivePlan(id="plan1", goal="Goal", steps=[step], current_step_index=0)
        session = _mock_session(active_plan=plan)
        cb = self._make_builder()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "▶️" in prompt

    def test_step_with_result_summary(self):
        from agent.session import ActivePlan, PlanStep
        step = PlanStep(index=0, description="Step", completed=True, result_summary="Done OK")
        plan = ActivePlan(id="plan1", goal="Goal", steps=[step], current_step_index=1)
        session = _mock_session(active_plan=plan)
        cb = self._make_builder()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "Done OK" in prompt

    def test_step_with_error(self):
        from agent.session import ActivePlan, PlanStep
        step = PlanStep(index=0, description="Step", error="Something failed")
        plan = ActivePlan(id="plan1", goal="Goal", steps=[step])
        session = _mock_session(active_plan=plan)
        cb = self._make_builder()
        prompt = cb._build_system_prompt(session, extra_system=None)
        assert "Something failed" in prompt


class TestContextBuilderMemoryBlock:
    @pytest.mark.asyncio
    async def test_no_context_returns_empty(self):
        from agent.context_builder import ContextBuilder
        cb = ContextBuilder(_mock_memory_manager(context=""), agent_name="Agent")
        result = await cb._build_memory_block("query", "sess-1")
        assert result == ""

    @pytest.mark.asyncio
    async def test_context_wrapped_in_tags(self):
        from agent.context_builder import ContextBuilder
        cb = ContextBuilder(_mock_memory_manager(context="some memory"), agent_name="Agent")
        result = await cb._build_memory_block("query", "sess-1")
        assert "<long_term_memory>" in result
        assert "some memory" in result

    @pytest.mark.asyncio
    async def test_long_context_truncated(self):
        from agent.context_builder import ContextBuilder
        long_ctx = "x" * 10_000
        cb = ContextBuilder(_mock_memory_manager(context=long_ctx), agent_name="Agent")
        result = await cb._build_memory_block("query", "sess-1")
        assert "truncated" in result

    @pytest.mark.asyncio
    async def test_memory_exception_returns_empty(self):
        from agent.context_builder import ContextBuilder
        from exceptions import MemoryError as NeuralClawMemoryError
        mm = AsyncMock()
        mm.build_memory_context = AsyncMock(side_effect=NeuralClawMemoryError("DB down"))
        cb = ContextBuilder(mm, agent_name="Agent")
        result = await cb._build_memory_block("query", "sess-1")
        assert result == ""


class TestContextBuilderTrimHistory:
    def _make_message(self, content: str):
        from brain.types import Message
        return Message.user(content)

    def _make_builder(self):
        from agent.context_builder import ContextBuilder
        return ContextBuilder(_mock_memory_manager())

    def test_empty_history_returns_empty(self):
        cb = self._make_builder()
        assert cb._trim_history([]) == []

    def test_short_history_unchanged(self):
        cb = self._make_builder()
        msgs = [self._make_message("short") for _ in range(3)]
        result = cb._trim_history(msgs)
        assert len(result) == 3

    def test_long_history_trimmed(self):
        cb = self._make_builder()
        # 20 messages of 2000 chars each = 40000 > 20000 budget
        msgs = [self._make_message("x" * 2000) for _ in range(20)]
        result = cb._trim_history(msgs)
        total = sum(len(m.content or "") for m in result)
        assert total <= 20_000

    def test_keeps_minimum_4_messages(self):
        cb = self._make_builder()
        # 4 huge messages — can't trim below 4
        msgs = [self._make_message("x" * 8000) for _ in range(4)]
        result = cb._trim_history(msgs)
        assert len(result) == 4


class TestContextBuilderBuild:
    @pytest.mark.asyncio
    async def test_build_returns_messages(self):
        from agent.context_builder import ContextBuilder
        cb = ContextBuilder(_mock_memory_manager(), agent_name="Agent")
        session = _mock_session()
        msgs = await cb.build(session, "hello")
        assert len(msgs) >= 2  # system + user

    @pytest.mark.asyncio
    async def test_user_message_not_duplicated_if_in_history(self):
        from agent.context_builder import ContextBuilder
        from brain.types import Message
        cb = ContextBuilder(_mock_memory_manager(), agent_name="Agent")
        history = [Message.user("hello")]
        session = _mock_session(history=history)
        msgs = await cb.build(session, "hello")
        user_msgs = [m for m in msgs if m.role.value == "user"]
        assert len(user_msgs) == 1

    @pytest.mark.asyncio
    async def test_memory_block_included_when_non_empty(self):
        from agent.context_builder import ContextBuilder
        from brain.types import Role
        cb = ContextBuilder(_mock_memory_manager(context="important memory"), agent_name="Agent")
        session = _mock_session()
        msgs = await cb.build(session, "tell me")
        contents = [m.content for m in msgs]
        assert any("important memory" in (c or "") for c in contents)


# ─────────────────────────────────────────────────────────────────────────────
# Planner
# ─────────────────────────────────────────────────────────────────────────────

class TestPlannerParsePlan:
    def _make_planner(self, response_content: str = "{}"):
        from agent.planner import Planner
        from brain.types import LLMConfig
        llm = _make_llm(response_content)
        config = LLMConfig(model="test")
        return Planner(llm, config)

    def test_valid_json_plan(self):
        import json
        from agent.planner import Planner
        from brain.types import LLMConfig
        payload = json.dumps({
            "steps": ["Step one", "Step two"],
            "estimated_duration": "1 minute",
            "risk_level": "LOW",
        })
        p = self._make_planner(payload)
        result = p._parse_plan(payload)
        assert len(result.steps) == 2
        assert result.risk_level == "LOW"
        assert result.estimated_duration == "1 minute"

    def test_fallback_on_invalid_json(self):
        from agent.planner import Planner
        from brain.types import LLMConfig
        p = self._make_planner()
        result = p._parse_plan("not json at all\nStep A\nStep B")
        assert len(result.steps) >= 1

    def test_empty_steps_falls_back_to_lines(self):
        import json
        from agent.planner import Planner
        from brain.types import LLMConfig
        payload = json.dumps({"steps": []})
        p = self._make_planner()
        result = p._parse_plan(payload)
        # Falls back to line extraction — no crash
        assert isinstance(result.steps, list)

    def test_strips_markdown_fences(self):
        import json
        from agent.planner import Planner
        payload = "```json\n" + json.dumps({"steps": ["Do it"]}) + "\n```"
        p = self._make_planner()
        result = p._parse_plan(payload)
        assert "Do it" in result.steps

    def test_strips_numbered_prefixes(self):
        from agent.planner import Planner
        p = self._make_planner()
        result = p._parse_plan("1. First step\n2. Second step")
        assert all(not s[0].isdigit() for s in result.steps)

    def test_risk_level_uppercased(self):
        import json
        from agent.planner import Planner
        payload = json.dumps({"steps": ["s"], "risk_level": "medium"})
        p = self._make_planner()
        result = p._parse_plan(payload)
        assert result.risk_level == "MEDIUM"


class TestPlannerParseRecovery:
    def _make_planner(self):
        from agent.planner import Planner
        from brain.types import LLMConfig
        return Planner(_make_llm(), LLMConfig(model="test"))

    def test_valid_recovery(self):
        import json
        p = self._make_planner()
        payload = json.dumps({
            "recovery_steps": ["Retry", "Fallback"],
            "can_recover": True,
            "skip_failed_step": True,
        })
        result = p._parse_recovery(payload)
        assert result.can_recover is True
        assert result.skip_failed_step is True
        assert len(result.recovery_steps) == 2

    def test_invalid_json_returns_no_recovery(self):
        p = self._make_planner()
        result = p._parse_recovery("garbage")
        assert result.can_recover is False
        assert result.recovery_steps == []

    def test_missing_fields_use_defaults(self):
        import json
        p = self._make_planner()
        result = p._parse_recovery(json.dumps({}))
        assert result.can_recover is False


class TestPlannerCreatePlan:
    @pytest.mark.asyncio
    async def test_returns_plan_result(self):
        import json
        from agent.planner import Planner
        from brain.types import LLMConfig
        payload = json.dumps({"steps": ["A", "B"], "risk_level": "LOW"})
        p = Planner(_make_llm(payload), LLMConfig(model="test"))
        result = await p.create_plan("Do X", ["web_search", "terminal"])
        assert len(result.steps) == 2

    @pytest.mark.asyncio
    async def test_llm_failure_returns_fallback(self):
        from agent.planner import Planner
        from brain.types import LLMConfig
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("LLM down"))
        p = Planner(llm, LLMConfig(model="test"))
        result = await p.create_plan("Do X", [])
        assert isinstance(result.steps, list)
        assert len(result.steps) >= 1

    @pytest.mark.asyncio
    async def test_context_appended_to_user_message(self):
        import json
        from agent.planner import Planner
        from brain.types import LLMConfig
        payload = json.dumps({"steps": ["step"], "risk_level": "LOW"})
        llm = _make_llm(payload)
        p = Planner(llm, LLMConfig(model="test"))
        await p.create_plan("Goal", [], context="extra context here")
        call_args = llm.generate.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        user_msg = next(m for m in messages if m.role.value == "user")
        assert "extra context here" in user_msg.content


class TestPlannerCreateRecovery:
    @pytest.mark.asyncio
    async def test_returns_recovery_result(self):
        import json
        from agent.planner import Planner
        from brain.types import LLMConfig
        payload = json.dumps({"recovery_steps": ["Retry"], "can_recover": True})
        p = Planner(_make_llm(payload), LLMConfig(model="test"))
        result = await p.create_recovery("Goal", "failed step", "error msg", [])
        assert result.can_recover is True

    @pytest.mark.asyncio
    async def test_llm_failure_returns_no_recovery(self):
        from agent.planner import Planner
        from brain.types import LLMConfig
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=RuntimeError("boom"))
        p = Planner(llm, LLMConfig(model="test"))
        result = await p.create_recovery("Goal", "step", "error", [])
        assert result.can_recover is False


# ─────────────────────────────────────────────────────────────────────────────
# Reasoner
# ─────────────────────────────────────────────────────────────────────────────

class TestReasonerParseVerdict:
    def _make_reasoner(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        return Reasoner(_make_llm(), LLMConfig(model="test"))

    def test_valid_verdict_proceed_true(self):
        import json
        r = self._make_reasoner()
        payload = json.dumps({"proceed": True, "confidence": 0.9, "reasoning": "Looks fine"})
        v = r._parse_verdict(payload)
        assert v.proceed is True
        assert v.confidence == 0.9
        assert v.reasoning == "Looks fine"

    def test_valid_verdict_proceed_false_with_concern(self):
        import json
        r = self._make_reasoner()
        payload = json.dumps({"proceed": False, "confidence": 0.3, "reasoning": "Risky", "concern": "Dangerous"})
        v = r._parse_verdict(payload)
        assert v.proceed is False
        assert v.concern == "Dangerous"

    def test_invalid_json_returns_safe_default(self):
        r = self._make_reasoner()
        v = r._parse_verdict("not json")
        assert v.proceed is True
        assert v.confidence == 0.5

    def test_strips_fences(self):
        import json
        r = self._make_reasoner()
        payload = "```json\n" + json.dumps({"proceed": True, "confidence": 0.8, "reasoning": "ok"}) + "\n```"
        v = r._parse_verdict(payload)
        assert v.proceed is True

    def test_is_confident_threshold(self):
        from agent.reasoner import EvalVerdict
        v_confident = EvalVerdict(proceed=True, confidence=0.7, reasoning="ok")
        v_not = EvalVerdict(proceed=True, confidence=0.69, reasoning="ok")
        assert v_confident.is_confident is True
        assert v_not.is_confident is False


class TestReasonerEvaluateToolCall:
    @pytest.mark.asyncio
    async def test_returns_verdict(self):
        import json
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        payload = json.dumps({"proceed": True, "confidence": 0.9, "reasoning": "fine"})
        r = Reasoner(_make_llm(payload), LLMConfig(model="test"))
        v = await r.evaluate_tool_call("web_search", {"query": "test"}, "Find something")
        assert v.proceed is True

    @pytest.mark.asyncio
    async def test_llm_failure_defaults_to_proceed(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        from brain.llm_client import LLMError
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=LLMError("oops"))
        r = Reasoner(llm, LLMConfig(model="test"))
        v = await r.evaluate_tool_call("terminal", {}, "Do it")
        assert v.proceed is True
        assert v.confidence == 0.5


class TestReasonerThink:
    @pytest.mark.asyncio
    async def test_returns_string(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        r = Reasoner(_make_llm("Some reasoning here"), LLMConfig(model="test"))
        result = await r.think("Should I do X?")
        assert result == "Some reasoning here"

    @pytest.mark.asyncio
    async def test_with_context(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        llm = _make_llm("reasoning")
        r = Reasoner(llm, LLMConfig(model="test"))
        await r.think("question", context="some context")
        call_args = llm.generate.call_args
        messages = call_args[1]["messages"] if "messages" in call_args[1] else call_args[0][0]
        user_msg = next(m for m in messages if m.role.value == "user")
        assert "some context" in user_msg.content

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        from brain.llm_client import LLMError
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=LLMError("fail"))
        r = Reasoner(llm, LLMConfig(model="test"))
        result = await r.think("question")
        assert result == ""


class TestReasonerReflect:
    @pytest.mark.asyncio
    async def test_returns_lesson(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        r = Reasoner(_make_llm("Lesson learned."), LLMConfig(model="test"))
        result = await r.reflect("Goal", ["step1", "step2"], "success")
        assert result == "Lesson learned."

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        from agent.reasoner import Reasoner
        from brain.types import LLMConfig
        from brain.llm_client import LLMError
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=LLMError("fail"))
        r = Reasoner(llm, LLMConfig(model="test"))
        result = await r.reflect("Goal", [], "outcome")
        assert result == ""


# ─────────────────────────────────────────────────────────────────────────────
# ResponseSynthesizer
# ─────────────────────────────────────────────────────────────────────────────

class TestResponseSynthesizerFromLLM:
    def _make_synth(self):
        from agent.response_synthesizer import ResponseSynthesizer
        return ResponseSynthesizer()

    def test_text_kind(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.from_llm(_mock_llm_response("Hello!"))
        assert resp.kind == ResponseKind.TEXT
        assert resp.text == "Hello!"

    def test_metadata_populated(self):
        s = self._make_synth()
        resp = s.from_llm(_mock_llm_response("Hi"))
        assert resp.metadata["model"] == "test-model"
        assert resp.metadata["tokens_in"] == 10

    def test_is_final_reflects_llm_complete(self):
        s = self._make_synth()
        llm_resp = _mock_llm_response("text")
        llm_resp.is_complete = False
        resp = s.from_llm(llm_resp)
        assert resp.is_final is False


class TestResponseSynthesizerToolResults:
    def _make_synth(self):
        from agent.response_synthesizer import ResponseSynthesizer
        return ResponseSynthesizer()

    def _make_tool_result(self, name: str, content: str, success: bool = True):
        from skills.types import SkillResult, RiskLevel
        return SkillResult.ok(
            skill_name=name,
            skill_call_id="call-1",
            output=content,
            duration_ms=42.0,
        )

    def test_tool_success(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        result = self._make_tool_result("web_search", "some results")
        resp = s.tool_success(result)
        assert resp.kind == ResponseKind.TOOL_RESULT
        assert "web_search" in resp.text
        assert "✅" in resp.text

    def test_tool_success_with_summary(self):
        s = self._make_synth()
        result = self._make_tool_result("web_search", "content")
        resp = s.tool_success(result, summary="Found 5 results")
        assert "Found 5 results" in resp.text

    def test_tool_error(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        result = self._make_tool_result("terminal", "error output")
        resp = s.tool_error(result)
        assert resp.kind == ResponseKind.ERROR
        assert "❌" in resp.text
        assert "terminal" in resp.text

    def test_tool_progress(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.tool_progress("terminal", step=3, total=10, detail="running")
        assert resp.kind == ResponseKind.PROGRESS
        assert resp.is_final is False
        assert "3/10" in resp.text

    def test_tool_progress_no_detail(self):
        s = self._make_synth()
        resp = s.tool_progress("tool", 1, 5)
        assert "1/5" in resp.text


class TestResponseSynthesizerConfirmation:
    def _make_synth(self):
        from agent.response_synthesizer import ResponseSynthesizer
        return ResponseSynthesizer()

    def _make_confirmation_request(self, risk_level_str: str = "HIGH", arguments: dict = None):
        from skills.types import ConfirmationRequest, RiskLevel
        return ConfirmationRequest(
            skill_name="terminal",
            skill_call_id="call-123",
            risk_level=RiskLevel.HIGH if risk_level_str == "HIGH" else RiskLevel.CRITICAL,
            reason="Dangerous command",
            arguments=arguments or {},
        )

    def test_confirmation_kind(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        cr = self._make_confirmation_request(arguments={"cmd": "rm -rf /"})
        resp = s.confirmation_request(cr)
        assert resp.kind == ResponseKind.CONFIRMATION
        assert resp.is_final is False
        assert resp.tool_call_id == "call-123"

    def test_confirmation_includes_tool_args(self):
        s = self._make_synth()
        cr = self._make_confirmation_request(arguments={"path": "/tmp/file"})
        resp = s.confirmation_request(cr)
        assert "path" in resp.text

    def test_critical_risk_icon(self):
        s = self._make_synth()
        cr = self._make_confirmation_request("CRITICAL")
        resp = s.confirmation_request(cr)
        assert "🚨" in resp.text

    def test_high_risk_icon(self):
        s = self._make_synth()
        cr = self._make_confirmation_request("HIGH")
        resp = s.confirmation_request(cr)
        assert "⚠️" in resp.text

    def test_long_arg_value_clipped(self):
        s = self._make_synth()
        long_val = "x" * 200
        cr = self._make_confirmation_request(arguments={"key": long_val})
        resp = s.confirmation_request(cr)
        assert "…" in resp.text


class TestResponseSynthesizerPlanAndStatus:
    def _make_synth(self):
        from agent.response_synthesizer import ResponseSynthesizer
        return ResponseSynthesizer()

    def test_plan_preview(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.plan_preview("My Goal", ["Step 1", "Step 2"])
        assert resp.kind == ResponseKind.PLAN
        assert "My Goal" in resp.text
        assert "Step 1" in resp.text
        assert resp.is_final is False

    def test_error_response(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.error("Something went wrong", detail="traceback here")
        assert resp.kind == ResponseKind.ERROR
        assert "Something went wrong" in resp.text
        assert "traceback here" in resp.text

    def test_error_without_detail(self):
        s = self._make_synth()
        resp = s.error("oops")
        assert "oops" in resp.text

    def test_cancelled(self):
        s = self._make_synth()
        resp = s.cancelled()
        assert "cancelled" in resp.text.lower() or "🛑" in resp.text

    def test_thinking(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.thinking()
        assert resp.is_final is False

    def test_info(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        resp = s.info("Just a heads up")
        assert "Just a heads up" in resp.text
        assert resp.is_final is False

    def test_progress_bar_full(self):
        s = self._make_synth()
        bar = s._bar(10, 10)
        assert "░" not in bar
        assert "█" * 10 == bar

    def test_progress_bar_empty(self):
        s = self._make_synth()
        bar = s._bar(0, 10)
        assert "█" not in bar

    def test_progress_bar_zero_total(self):
        s = self._make_synth()
        bar = s._bar(0, 0)
        assert len(bar) == 10

    def test_status_response(self):
        from agent.response_synthesizer import ResponseKind
        s = self._make_synth()
        session = MagicMock()
        session.active_plan = None
        session.status_summary = MagicMock(return_value={
            "session_id": "sess-1",
            "trust_level": "low",
            "turns": 5,
            "tool_calls": 3,
            "tokens_in": 100,
            "tokens_out": 200,
            "uptime_seconds": 60,
        })
        resp = s.status(session)
        assert resp.kind == ResponseKind.STATUS
        assert "sess-1" in resp.text

    def test_status_with_plan(self):
        s = self._make_synth()
        session = MagicMock()
        plan = MagicMock()
        plan.goal = "Big goal"
        plan.progress_summary = "1/3 steps complete"
        session.active_plan = plan
        session.status_summary = MagicMock(return_value={
            "session_id": "s",
            "trust_level": "high",
            "turns": 1,
            "tool_calls": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "uptime_seconds": 1,
        })
        resp = s.status(session)
        assert "Big goal" in resp.text


class TestAgentResponseStr:
    def test_str_returns_text(self):
        from agent.response_synthesizer import AgentResponse, ResponseKind
        r = AgentResponse(kind=ResponseKind.TEXT, text="hello world")
        assert str(r) == "hello world"