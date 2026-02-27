"""
tests/unit/test_cli.py — CLI Interface Unit Tests

Tests CLIInterface command dispatch, response rendering, and
confirmation handling with mocked orchestrator and session.

Run with:
    pytest tests/unit/test_cli.py -v
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call
from io import StringIO

import pytest

from agent.response_synthesizer import AgentResponse, ResponseKind
from agent.orchestrator import TurnResult, TurnStatus
from skills.types import RiskLevel, SafetyDecision, SafetyStatus, TrustLevel


# ── Helpers ───────────────────────────────────────────────────────────────────


def make_response(
    kind: ResponseKind = ResponseKind.TEXT,
    text: str = "Hello!",
    is_final: bool = True,
    tool_call_id: str = "",
    risk_level: RiskLevel = RiskLevel.LOW,
) -> AgentResponse:
    return AgentResponse(
        kind=kind,
        text=text,
        is_final=is_final,
        tool_call_id=tool_call_id or None,
        risk_level=risk_level,
    )


def make_settings(
    provider: str = "openai",
    model: str = "gpt-4o",
    trust: str = "low",
) -> MagicMock:
    settings = MagicMock()
    settings.agent = {"name": "NeuralClaw", "version": "1.0.0", "default_trust_level": trust}
    settings.llm = {"default_provider": provider, "default_model": model}
    settings.default_llm_provider = provider
    settings.default_llm_model = model
    settings.memory = {"max_short_term_turns": 20, "chroma_persist_dir": "/tmp/chroma",
                       "sqlite_path": "/tmp/ep.db", "embedding_model": "all-MiniLM-L6-v2"}
    settings.tools = {
        "filesystem": {"allowed_paths": ["~/agent_files"]},
        "terminal": {"default_timeout_seconds": 30, "whitelist_extra": []},
    }
    return settings


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_session():
    from agent.session import Session
    session = Session.create(user_id="test_user", trust_level=TrustLevel.LOW)
    return session


@pytest.fixture
def mock_orchestrator():
    orc = MagicMock()
    response = make_response(text="Test response")
    orc.run_turn = AsyncMock(return_value=TurnResult(
        status=TurnStatus.SUCCESS,
        response=response,
        steps_taken=0,
        duration_ms=10.0
    ))
    return orc


@pytest.fixture
def mock_memory():
    mem = MagicMock()
    mem.init = AsyncMock()
    mem.close = AsyncMock()
    mem.search_all = AsyncMock(return_value={})
    return mem


@pytest.fixture
def cli(mock_session, mock_orchestrator, mock_memory):
    """Create a pre-wired CLIInterface bypassing _init_components."""
    from rich.console import Console
    from interfaces.cli import CLIInterface
    settings = make_settings()
    cli = CLIInterface(settings=settings)
    cli._session = mock_session
    cli._orchestrator = mock_orchestrator
    cli._memory = mock_memory
    cli._skill_bus = MagicMock()  # Added to fix TestCmdTools
    cli.console = Console(file=StringIO(), highlight=False)   # silent console
    return cli


# ── _dispatch routing ─────────────────────────────────────────────────────────


class TestDispatch:
    @pytest.mark.asyncio
    async def test_plain_text_routes_to_ask(self, cli):
        cli._cmd_ask = AsyncMock()
        await cli._dispatch("Hello agent!")
        cli._cmd_ask.assert_awaited_once_with("Hello agent!")

    @pytest.mark.asyncio
    async def test_slash_ask_routes_to_ask(self, cli):
        cli._cmd_ask = AsyncMock()
        await cli._dispatch("/ask Tell me about asyncio")
        cli._cmd_ask.assert_awaited_once_with("Tell me about asyncio")

    @pytest.mark.asyncio
    async def test_slash_run_routes_to_run(self, cli):
        cli._cmd_run = AsyncMock()
        await cli._dispatch("/run Research WebGPU")
        cli._cmd_run.assert_awaited_once_with("Research WebGPU")

    @pytest.mark.asyncio
    async def test_slash_status_routes_to_status(self, cli):
        cli._cmd_status = MagicMock()
        await cli._dispatch("/status")
        cli._cmd_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_slash_tools_routes_to_tools(self, cli):
        cli._cmd_tools = MagicMock()
        await cli._dispatch("/tools")
        cli._cmd_tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_slash_clear_routes_to_clear(self, cli):
        cli._cmd_clear = MagicMock()
        await cli._dispatch("/clear")
        cli._cmd_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_slash_cancel_routes_to_cancel(self, cli):
        cli._cmd_cancel = MagicMock()
        await cli._dispatch("/cancel")
        cli._cmd_cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_unknown_command_prints_error(self, cli):
        cli.console = MagicMock()
        await cli._dispatch("/nonexistent blah")
        cli.console.print.assert_called()
        call_args = str(cli.console.print.call_args)
        assert "Unknown command" in call_args or "nonexistent" in call_args

    @pytest.mark.asyncio
    async def test_empty_ask_prints_usage(self, cli):
        cli._cmd_ask = AsyncMock()
        await cli._dispatch("/ask")
        cli._cmd_ask.assert_awaited_once_with("")


# ── _cmd_ask ──────────────────────────────────────────────────────────────────


class TestCmdAsk:
    @pytest.mark.asyncio
    async def test_calls_orchestrator_run_turn(self, cli, mock_orchestrator):
        await cli._cmd_ask("What is Python?")
        mock_orchestrator.run_turn.assert_awaited_once_with(cli._session, "What is Python?")

    @pytest.mark.asyncio
    async def test_renders_response(self, cli, mock_orchestrator):
        cli._render_response = MagicMock()
        await cli._cmd_ask("Hello")
        cli._render_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_message_skips_call(self, cli, mock_orchestrator):
        await cli._cmd_ask("")
        mock_orchestrator.run_turn.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_error_response_rendered(self, cli, mock_orchestrator):
        response = make_response(
            kind=ResponseKind.ERROR, text="Something went wrong"
        )
        mock_orchestrator.run_turn.return_value = TurnResult(
            status=TurnStatus.ERROR,
            response=response,
            steps_taken=0,
            duration_ms=10.0
        )
        cli._render_response = MagicMock()
        await cli._cmd_ask("Trigger error")
        cli._render_response.assert_called_once()
        rendered = cli._render_response.call_args[0][0]
        assert rendered.kind == ResponseKind.ERROR


# ── _cmd_run ──────────────────────────────────────────────────────────────────


class TestCmdRun:
    @pytest.mark.asyncio
    async def test_empty_goal_skips(self, cli, mock_orchestrator):
        mock_orchestrator.run_autonomous = AsyncMock()
        await cli._cmd_run("")
        mock_orchestrator.run_autonomous.assert_not_called()

    @pytest.mark.asyncio
    async def test_iterates_autonomous_responses(self, cli, mock_orchestrator):
        responses = [
            make_response(kind=ResponseKind.PLAN, text="Plan here", is_final=False),
            make_response(kind=ResponseKind.PROGRESS, text="Step 1", is_final=False),
            make_response(kind=ResponseKind.TEXT, text="Done!", is_final=True),
        ]

        async def fake_autonomous(session, goal):
            for r in responses:
                yield r

        mock_orchestrator.run_autonomous = fake_autonomous
        rendered = []
        cli._render_response = lambda r: rendered.append(r)

        await cli._cmd_run("Research WebGPU")
        assert len(rendered) == 3
        assert rendered[0].kind == ResponseKind.PLAN
        assert rendered[-1].kind == ResponseKind.TEXT


# ── _cmd_trust ────────────────────────────────────────────────────────────────


class TestCmdTrust:
    @pytest.mark.asyncio
    async def test_set_to_medium(self, cli, mock_session):
        cli._session = mock_session
        await cli._cmd_trust("medium")
        assert cli._session.trust_level == TrustLevel.MEDIUM

    @pytest.mark.asyncio
    async def test_invalid_trust_level_no_change(self, cli, mock_session):
        cli._session = mock_session
        original = cli._session.trust_level
        await cli._cmd_trust("supermax")
        assert cli._session.trust_level == original

    @pytest.mark.asyncio
    async def test_high_trust_requires_confirmation(self, cli, mock_session):
        cli._session = mock_session
        with patch("interfaces.cli.Confirm.ask", return_value=True):
            await cli._cmd_trust("high")
        assert cli._session.trust_level == TrustLevel.HIGH

    @pytest.mark.asyncio
    async def test_high_trust_denied_keeps_level(self, cli, mock_session):
        cli._session = mock_session
        with patch("interfaces.cli.Confirm.ask", return_value=False):
            await cli._cmd_trust("high")
        assert cli._session.trust_level == TrustLevel.LOW


# ── _cmd_memory ───────────────────────────────────────────────────────────────


class TestCmdMemory:
    @pytest.mark.asyncio
    async def test_empty_query_skips(self, cli, mock_memory):
        await cli._cmd_memory("")
        mock_memory.search_all.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_search_all(self, cli, mock_memory):
        await cli._cmd_memory("async python")
        mock_memory.search_all.assert_awaited_once_with("async python", n_per_collection=3)

    @pytest.mark.asyncio
    async def test_no_results_prints_message(self, cli, mock_memory):
        mock_memory.search_all.return_value = {}
        cli.console = MagicMock()
        await cli._cmd_memory("obscure query")
        cli.console.print.assert_called()

    @pytest.mark.asyncio
    async def test_results_rendered_as_table(self, cli, mock_memory):
        from memory.long_term import MemoryEntry
        mock_memory.search_all.return_value = {
            "knowledge": [
                MemoryEntry(
                    id="1",
                    text="Python asyncio is great for I/O bound tasks",
                    collection="knowledge",
                    distance=0.2,
                )
            ]
        }
        cli.console = MagicMock()
        await cli._cmd_memory("asyncio")
        cli.console.print.assert_called()


# ── _cmd_clear ────────────────────────────────────────────────────────────────


class TestCmdClear:
    def test_clears_session_conversation(self, cli, mock_session):
        from brain.types import Message
        mock_session.add_user_message("Hello")
        mock_session.add_assistant_message("Hi!")
        cli._cmd_clear()
        msgs = mock_session.get_recent_messages()
        assert len(msgs) == 0


# ── _cmd_cancel ───────────────────────────────────────────────────────────────


class TestCmdCancel:
    def test_sets_cancel_on_session(self, cli, mock_session):
        """When no task is running, cancel should NOT set the cancel event (prevents stale cancel)."""
        assert not mock_session.is_cancelled()
        cli._cmd_cancel()
        # No running task → cancel should be cleared, not set
        assert not mock_session.is_cancelled()

    def test_cancel_with_running_task(self, cli, mock_session):
        """When a task IS running, cancel should set the cancel event."""
        import asyncio
        cli._running_task = asyncio.Future()  # simulate a running task
        cli._cmd_cancel()
        assert mock_session.is_cancelled()


# ── _cmd_status ───────────────────────────────────────────────────────────────


class TestCmdStatus:
    def test_prints_status(self, cli):
        cli.console = MagicMock()
        cli._cmd_status()
        cli.console.print.assert_called()


# ── _cmd_tools ────────────────────────────────────────────────────────────────


class TestCmdTools:
    def test_renders_table(self, cli):
        from skills.types import SkillManifest
        fake_manifest = SkillManifest(
            name="test_skill",
            version="1.0.0",
            description="A test skill for unit testing",
            category="testing",
            risk_level=RiskLevel.LOW,
            parameters={"properties": {"arg1": {"type": "string", "description": "test arg"}}, "required": ["arg1"]},
        )
        cli._skill_bus._registry.list_schemas.return_value = [fake_manifest]
        cli.console = MagicMock()
        cli._cmd_tools()
        cli.console.print.assert_called()

    def test_no_tools_message(self, cli):
        cli._skill_bus._registry.list_schemas.return_value = []
        cli.console = MagicMock()
        cli._cmd_tools()
        cli.console.print.assert_called()
        call_args = str(cli.console.print.call_args)
        assert "No tools" in call_args

    def test_skill_detail(self, cli):
        from skills.types import SkillManifest
        fake_manifest = SkillManifest(
            name="file_read",
            version="1.0.0",
            description="Read a file from the filesystem",
            category="filesystem",
            risk_level=RiskLevel.MEDIUM,
            parameters={"properties": {"path": {"type": "string", "description": "file path"}}, "required": ["path"]},
            capabilities=frozenset({"fs:read"}),
        )
        cli._skill_bus._registry.get_manifest.return_value = fake_manifest
        cli.console = MagicMock()
        cli._cmd_skill_detail("file_read")
        cli.console.print.assert_called()


# ── _render_response ─────────────────────────────────────────────────────────


class TestRenderResponse:
    def test_text_response(self, cli):
        cli.console = MagicMock()
        r = make_response(kind=ResponseKind.TEXT, text="Hello world")
        cli._render_response(r)
        cli.console.print.assert_called()

    def test_error_response(self, cli):
        cli.console = MagicMock()
        r = make_response(kind=ResponseKind.ERROR, text="Something failed")
        cli._render_response(r)
        cli.console.print.assert_called()

    def test_progress_response(self, cli):
        cli.console = MagicMock()
        r = make_response(kind=ResponseKind.PROGRESS, text="Step 1/3", is_final=False)
        cli._render_response(r)
        cli.console.print.assert_called()

    def test_plan_response(self, cli):
        cli.console = MagicMock()
        r = make_response(kind=ResponseKind.PLAN, text="1. Step one\n2. Step two")
        cli._render_response(r)
        cli.console.print.assert_called()

    def test_empty_text_response_skipped(self, cli):
        cli.console = MagicMock()
        r = make_response(kind=ResponseKind.TEXT, text="")
        cli._render_response(r)
        cli.console.print.assert_not_called()

    def test_confirmation_delegates(self, cli):
        cli._handle_confirmation = MagicMock()
        r = make_response(
            kind=ResponseKind.CONFIRMATION,
            text="Allow tool?",
            tool_call_id="call_1",
            risk_level=RiskLevel.HIGH,
        )
        cli._render_response(r)
        cli._handle_confirmation.assert_called_once_with(r)


# ── _handle_confirmation ──────────────────────────────────────────────────────


class TestHandleConfirmation:
    @pytest.mark.asyncio
    async def test_approved_resolves_session(self, cli, mock_session):
        mock_session.register_confirmation("call_abc")
        r = make_response(
            kind=ResponseKind.CONFIRMATION,
            text="Allow dangerous action?",
            tool_call_id="call_abc",
        )
        with patch("interfaces.cli.Confirm.ask", return_value=True):
            cli._handle_confirmation(r)
        # Future should be resolved (won't raise if done)
        fut = mock_session._pending_confirmations.get("call_abc")
        assert fut is None  # resolved and popped

    @pytest.mark.asyncio
    async def test_denied_resolves_session_false(self, cli, mock_session):
        mock_session.register_confirmation("call_xyz")
        r = make_response(
            kind=ResponseKind.CONFIRMATION,
            text="Allow dangerous action?",
            tool_call_id="call_xyz",
        )
        with patch("interfaces.cli.Confirm.ask", return_value=False):
            cli._handle_confirmation(r)
        fut = mock_session._pending_confirmations.get("call_xyz")
        assert fut is None  # resolved and popped

    def test_no_tool_call_id_no_crash(self, cli):
        r = make_response(kind=ResponseKind.CONFIRMATION, text="Confirm?", tool_call_id="")
        with patch("interfaces.cli.Confirm.ask", return_value=True):
            cli._handle_confirmation(r)  # should not raise


# ── _on_streamed_response ─────────────────────────────────────────────────────


class TestStreamedResponse:
    def test_progress_rendered(self, cli):
        cli._render_response = MagicMock()
        r = make_response(kind=ResponseKind.PROGRESS, text="Working...", is_final=False)
        cli._on_streamed_response(r)
        cli._render_response.assert_called_once_with(r)

    def test_tool_result_rendered(self, cli):
        cli._render_response = MagicMock()
        r = make_response(kind=ResponseKind.TOOL_RESULT, text="✓ file_read — contents")
        cli._on_streamed_response(r)
        cli._render_response.assert_called_once_with(r)

    def test_confirmation_handled(self, cli):
        cli._handle_confirmation = MagicMock()
        cli._render_response = MagicMock()
        r = make_response(kind=ResponseKind.CONFIRMATION, text="Confirm?", tool_call_id="c1")
        cli._on_streamed_response(r)
        cli._handle_confirmation.assert_called_once_with(r)
        cli._render_response.assert_not_called()


# ── _build_prompt ─────────────────────────────────────────────────────────────


class TestBuildPrompt:
    def test_contains_trust_level(self, cli, mock_session):
        prompt = cli._build_prompt()
        assert "low" in prompt

    def test_contains_turn_count(self, cli, mock_session):
        prompt = cli._build_prompt()
        # Turn count starts at 0
        assert "0" in prompt

    def test_prompt_suffix(self, cli):
        prompt = cli._build_prompt()
        assert ">" in prompt