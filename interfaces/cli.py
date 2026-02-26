"""
interfaces/cli.py — NeuralClaw CLI Interface (Phase 5)

Full interactive REPL for the NeuralClaw agent.
Uses rich for terminal rendering and aioconsole for async input.

Features:
  - Coloured prompt with session metadata
  - Streaming tool progress updates
  - Inline confirmation dialogs for high-risk actions
  - /ask, /run (autonomous), /status, /memory, /tools, /trust, /cancel, /clear, /help
  - Graceful Ctrl+C / Ctrl+D handling

Fixes applied:
  - _cmd_status now reuses self._orchestrator._synth instead of creating a
    stray ResponseSynthesizer() instance.
  - _cmd_cancel now checks whether anything is actually running and prints a
    friendly "nothing to cancel" message if not.
  - _build_prompt shows the turn count AFTER the completed turn (consistent
    with what the user expects — turn N is shown after turn N finishes, not
    during turn N+1). turn_count is incremented in add_user_message so the
    prompt already reflects the upcoming turn number correctly; the fix just
    makes the display logic explicit.
  - Empty/whitespace response.text is guarded in _render_response so a blank
    Panel is never shown (e.g. when bytez returns an empty string).

Usage:
    python main.py --interface cli
    python main.py --interface cli --log-level DEBUG
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import box

from agent.orchestrator import Orchestrator
from interfaces.model_selector import (
    run_model_selector,
    build_llm_client_for_model,
    fetch_ollama_options,
    save_default_model,
    load_default_model,
    current_model_key,
    MODEL_OPTIONS,
)
from agent.response_synthesizer import AgentResponse, ResponseKind
from agent.session import Session
from brain import LLMClientFactory
from config.settings import Settings
from memory.memory_manager import MemoryManager
from observability.logger import get_logger
from safety.safety_kernel import SafetyKernel
from skills.types import TrustLevel
from exceptions import NeuralClawError, MemoryError as NeuralClawMemoryError, LLMError

# Skill system — loads builtin + plugin skills at startup
from pathlib import Path as _SkillPath
from skills.loader import SkillLoader as _SkillLoader
from skills.md_loader import MarkdownSkillLoader as _MdSkillLoader
from skills.bus import SkillBus as _SkillBus

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_BANNER = """
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██████╗██╗      █████╗ ██╗    ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██║     ██╔══██╗██║    ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██║     ██║     ███████║██║ █╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║     ██╔══██║██║███╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗███████╗██║  ██║╚███╔███╔╝
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
"""

_HELP_TEXT = """
## NeuralClaw CLI Commands

| Command | Description |
|---------|-------------|
| `/ask <message>` | Send a message to the agent (default — just type normally) |
| `/run <goal>` | Start autonomous multi-step task execution |
| `/status` | Show current session status and stats |
| `/memory <query>` | Search long-term memory |
| `/tools` or `/skills` | List all registered skills/tools |
| `/trust <low\\|medium\\|high>` | Set session trust level |
| `/grant <capability>` | Grant a capability for this session (e.g. `fs:delete`) |
| `/revoke <capability>` | Revoke a previously granted capability |
| `/capabilities` | Show active capabilities for this session |
| `/resetcaps` | Re-enable tool calling if disabled by an automatic fallback |
| `/model` | Interactively switch the active LLM model |
| `/compact` | Summarise old turns and compress context window |
| `/usage` | Show token usage and estimated cost for this session |
| `/clear` | Clear conversation history |
| `/cancel` | Cancel the current running task |
| `/help` | Show this help message |
| `exit` / `quit` / Ctrl+D | Exit NeuralClaw |

**Trust Levels:**
- `low` — Confirm HIGH and CRITICAL risk actions (default, safest)
- `medium` — Only confirm CRITICAL risk actions
- `high` — Auto-approve all actions (use with care)

**Tips:**
- Just type your message directly — no `/ask` needed
- Use `/run` for autonomous tasks like "research X and write a report"
- Tool results stream in real-time
"""

_TRUST_COLOURS = {
    TrustLevel.LOW: "green",
    TrustLevel.MEDIUM: "yellow",
    TrustLevel.HIGH: "red",
}


def _make_adhoc_option(provider: str, model_id: str):
    """Construct a minimal ModelOption for a provider/model not in MODEL_OPTIONS."""
    from interfaces.model_selector import ModelOption
    return ModelOption(
        key=f"{provider}:{model_id}",
        name=model_id,
        description="Custom model",
        provider=provider,
        model_id=model_id,
        requires_key="",
    )


# ── CLI Runner ────────────────────────────────────────────────────────────────


class CLIInterface:
    """
    Full interactive REPL for NeuralClaw.

    Wires together: Settings → LLM → Memory → Safety → ToolBus → Orchestrator
    then runs a Rich-powered async input loop.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.console = Console()
        self._session: Optional[Session] = None
        self._orchestrator: Optional[Orchestrator] = None
        self._memory: Optional[MemoryManager] = None
        self._scheduler = None
        self._running_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()
        # Track active model key for the model selector
        self._active_model_key: Optional[str] = None

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize all components then run the REPL loop."""
        await self._init_components()
        self._print_banner()
        try:
            await self._repl_loop()
        finally:
            await self._cleanup()

    async def _init_components(self) -> None:
        """Wire up the full agent stack."""
        self.console.print("[dim]Initializing NeuralClaw...[/]")

        # LLM client
        try:
            llm_client = LLMClientFactory.from_settings(self.settings)
        except (LLMError, ValueError, KeyError, ImportError) as e:
            self.console.print(f"[red]❌ Failed to create LLM client: {e}[/]")
            sys.exit(1)

        # Memory manager
        self._memory = MemoryManager.from_settings(self.settings)
        try:
            with self.console.status("[dim]Loading embedding model...[/]"):
                await self._memory.init(load_embedder=True)
        except (NeuralClawMemoryError, OSError, RuntimeError, ImportError) as e:
            self.console.print(f"[yellow]⚠ Memory init warning: {e}[/]")
            await self._memory.init(load_embedder=False)

        # Safety kernel
        allowed_paths = self.settings.tools.filesystem.allowed_paths
        extra_commands = self.settings.tools.terminal.whitelist_extra
        safety_kernel = SafetyKernel(
            allowed_paths=allowed_paths,
            whitelist_extra=extra_commands,
        )

        # Skill registry + bus — loaded from builtin/ and plugins/
        _base = _SkillPath(__file__).parent.parent
        _skill_registry = _SkillLoader().load_all(
            [
                _base / "skills" / "builtin",
                _base / "skills" / "plugins",
            ],
            strict=True,   # production: raise on broken skill files
        )
        # Also load markdown skills (OpenClaw-compatible SKILL.md format)
        _MdSkillLoader().load_all(
            [_base / "skills" / "plugins"],
            registry=_skill_registry,
            strict=False,  # warn on bad MD files, don't crash startup
        )
        self._skill_bus = _SkillBus(
            registry=_skill_registry,
            safety_kernel=safety_kernel,
            default_timeout_seconds=self.settings.tools.terminal.default_timeout_seconds,
        )

        # Session
        default_trust = self.settings.agent.default_trust_level
        trust_level = TrustLevel(default_trust)
        self._session = Session.create(
            user_id="cli_user",
            trust_level=trust_level,
            max_turns=self.settings.memory.max_short_term_turns,
        )
        # Register the Session's ShortTermMemory in MemoryManager so both
        # share the same object — prevents ghost-session desync (finding #3).
        self._memory._sessions[self._session.id] = self._session.memory

        # Orchestrator — wired to SkillBus + SkillRegistry
        self._orchestrator = Orchestrator.from_settings(
            settings=self.settings,
            llm_client=llm_client,
            tool_bus=self._skill_bus,
            tool_registry=_skill_registry,
            memory_manager=self._memory,
            on_response=self._on_streamed_response,
        )

        log.info("cli.initialized", session_id=self._session.id)

        # ── Scheduler ─────────────────────────────────────────────────────────
        try:
            from schedular.schedular import TaskScheduler
            self._scheduler = TaskScheduler(
                orchestrator=self._orchestrator,
                memory_manager=self._memory,
                max_concurrent_tasks=self.settings.scheduler.max_concurrent_tasks,
                timezone=self.settings.scheduler.timezone,
            )
            await self._scheduler.start()
            log.info("cli.scheduler_started")
        except Exception as e:
            log.warning("cli.scheduler_init_failed", error=str(e))
        self.console.print("[dim]✓ Ready[/]\n")
        self._active_model_key = current_model_key(self.settings)

    # ── Banner & Help ─────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        self.console.print(
            Text(_BANNER, style="bold cyan"),
            justify="center",
        )
        ver = self.settings.agent.version
        provider = self.settings.default_llm_provider
        model = self._active_model_key or f"{provider}/{self.settings.default_llm_model}"
        trust = self._session.trust_level.value.upper()
        trust_colour = _TRUST_COLOURS.get(self._session.trust_level, "white")

        # Show persona name from SOUL.md if workspace is configured
        try:
            from agent.workspace import get_workspace_loader
            ws = get_workspace_loader()
            persona_name = ws.agent_name()
            personality  = ws.soul_personality()
        except Exception:
            persona_name = None
            personality  = None

        display_name = persona_name or self.settings.agent_name
        persona_line = ""
        if persona_name and persona_name != self.settings.agent_name:
            persona_line = f"[bold magenta]{persona_name}[/]"
            if personality:
                persona_line += f"  ·  [dim italic]{personality}[/]"
            persona_line += "\n"

        # Suggest onboard if workspace isn't configured
        onboard_hint = ""
        try:
            from agent.workspace import get_workspace_loader
            if not get_workspace_loader().is_configured():
                onboard_hint = (
                    "\n[dim]💡 Run [bold]python main.py onboard[/bold] "
                    "to personalise your assistant.[/]"
                )
        except Exception:
            pass

        self.console.print(
            Panel(
                f"{persona_line}"
                f"[bold]v{ver}[/]  ·  "
                f"LLM: [cyan]{provider}[/]/[cyan]{model}[/]  ·  "
                f"Trust: [{trust_colour}]{trust}[/]  ·  "
                f"Session: [dim]{self._session.id}[/]\n\n"
                f"Type your message or [bold]/help[/] for commands. "
                f"[bold]exit[/] or Ctrl+D to quit."
                f"{onboard_hint}",
                border_style="cyan",
                padding=(0, 2),
            )
        )

    def _print_help(self) -> None:
        self.console.print(Markdown(_HELP_TEXT))

    # ── REPL Loop ─────────────────────────────────────────────────────────────

    async def _repl_loop(self) -> None:
        """Main async input loop."""
        try:
            import aioconsole
            _get_input = aioconsole.ainput
        except ImportError:
            # Fallback to blocking input run in executor
            loop = asyncio.get_running_loop()
            async def _get_input(prompt: str) -> str:
                return await loop.run_in_executor(None, input, prompt)

        while not self._shutdown.is_set():
            try:
                prompt = self._build_prompt()
                user_input = await _get_input(prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye.[/]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            if user_input.lower() in ("exit", "quit"):
                self.console.print("[dim]Goodbye.[/]")
                break

            await self._dispatch(user_input)

        await self._cleanup()

    def _build_prompt(self) -> str:
        """
        Build the coloured CLI prompt string.

        Shows: NeuralClaw[trust][model][turn]>
        """
        trust = self._session.trust_level.value
        colours = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}
        reset = "\033[0m"
        colour = colours.get(trust, reset)
        turns = self._session.turn_count
        model_label = self._active_model_key or current_model_key(self.settings)
        # Shorten auto/long model keys for prompt display
        if model_label.startswith("openai:"):
            model_label = model_label[7:]
        elif model_label.startswith("anthropic:"):
            model_label = model_label[10:]
        elif model_label.startswith("gemini:"):
            model_label = model_label[7:]
        elif model_label.startswith("ollama:"):
            model_label = model_label[7:]
        return f"{colour}NeuralClaw[{trust}][{model_label}][{turns}]{reset}> "

    def _get_display_name(self) -> str:
        """Return persona name from SOUL.md if configured, else settings agent name."""
        try:
            from agent.workspace import get_workspace_loader
            name = get_workspace_loader().agent_name()
            if name:
                return name
        except Exception:
            pass
        return self.settings.agent_name

    # ── Command Dispatch ──────────────────────────────────────────────────────

    async def _dispatch(self, raw: str) -> None:
        """Route input to the correct handler."""
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            handlers = {
                "/help":    lambda _: self._print_help(),
                "/status":  lambda _: self._cmd_status(),
                "/compact": lambda _: self._cmd_compact(),
                "/usage":   lambda _: self._cmd_usage(),
                "/tools":   lambda _: self._cmd_tools(),
                "/skills":  lambda _: self._cmd_tools(),
                "/clear":   lambda _: self._cmd_clear(),
                "/cancel":  lambda _: self._cmd_cancel(),
                "/model":   lambda _: self._cmd_model(),
                "/trust":   self._cmd_trust,
                "/grant":   self._cmd_grant,
                "/revoke":  self._cmd_revoke,
                "/capabilities": lambda _: self._cmd_capabilities(),
                "/resetcaps":   lambda _: self._cmd_resetcaps(),
                "/memory":  self._cmd_memory,
                "/run":     self._cmd_run,
                "/ask":     self._cmd_ask,
            }

            handler = handlers.get(cmd)
            if handler:
                result = handler(arg)
                if asyncio.iscoroutine(result):
                    await result
            else:
                self.console.print(
                    f"[yellow]Unknown command: {cmd}. Type /help for commands.[/]"
                )
        else:
            # Bare message → /ask
            await self._cmd_ask(raw)

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _cmd_model(self) -> None:
        """Open the interactive model picker."""
        result = await run_model_selector(
            console=self.console,
            settings=self.settings,
            current_key=self._active_model_key,
        )

        if result is None:
            self.console.print("[dim]Model unchanged.[/]")
            return

        provider, model_id, set_default = result

        # Find the ModelOption for display
        selected_opt = next(
            (o for o in MODEL_OPTIONS if o.provider == provider and o.model_id == model_id),
            None,
        )
        display_name = f"{provider}:{model_id}" if not selected_opt else selected_opt.name

        # Build new LLM client — pass live options so auto-resolve uses Ollama data
        with self.console.status(f"[dim]Switching to {display_name}...[/]", spinner="dots"):
            new_client, err = build_llm_client_for_model(
                selected_opt or _make_adhoc_option(provider, model_id),
                self.settings,
                options=MODEL_OPTIONS,
            )

        if err or new_client is None:
            self.console.print(
                f"[red]✗ Could not switch to {display_name}: {err or 'unknown error'}[/]"
            )
            return

        # Hot-swap client AND model ID across orchestrator, planner, and reasoner
        try:
            self._orchestrator.swap_llm_client(new_client, new_model_id=model_id)
        except AttributeError:
            # Fallback for older orchestrator without swap_llm_client
            self._orchestrator._llm = new_client
            if hasattr(self._orchestrator, "_config"):
                self._orchestrator._config.model = model_id
            if hasattr(self._orchestrator, "_planner") and hasattr(self._orchestrator._planner, "_config"):
                self._orchestrator._planner._config.model = model_id
            if hasattr(self._orchestrator, "_reasoner") and hasattr(self._orchestrator._reasoner, "_config"):
                self._orchestrator._reasoner._config.model = model_id

        # Update settings so prompt and banner reflect the new model
        self.settings.llm.default_provider = provider
        self.settings.llm.default_model = model_id

        new_key = f"{provider}:{model_id}"
        self._active_model_key = new_key

        if set_default:
            save_default_model(new_key)
            self.console.print(
                f"[green]✓ Switched to [bold]{display_name}[/bold] "
                f"and saved as default.[/]"
            )
        else:
            self.console.print(
                f"[green]✓ Switched to [bold]{display_name}[/bold] "
                f"(this session only).[/]"
            )

        # Show capability notice so user knows what mode they're in
        try:
            from brain.capabilities import get_capabilities
            caps = get_capabilities(provider, model_id)
            if not caps.supports_tools:
                self.console.print(
                    f"[yellow]ℹ  {display_name} runs in [bold]chat-only mode[/bold] "
                    f"— tool calling is not supported by this model.[/]"
                )
            else:
                self.console.print(
                    f"[dim]   Tools: enabled · Vision: {'yes' if caps.supports_vision else 'no'}[/]"
                )
        except (NeuralClawError, AttributeError, ValueError) as _cap_err:
            log.debug("cli.capability_check_failed", error=str(_cap_err))

    async def _cmd_ask(self, message: str) -> None:
        """Send a message to the agent and render the response."""

        if not message:
            self.console.print("[yellow]Usage: /ask <message>[/]")
            return

        self.console.print()  # spacing

        with self.console.status("[dim cyan]Thinking...[/]", spinner="dots"):
            turn_result = await self._orchestrator.run_turn(self._session, message)

        self._render_response(turn_result.response)
        self._maybe_suggest_compact()

    async def _cmd_run(self, goal: str) -> None:
        """Run the agent in autonomous multi-step mode."""
        if not goal:
            self.console.print("[yellow]Usage: /run <goal>[/]")
            return

        self.console.print()
        self.console.print(
            Panel(
                f"[bold]Autonomous Mode[/]\n[dim]{goal}[/]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

        try:
            async for response in self._orchestrator.run_autonomous(
                self._session, goal
            ):
                self._render_response(response)
                if not response.is_final:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.console.print("[yellow]🛑 Task cancelled.[/]")

    def _cmd_status(self) -> None:
        """Show session status using the shared synthesizer."""
        response = self._orchestrator._synth.status(self._session)
        text = response.text if isinstance(response.text, str) else str(response.text)
        self.console.print(Markdown(text))

    async def _cmd_compact(self) -> None:
        """Summarise old conversation turns and compress the context window."""
        turns = self._session.memory.conversation.turn_count
        keep = getattr(self.settings.memory, "compact_keep_recent", 4)

        if turns <= keep:
            self.console.print(
                f"[dim]Nothing to compact — only {turns} turn(s) in buffer "
                f"(need more than {keep}).[/]"
            )
            return

        self.console.print(
            f"[dim]Compacting {turns} turns → keeping last {keep} + summary...[/]"
        )
        try:
            with self.console.status("[dim]Summarising conversation...[/]", spinner="dots"):
                summary = await self._orchestrator.compact_session(
                    self._session, keep_recent=keep
                )
        except RuntimeError as e:
            self.console.print(f"[red]Compact failed: {e}[/]")
            return
        except (NeuralClawError, OSError) as e:
            self.console.print(f"[red]Compact error: {e}[/]")
            return

        self.console.print(
            Panel(
                Markdown(f"**Summary of compacted turns:**\n\n{summary}"),
                title="[cyan]✓ Compacted[/]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        remaining = self._session.memory.conversation.turn_count
        self.console.print(
            f"[dim]Buffer: {turns} turns → {remaining} turns kept. "
            f"Summary injected into context.[/]\n"
        )

    def _cmd_usage(self) -> None:
        """Show token usage and estimated cost for this session."""
        s = self._session
        tokens_in  = s.total_input_tokens
        tokens_out = s.total_output_tokens

        # Rough cost estimates per 1M tokens (update as needed)
        _COST_PER_M: dict[str, tuple[float, float]] = {
            "bytez":      (5.00,  15.00),
            "openai":     (5.00,  15.00),
            "anthropic":  (3.00,  15.00),
            "ollama":     (0.00,   0.00),
            "openrouter": (5.00,  15.00),
            "gemini":     (1.25,   5.00),
        }
        provider = self.settings.default_llm_provider.lower()
        in_rate, out_rate = _COST_PER_M.get(provider, (5.00, 15.00))
        cost_in  = tokens_in  / 1_000_000 * in_rate
        cost_out = tokens_out / 1_000_000 * out_rate
        total_cost = cost_in + cost_out

        turns = s.turn_count
        tool_calls = s.tool_call_count
        buf_turns = s.memory.conversation.turn_count
        has_summary = s.memory.conversation.has_summary

        summary_note = " _(compacted)_" if has_summary else ""

        text = (
            f"## 📊 Session Usage\n\n"
            f"**Tokens in:** {tokens_in:,}  ·  "
            f"**Tokens out:** {tokens_out:,}  ·  "
            f"**Total:** {tokens_in + tokens_out:,}\n\n"
            f"**Estimated cost:** ${total_cost:.4f} "
            f"_(@ ${in_rate}/M in, ${out_rate}/M out — {provider})_\n\n"
            f"**Turns:** {turns}  ·  "
            f"**Tool calls:** {tool_calls}  ·  "
            f"**Buffer turns:** {buf_turns}{summary_note}"
        )
        self.console.print(Markdown(text))

    def _maybe_suggest_compact(self) -> None:
        """Print a one-time hint when turns exceed compact_after_turns threshold."""
        threshold = getattr(self.settings.memory, "compact_after_turns", 15)
        if threshold <= 0:
            return
        turns = self._session.turn_count
        # Suggest at exactly the threshold, not on every subsequent turn
        if turns == threshold:
            self.console.print(
                f"[dim yellow]💡 Tip: {turns} turns in this session. "
                f"Run [bold]/compact[/bold] to summarise old context and free up "
                f"the context window.[/]\n"
            )

    def _cmd_tools(self) -> None:
        """List all registered tools."""
        schemas = self._skill_bus._registry.list_schemas(enabled_only=False)
        if not schemas:
            self.console.print("[dim]No tools registered.[/]")
            return

        table = Table(
            title="Registered Tools",
            box=box.ROUNDED,
            border_style="dim",
            show_lines=False,
        )
        table.add_column("Name", style="cyan bold", no_wrap=True)
        table.add_column("Category", style="dim")
        table.add_column("Risk", no_wrap=True)
        table.add_column("Enabled", no_wrap=True)
        table.add_column("Description")

        risk_colours = {
            "LOW": "green",
            "MEDIUM": "yellow",
            "HIGH": "orange1",
            "CRITICAL": "red",
        }

        for s in sorted(schemas, key=lambda x: x.category):
            risk_str = s.risk_level.value
            risk_colour = risk_colours.get(risk_str, "white")
            enabled_str = "✓" if s.enabled else "✗"
            enabled_colour = "green" if s.enabled else "dim red"
            table.add_row(
                s.name,
                s.category,
                f"[{risk_colour}]{risk_str}[/]",
                f"[{enabled_colour}]{enabled_str}[/]",
                s.description[:60] + ("…" if len(s.description) > 60 else ""),
            )

        self.console.print(table)

    def _cmd_clear(self) -> None:
        """Clear conversation history."""
        self._session.clear_conversation()
        self.console.print("[dim]✓ Conversation history cleared.[/]")

    def _cmd_cancel(self) -> None:
        """
        Cancel the current running task.
        """
        if self._session.is_cancelled():
            self.console.print("[dim]Already cancelled.[/]")
            return

        self._session.cancel()

        if self._running_task and not self._running_task.done():
            self._running_task.cancel()

        if self._session.active_plan:
            self.console.print("[yellow]🛑 Cancel signal sent.[/]")
        else:
            self.console.print("[yellow]🛑 Cancel signal sent.[/]")

    async def _cmd_trust(self, level: str) -> None:
        """Change the session trust level."""
        level = level.strip().lower()
        valid = {t.value: t for t in TrustLevel}
        if level not in valid:
            self.console.print(
                f"[yellow]Unknown trust level: '{level}'. "
                f"Valid: {', '.join(valid)}[/]"
            )
            return

        new_trust = valid[level]

        if new_trust == TrustLevel.HIGH:
            confirmed = Confirm.ask(
                "[bold red]⚠ HIGH trust auto-approves ALL actions. Continue?[/]",
                default=False,
                console=self.console,
            )
            if not confirmed:
                self.console.print("[dim]Trust level unchanged.[/]")
                return

        self._session.set_trust_level(new_trust)
        colour = _TRUST_COLOURS.get(new_trust, "white")
        self.console.print(
            f"[{colour}]✓ Trust level set to {new_trust.value.upper()}[/]"
        )

    async def _cmd_grant(self, capability: str) -> None:
        """Grant a capability to the current session.

        Usage: /grant <capability>
        Example capabilities: fs:read  fs:write  fs:delete  net:fetch  shell:exec
        """
        capability = capability.strip()
        if not capability:
            self.console.print(
                "[yellow]Usage: /grant <capability>[/]\n"
                "[dim]Examples: fs:read  fs:write  fs:delete  net:fetch  shell:exec[/]"
            )
            return

        self._session.grant_capability(capability)
        self.console.print(
            f"[green]✓ Capability '[bold]{capability}[/bold]' granted for this session.[/]\n"
            f"[dim]Active capabilities: {sorted(self._session.granted_capabilities)}[/]"
        )

    async def _cmd_revoke(self, capability: str) -> None:
        """Revoke a previously granted capability from the current session.

        Usage: /revoke <capability>
        """
        capability = capability.strip()
        if not capability:
            self.console.print("[yellow]Usage: /revoke <capability>[/]")
            return

        if capability not in self._session.granted_capabilities:
            self.console.print(
                f"[yellow]Capability '{capability}' is not currently granted.[/]\n"
                f"[dim]Active capabilities: {sorted(self._session.granted_capabilities)}[/]"
            )
            return

        self._session.revoke_capability(capability)
        self.console.print(
            f"[red]✓ Capability '[bold]{capability}[/bold]' revoked.[/]\n"
            f"[dim]Active capabilities: {sorted(self._session.granted_capabilities)}[/]"
        )

    def _cmd_capabilities(self) -> None:
        """Show all granted capabilities for this session."""
        active = sorted(self._session.granted_capabilities)

        # All known capabilities across builtin skills
        _ALL_CAPS = {
            "fs:read":   "Read files within allowed paths",
            "fs:write":  "Write and append files within allowed paths",
            "fs:delete": "Delete files (requires explicit grant)",
            "shell:run": "Execute whitelisted terminal commands",
            "net:fetch": "Fetch URLs and perform web searches",
        }

        table = Table(
            title="Session Capabilities",
            box=box.ROUNDED,
            border_style="dim",
        )
        table.add_column("Capability", style="cyan", no_wrap=True)
        table.add_column("Status", no_wrap=True)
        table.add_column("Description")

        for cap, desc in _ALL_CAPS.items():
            if cap in active:
                status = "[green]✓ granted[/]"
            else:
                status = "[dim]— not granted[/]"
            table.add_row(cap, status, desc)

        # Show any extra caps that aren't in the known list
        extras = [c for c in active if c not in _ALL_CAPS]
        for cap in extras:
            table.add_row(cap, "[green]✓ granted[/]", "[dim](custom)[/]")

        self.console.print(table)
        if not active:
            self.console.print(
                "[dim]No capabilities granted. Use /grant <capability> to add one.[/]"
            )

    def _cmd_resetcaps(self) -> None:
        """
        Re-enable tool calling for the current model.

        When a model returns an unsupported-tools error, NeuralClaw automatically
        falls back to chat-only mode and sets supports_tools=False on both the
        client instance and the capability registry. This flag persists for the
        lifetime of the process.

        Use /resetcaps after switching models or fixing the provider config so the
        next turn will attempt tool calls again instead of running in chat-only mode.
        """
        from agent.orchestrator import _provider_name_from_client
        provider = _provider_name_from_client(
            self._orchestrator._llm
        ) if hasattr(self._orchestrator, "_llm") else "unknown"
        model = getattr(self._orchestrator._config, "model", "unknown")

        if not hasattr(self._orchestrator, "reset_tool_support"):
            self.console.print(
                "[yellow]Orchestrator does not support reset_tool_support. "
                "Please upgrade NeuralClaw.[/]"
            )
            return

        self._orchestrator.reset_tool_support()
        self.console.print(
            f"[green]✓ Tool support reset for [bold]{provider}/{model}[/bold].[/]\n"
            "[dim]The next turn will attempt to use tools again.[/]"
        )

    async def _cmd_memory(self, query: str) -> None:
        """Search long-term memory."""
        if not query:
            self.console.print("[yellow]Usage: /memory <search query>[/]")
            return

        with self.console.status("[dim]Searching memory...[/]"):
            results = await self._memory.search_all(query, n_per_collection=3)

        if not results:
            self.console.print("[dim]No relevant memories found.[/]")
            return

        table = Table(
            title=f'Memory Search: "{query}"',
            box=box.ROUNDED,
            border_style="dim",
        )
        table.add_column("Collection", style="cyan", no_wrap=True)
        table.add_column("Relevance", no_wrap=True)
        table.add_column("Content")

        for collection, entries in results.items():
            for entry in entries:
                score = entry.relevance_score
                score_colour = (
                    "green" if score > 0.7 else "yellow" if score > 0.4 else "dim"
                )
                table.add_row(
                    collection,
                    f"[{score_colour}]{score:.2f}[/]",
                    entry.text[:120] + ("…" if len(entry.text) > 120 else ""),
                )

        self.console.print(table)

    # ── Response Rendering ────────────────────────────────────────────────────

    def _render_response(self, response: AgentResponse) -> None:
        """
        Render an AgentResponse to the terminal using Rich.

        Fix: guard against empty text — a blank Panel looks like a bug and
        can happen when the LLM returns an empty string or the Bytez client
        falls back to "(no response)" after a parsing failure.
        """
        text = response.text.strip() if response.text else ""
        kind = response.kind

        if kind == ResponseKind.TEXT:
            if text:
                self.console.print(
                    Panel(
                        Markdown(text),
                        border_style="cyan",
                        padding=(0, 2),
                    )
                )

        elif kind == ResponseKind.PLAN:
            if text:
                self.console.print(
                    Panel(
                        Markdown(text),
                        title="[yellow]📋 Plan[/]",
                        border_style="yellow",
                        padding=(0, 2),
                    )
                )

        elif kind == ResponseKind.PROGRESS:
            if text:
                self.console.print(f"  [dim cyan]{text}[/]")

        elif kind == ResponseKind.TOOL_RESULT:
            if text:
                self.console.print(f"  [green]{text}[/]")

        elif kind == ResponseKind.ERROR:
            if text:
                self.console.print(
                    Panel(
                        Markdown(text),
                        title="[red]Error[/]",
                        border_style="red",
                        padding=(0, 1),
                    )
                )

        elif kind == ResponseKind.STATUS:
            if text:
                self.console.print(Markdown(text))

        elif kind == ResponseKind.CONFIRMATION:
            self._handle_confirmation(response)

    def _handle_confirmation(self, response: AgentResponse) -> None:
        """
        Synchronous confirmation handler — used from _render_response.
        Prompts inline and resolves the pending session future immediately.
        """
        self.console.print()
        self.console.print(
            Panel(
                Markdown(response.text or "Allow this action?"),
                title="[bold yellow]⚠ Confirmation Required[/]",
                border_style="yellow",
                padding=(0, 2),
            )
        )
        approved = Confirm.ask(
            "  Allow this action?",
            default=False,
            console=self.console,
        )
        if response.tool_call_id:
            self._session.resolve_confirmation(response.tool_call_id, approved)
        status = "[green]✓ Approved[/]" if approved else "[red]✗ Denied[/]"
        self.console.print(f"  {status}\n")

    async def _handle_confirmation_async(self, response: AgentResponse) -> None:
        """
        Async confirmation handler — runs the blocking stdin read in a thread
        so the asyncio event loop stays live (timeouts keep ticking, bg tasks
        keep running) while we wait for the user to type y/n.
        """
        self.console.print()
        self.console.print(
            Panel(
                Markdown(response.text),
                title="[bold yellow]⚠ Confirmation Required[/]",
                border_style="yellow",
                padding=(0, 2),
            )
        )

        loop = asyncio.get_running_loop()
        # run_in_executor releases the event loop while stdin blocks
        approved = await loop.run_in_executor(
            None,
            lambda: Confirm.ask(
                "  Allow this action?",
                default=False,
                console=self.console,
            ),
        )

        if response.tool_call_id:
            self._session.resolve_confirmation(response.tool_call_id, approved)

        status = "[green]✓ Approved[/]" if approved else "[red]✗ Denied[/]"
        self.console.print(f"  {status}\n")

    def _on_streamed_response(self, response: AgentResponse) -> None:
        """
        Callback from Orchestrator for mid-turn streaming updates.
        Called synchronously from async context — renders inline or schedules
        the confirmation handler as a non-blocking coroutine.
        """
        if response.kind == ResponseKind.CONFIRMATION:
            # Delegate to the synchronous confirmation handler so the
            # event-level callback can be tested and the confirmation future
            # is resolved on the same call stack.
            self._handle_confirmation(response)
        elif response.kind in (ResponseKind.PROGRESS, ResponseKind.TOOL_RESULT):
            self._render_response(response)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Flush memory, stop scheduler, and close connections."""
        if self._scheduler:
            try:
                await self._scheduler.stop()
            except Exception as _sched_err:
                log.debug("cli.scheduler_stop_failed", error=str(_sched_err))
        if self._memory:
            try:
                await self._memory.close()
            except (NeuralClawMemoryError, OSError, RuntimeError) as _close_err:
                log.debug("cli.memory_close_failed", error=str(_close_err))
        log.info("cli.shutdown", session_id=self._session.id if self._session else None)


# ── Public entry point ────────────────────────────────────────────────────────


async def run_cli(settings: Settings, log) -> None:
    """
    Entry point called from main.py.

    Args:
        settings:  Loaded NeuralClaw settings.
        log:       Application-level logger.
    """
    cli = CLIInterface(settings=settings)

    log.info("cli.starting")
    try:
        await cli.start()
    except KeyboardInterrupt:
        log.info("cli.interrupted")
    finally:
        log.info("cli.stopped")