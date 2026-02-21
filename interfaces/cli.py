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
from agent.response_synthesizer import AgentResponse, ResponseKind
from agent.session import Session
from brain import LLMClientFactory
from config.settings import Settings
from memory.memory_manager import MemoryManager
from observability.logger import get_logger
from safety.safety_kernel import SafetyKernel
from tools.tool_registry import registry as global_registry
from tools.tool_bus import ToolBus
from tools.types import TrustLevel

# Import tools so they self-register on the global registry
import tools.filesystem  # noqa: F401
import tools.terminal    # noqa: F401
import tools.search      # noqa: F401

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
| `/tools` | List all registered tools |
| `/trust <low\|medium\|high>` | Set session trust level |
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
        self._running_task: Optional[asyncio.Task] = None
        self._shutdown = asyncio.Event()

    # ── Startup ───────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize all components then run the REPL loop."""
        await self._init_components()
        self._print_banner()
        await self._repl_loop()

    async def _init_components(self) -> None:
        """Wire up the full agent stack."""
        self.console.print("[dim]Initializing NeuralClaw...[/]")

        # LLM client
        try:
            llm_client = LLMClientFactory.from_settings(self.settings)
        except Exception as e:
            self.console.print(f"[red]❌ Failed to create LLM client: {e}[/]")
            sys.exit(1)

        # Memory manager
        self._memory = MemoryManager.from_settings(self.settings)
        try:
            with self.console.status("[dim]Loading embedding model...[/]"):
                await self._memory.init(load_embedder=True)
        except Exception as e:
            self.console.print(f"[yellow]⚠ Memory init warning: {e}[/]")
            await self._memory.init(load_embedder=False)

        # Safety kernel
        allowed_paths = self.settings.tools.get("filesystem", {}).get(
            "allowed_paths", ["~/agent_files"]
        )
        extra_commands = self.settings.tools.get("terminal", {}).get(
            "whitelist_extra", []
        )
        safety_kernel = SafetyKernel(
            allowed_paths=allowed_paths,
            extra_allowed_commands=extra_commands,
        )

        # Tool bus
        tool_bus = ToolBus(
            registry=global_registry,
            safety_kernel=safety_kernel,
            timeout_seconds=self.settings.tools.get("terminal", {}).get(
                "default_timeout_seconds", 30
            ),
        )

        # Session
        default_trust = self.settings.agent.get("default_trust_level", "low")
        trust_level = TrustLevel(default_trust)
        self._session = Session.create(
            user_id="cli_user",
            trust_level=trust_level,
            max_turns=self.settings.memory.get("max_short_term_turns", 20),
        )

        # Orchestrator — with streaming callback
        self._orchestrator = Orchestrator.from_settings(
            settings=self.settings,
            llm_client=llm_client,
            tool_bus=tool_bus,
            tool_registry=global_registry,
            memory_manager=self._memory,
            on_response=self._on_streamed_response,
        )

        log.info("cli.initialized", session_id=self._session.id)
        self.console.print("[dim]✓ Ready[/]\n")

    # ── Banner & Help ─────────────────────────────────────────────────────────

    def _print_banner(self) -> None:
        self.console.print(
            Text(_BANNER, style="bold cyan"),
            justify="center",
        )
        ver = self.settings.agent.get("version", "1.0.0")
        provider = self.settings.default_llm_provider
        model = self.settings.default_llm_model
        trust = self._session.trust_level.value.upper()
        trust_colour = _TRUST_COLOURS.get(self._session.trust_level, "white")

        self.console.print(
            Panel(
                f"[bold]v{ver}[/]  ·  "
                f"LLM: [cyan]{provider}[/]/[cyan]{model}[/]  ·  "
                f"Trust: [{trust_colour}]{trust}[/]  ·  "
                f"Session: [dim]{self._session.id}[/]\n\n"
                f"Type your message or [bold]/help[/] for commands. "
                f"[bold]exit[/] or Ctrl+D to quit.",
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
            loop = asyncio.get_event_loop()
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
        """Build the coloured CLI prompt string."""
        trust = self._session.trust_level.value
        colours = {"low": "\033[32m", "medium": "\033[33m", "high": "\033[31m"}
        reset = "\033[0m"
        colour = colours.get(trust, reset)
        turns = self._session.turn_count
        return f"{colour}NeuralClaw[{trust}][{turns}]{reset}> "

    # ── Command Dispatch ──────────────────────────────────────────────────────

    async def _dispatch(self, raw: str) -> None:
        """Route input to the correct handler."""
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            handlers = {
                "/help": lambda _: self._print_help(),
                "/status": lambda _: self._cmd_status(),
                "/tools": lambda _: self._cmd_tools(),
                "/clear": lambda _: self._cmd_clear(),
                "/cancel": lambda _: self._cmd_cancel(),
                "/trust": self._cmd_trust,
                "/memory": self._cmd_memory,
                "/run": self._cmd_run,
                "/ask": self._cmd_ask,
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

    async def _cmd_ask(self, message: str) -> None:
        """Send a message to the agent and render the response."""
        if not message:
            self.console.print("[yellow]Usage: /ask <message>[/]")
            return

        self.console.print()  # spacing

        with self.console.status("[dim cyan]Thinking...[/]", spinner="dots"):
            response = await self._orchestrator.run_turn(self._session, message)

        self._render_response(response)

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
                # Small breathing room between streaming updates
                if not response.is_final:
                    await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.console.print("[yellow]🛑 Task cancelled.[/]")

    def _cmd_status(self) -> None:
        """Show session status."""
        from agent.response_synthesizer import ResponseSynthesizer
        synth = ResponseSynthesizer()
        response = synth.status(self._session)
        self.console.print(Markdown(response.text))

    def _cmd_tools(self) -> None:
        """List all registered tools."""
        schemas = global_registry.list_schemas(enabled_only=False)
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
        """Cancel the current running task."""
        self._session.cancel()
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
        """Render an AgentResponse to the terminal using Rich."""

        kind = response.kind

        if kind == ResponseKind.TEXT:
            if response.text:
                self.console.print(
                    Panel(
                        Markdown(response.text),
                        border_style="cyan",
                        padding=(0, 2),
                    )
                )

        elif kind == ResponseKind.PLAN:
            self.console.print(
                Panel(
                    Markdown(response.text),
                    title="[yellow]📋 Plan[/]",
                    border_style="yellow",
                    padding=(0, 2),
                )
            )

        elif kind == ResponseKind.PROGRESS:
            # Inline — no panel, just a status line
            self.console.print(f"  [dim cyan]{response.text}[/]")

        elif kind == ResponseKind.TOOL_RESULT:
            self.console.print(f"  [green]{response.text}[/]")

        elif kind == ResponseKind.ERROR:
            self.console.print(
                Panel(
                    Markdown(response.text),
                    title="[red]Error[/]",
                    border_style="red",
                    padding=(0, 1),
                )
            )

        elif kind == ResponseKind.STATUS:
            self.console.print(Markdown(response.text))

        elif kind == ResponseKind.CONFIRMATION:
            self._handle_confirmation(response)

    def _handle_confirmation(self, response: AgentResponse) -> None:
        """
        Display a confirmation request and resolve it via the session.
        Blocks until user responds (or times out in background).
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

        approved = Confirm.ask(
            "  Allow this action?",
            default=False,
            console=self.console,
        )

        if response.tool_call_id:
            self._session.resolve_confirmation(response.tool_call_id, approved)

        status = "[green]✓ Approved[/]" if approved else "[red]✗ Denied[/]"
        self.console.print(f"  {status}\n")

    def _on_streamed_response(self, response: AgentResponse) -> None:
        """
        Callback from Orchestrator for mid-turn streaming updates.
        Called synchronously from async context — just renders inline.
        """
        # For confirmation requests we handle them here directly
        if response.kind == ResponseKind.CONFIRMATION:
            self._handle_confirmation(response)
        elif response.kind in (ResponseKind.PROGRESS, ResponseKind.TOOL_RESULT):
            self._render_response(response)

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def _cleanup(self) -> None:
        """Flush memory and close connections."""
        if self._memory:
            try:
                await self._memory.close()
            except Exception:
                pass
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