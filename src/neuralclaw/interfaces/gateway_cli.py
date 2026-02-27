"""
interfaces/gateway_cli.py — Lightweight CLI over Gateway

A thin CLI REPL that connects through GatewayClient instead of wiring
its own orchestrator. Same Rich rendering and slash commands — but sends
everything over WebSocket and renders streamed responses from the gateway.

Usage:
    python main.py --interface gateway-cli
    python main.py --interface gateway-cli --gateway-url ws://remote:9090
"""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from rich import box

from neuralclaw.gateway.gateway_client import GatewayClient
from neuralclaw.gateway.protocol import MessageType
from neuralclaw.observability.logger import get_logger

log = get_logger(__name__)


_HELP_TEXT = """
## Gateway CLI Commands

| Command | Description |
|---------|-------------|
| `/ask <message>` | Send a message (default — just type normally) |
| `/run <goal>` | Start autonomous multi-step task |
| `/status` | Show session status |
| `/skills` | List available skills |
| `/trust <low\|medium\|high>` | Change trust level |
| `/grant <cap>` | Grant a capability |
| `/revoke <cap>` | Revoke a capability |
| `/cancel` | Cancel running task |
| `/help` | Show this help |
| `exit` / `Ctrl+D` | Quit |
"""


class GatewayCLI:
    """
    Thin CLI REPL that communicates through the WebSocket gateway.

    No agent imports — only knows about GatewayClient and the protocol.
    """

    def __init__(
        self,
        gateway_url: str = "ws://127.0.0.1:9090",
        auth_token: Optional[str] = None,
    ):
        self._url = gateway_url
        self._auth_token = auth_token
        self._client: Optional[GatewayClient] = None
        self._session_id: Optional[str] = None
        self._trust_level = "low"
        self.console = Console()

    async def start(self) -> None:
        """Connect to gateway and run the REPL loop."""
        self.console.print(Panel(
            Text("🐾 NeuralClaw — Gateway CLI", style="bold cyan"),
            title="NeuralClaw",
            subtitle=f"connecting to {self._url}",
            box=box.DOUBLE,
            border_style="bright_cyan",
        ))

        try:
            async with GatewayClient(self._url, self._auth_token) as client:
                self._client = client

                # Health check
                healthy = await client.ping()
                if not healthy:
                    self.console.print("[red]❌ Gateway is not responding.[/]")
                    return

                self.console.print(
                    f"[green]✓ Connected to gateway at {self._url}[/]\n"
                )

                # Create a session
                self._session_id = await client.create_session(
                    trust_level=self._trust_level
                )
                self.console.print(
                    f"[dim]Session: {self._session_id}[/]\n"
                )

                # Subscribe to unsolicited messages (confirmations)
                event_queue = client.subscribe(self._session_id)
                asyncio.create_task(self._event_listener(event_queue))

                await self._repl_loop()

        except ConnectionRefusedError:
            self.console.print(
                f"[red]❌ Cannot connect to gateway at {self._url}.\n"
                f"   Start the gateway first: python main.py --interface gateway[/]"
            )
        except Exception as e:
            self.console.print(f"[red]❌ Gateway error: {e}[/]")
            log.error("gateway_cli.error", error=str(e), exc_info=True)

    async def _event_listener(self, queue: asyncio.Queue) -> None:
        """Listen for unsolicited server messages (confirmations)."""
        while True:
            try:
                msg = await queue.get()
                if msg.type == MessageType.CONFIRM_REQUEST.value:
                    await self._handle_confirmation(msg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("gateway_cli.event_error", error=str(e))

    async def _handle_confirmation(self, msg) -> None:
        """Handle a confirmation request from the gateway."""
        data = msg.data
        skill = data.get("skill_name", "unknown")
        risk = data.get("risk_level", "HIGH")
        reason = data.get("reason", "")
        text = data.get("text", "")
        tcid = data.get("tool_call_id", "")

        risk_icon = {"HIGH": "⚠️", "CRITICAL": "🚨"}.get(risk, "❓")
        self.console.print(f"\n{risk_icon} [bold]Confirmation Required[/]")
        self.console.print(f"  Tool: [cyan]{skill}[/]  Risk: [bold]{risk}[/]")
        if reason:
            self.console.print(f"  Reason: {reason}")
        self.console.print()

        try:
            from aioconsole import ainput
            answer = await ainput("Allow? [y/N] ")
            approved = answer.strip().lower() in ("y", "yes")
        except EOFError:
            approved = False

        await self._client.confirm(self._session_id, tcid, approved)
        status = "[green]✓ Approved[/]" if approved else "[red]✗ Denied[/]"
        self.console.print(f"  {status}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # REPL
    # ─────────────────────────────────────────────────────────────────────────

    async def _repl_loop(self) -> None:
        """Main async input loop."""
        from aioconsole import ainput

        while True:
            try:
                prompt = self._build_prompt()
                raw = await ainput(prompt)
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Goodbye![/]")
                break

            raw = raw.strip()
            if not raw:
                continue
            if raw.lower() in ("exit", "quit"):
                self.console.print("[dim]Goodbye![/]")
                break

            await self._dispatch(raw)

    def _build_prompt(self) -> str:
        short_id = (self._session_id or "?")[:8]
        return f"NeuralClaw[gw][{self._trust_level}][{short_id}]> "

    async def _dispatch(self, raw: str) -> None:
        """Route input to the correct handler."""
        if raw.startswith("/"):
            parts = raw.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""

            if cmd == "/help":
                self.console.print(Markdown(_HELP_TEXT))
            elif cmd == "/ask":
                await self._cmd_ask(arg)
            elif cmd == "/run":
                await self._cmd_run(arg)
            elif cmd == "/cancel":
                await self._cmd_cancel()
            elif cmd == "/status":
                await self._cmd_status()
            elif cmd == "/skills" or cmd == "/tools":
                await self._cmd_skills()
            elif cmd == "/trust":
                await self._cmd_trust(arg)
            elif cmd == "/grant":
                await self._cmd_grant(arg)
            elif cmd == "/revoke":
                await self._cmd_revoke(arg)
            else:
                self.console.print(f"[dim]Unknown command: {cmd}. Type /help.[/]")
        else:
            await self._cmd_ask(raw)

    # ─────────────────────────────────────────────────────────────────────────
    # Commands
    # ─────────────────────────────────────────────────────────────────────────

    async def _cmd_ask(self, message: str) -> None:
        if not message:
            self.console.print("[dim]Usage: /ask <message>[/]")
            return

        self.console.print("[dim]🤔 Thinking…[/]")
        async for resp in self._client.ask(self._session_id, message):
            kind = resp.data.get("kind", "text")
            text = resp.data.get("text", "")
            is_final = resp.data.get("is_final", True)

            if kind == "text" and is_final:
                self.console.print()
                self.console.print(Markdown(text))
                self.console.print()
            elif kind == "progress":
                self.console.print(f"  [dim]{text}[/]", end="\r")
            elif kind == "tool_result":
                self.console.print(f"  {text}")
            elif kind == "error":
                self.console.print(f"  [red]{text}[/]")

    async def _cmd_run(self, goal: str) -> None:
        if not goal:
            self.console.print("[dim]Usage: /run <goal>[/]")
            return

        self.console.print(f"[cyan]🚀 Starting autonomous mode: {goal}[/]\n")
        async for resp in self._client.run(self._session_id, goal):
            kind = resp.data.get("kind", "text")
            text = resp.data.get("text", "")
            is_final = resp.data.get("is_final", True)

            if kind == "plan":
                self.console.print(Markdown(text))
                self.console.print()
            elif kind == "text" and is_final:
                self.console.print(Markdown(text))
                self.console.print()
            elif kind == "progress":
                self.console.print(f"  [dim]{text}[/]")
            elif kind == "tool_result":
                self.console.print(f"  {text}")
            elif kind == "error":
                self.console.print(f"  [red]{text}[/]")

    async def _cmd_cancel(self) -> None:
        resp = await self._client.cancel(self._session_id)
        self.console.print(f"[yellow]{resp.data.get('text', 'Cancelled.')}[/]")

    async def _cmd_status(self) -> None:
        resp = await self._client.status(self._session_id)
        data = resp.data
        self.console.print(
            f"🤖 [bold]Session Status[/]\n"
            f"  Session: {data.get('session_id', '?')}\n"
            f"  Trust: {data.get('trust_level', '?')}\n"
            f"  Turns: {data.get('turns', 0)}  ·  Tool calls: {data.get('tool_calls', 0)}\n"
            f"  Tokens: {data.get('tokens_in', 0):,} in / {data.get('tokens_out', 0):,} out"
        )

    async def _cmd_skills(self) -> None:
        resp = await self._client.list_skills(self._session_id)
        skills = resp.data.get("skills", [])
        if not skills:
            self.console.print("[dim]No skills registered.[/]")
            return
        self.console.print(f"[bold]{len(skills)} skills available:[/]")
        for s in skills:
            risk = s.get("risk_level", "LOW")
            icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(risk, "⚪")
            self.console.print(f"  {icon} {s['name']} — {s.get('description', '')[:60]}")

    async def _cmd_trust(self, level: str) -> None:
        level = level.strip().lower()
        if level not in ("low", "medium", "high"):
            self.console.print("[dim]Usage: /trust <low|medium|high>[/]")
            return
        resp = await self._client.set_trust(self._session_id, level)
        if resp.type == MessageType.ERROR.value:
            self.console.print(f"[red]{resp.data.get('message', 'Error')}[/]")
        else:
            self._trust_level = level
            self.console.print(f"[green]✓ Trust level set to {level.upper()}[/]")

    async def _cmd_grant(self, cap: str) -> None:
        cap = cap.strip()
        if not cap:
            self.console.print("[dim]Usage: /grant <capability>[/]")
            return
        resp = await self._client.grant(self._session_id, cap)
        if resp.type == MessageType.ERROR.value:
            self.console.print(f"[red]{resp.data.get('message', 'Error')}[/]")
        else:
            self.console.print(f"[green]✓ Granted: {cap}[/]")

    async def _cmd_revoke(self, cap: str) -> None:
        cap = cap.strip()
        if not cap:
            self.console.print("[dim]Usage: /revoke <capability>[/]")
            return
        await self._client.revoke(self._session_id, cap)
        self.console.print(f"[yellow]✓ Revoked: {cap}[/]")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def run_gateway_cli(
    gateway_url: str = "ws://127.0.0.1:9090",
    auth_token: Optional[str] = None,
) -> None:
    """Entry point for the gateway CLI."""
    cli = GatewayCLI(gateway_url=gateway_url, auth_token=auth_token)
    await cli.start()
