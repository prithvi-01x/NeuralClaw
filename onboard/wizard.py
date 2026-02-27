"""
onboard/wizard.py — NeuralClaw Interactive Setup Wizard

Guides a first-time user through:
    1. Naming their assistant (persona)
    2. Introducing themselves (user context)
    3. Choosing an LLM provider + entering API key
    4. Selecting a model
    5. Optional Telegram bot token
    6. LLM health-check
    7. Writing workspace files (SOUL.md, USER.md, HEARTBEAT.md, MEMORY.md)
    8. Writing .env with gathered secrets
    9. Confirming config.yaml agent.name

Usage:
    python main.py onboard
    neuralclaw onboard
"""

from __future__ import annotations

import os
import sys
import asyncio
from pathlib import Path
from typing import Optional


# ── Rich is optional; fall back to plain print if not installed ──────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich import print as rprint
    _RICH = True
except ImportError:
    _RICH = False

    class Console:  # type: ignore[no-redef]
        def print(self, *a, **kw):
            text = " ".join(str(x) for x in a)
            import re
            text = re.sub(r"\[/?[a-zA-Z_ ]+\]", "", text)
            print(text)
        def rule(self, title="", **kw):
            print(f"\n{'─' * 20} {title} {'─' * 20}")

    class Panel:  # type: ignore[no-redef]
        def __init__(self, text, **kw):
            self._t = text
        def __str__(self):
            return str(self._t)

    class Prompt:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt, default=None, password=False):
            suffix = f" [{default}]" if default else ""
            raw = input(f"{prompt}{suffix}: ").strip()
            return raw or default or ""

    class Confirm:  # type: ignore[no-redef]
        @staticmethod
        def ask(prompt, default=True):
            suffix = " [Y/n]" if default else " [y/N]"
            raw = input(f"{prompt}{suffix}: ").strip().lower()
            if not raw:
                return default
            return raw in ("y", "yes")


# ── Constants ─────────────────────────────────────────────────────────────────

WORKSPACE_DIR = Path("~/neuralclaw").expanduser()
PROJECT_ROOT  = Path(__file__).parent.parent
ENV_FILE      = PROJECT_ROOT / ".env"
CONFIG_FILE   = PROJECT_ROOT / "config" / "config.yaml"

PROVIDERS = {
    "openai":    ("OpenAI (GPT-4o, GPT-4-turbo)", "OPENAI_API_KEY",    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]),
    "anthropic": ("Anthropic (Claude)",             "ANTHROPIC_API_KEY", ["claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5"]),
    "gemini":    ("Google Gemini",                  "GEMINI_API_KEY",    ["gemini-1.5-pro", "gemini-1.5-flash"]),
    "ollama":    ("Ollama (local, free)",            None,                ["llama3.2", "mistral", "qwen2.5", "codellama"]),
    "bytez":     ("Bytez",                          "BYTEZ_API_KEY",     ["openai/gpt-4o", "anthropic/claude-sonnet-4-5"]),
}

from interfaces.branding import NEURALCLAW_BANNER as BANNER


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ask(console: Console, prompt: str, default: str = "", password: bool = False) -> str:
    if _RICH:
        return Prompt.ask(f"[cyan]{prompt}[/cyan]", default=default or None, password=password) or default
    else:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        return raw or default


def _confirm(console: Console, prompt: str, default: bool = True) -> bool:
    if _RICH:
        return Confirm.ask(f"[cyan]{prompt}[/cyan]", default=default)
    else:
        suffix = " [Y/n]" if default else " [y/N]"
        raw = input(f"{prompt}{suffix}: ").strip().lower()
        if not raw:
            return default
        return raw in ("y", "yes")


def _step(console: Console, n: int, total: int, title: str) -> None:
    console.rule(f"[bold white]Step {n}/{total} — {title}[/bold white]" if _RICH else f"Step {n}/{total} — {title}")


def _ok(console: Console, msg: str) -> None:
    console.print(f"[green]✓[/green] {msg}" if _RICH else f"✓ {msg}")


def _warn(console: Console, msg: str) -> None:
    console.print(f"[yellow]⚠[/yellow]  {msg}" if _RICH else f"⚠  {msg}")


def _err(console: Console, msg: str) -> None:
    console.print(f"[red]✗[/red] {msg}" if _RICH else f"✗ {msg}")


# ── Workspace file writers ────────────────────────────────────────────────────

def _write_soul(name: str, personality: str, workspace_dir: Path) -> None:
    content = f"""# {name} — Identity & Soul

## Name
{name}

## Personality
{personality}

## Core Principles
- I am honest about my capabilities and limitations.
- I act, I don't just plan — I complete tasks end-to-end.
- I remember what I'm told and build on it over time.
- I ask before acting on anything irreversible or high-risk.
- I keep responses concise and useful.

## Communication Style
Friendly, direct, and efficient. I use markdown when it aids readability.
I report errors honestly and suggest next steps when something fails.

## What I Am
A local-first autonomous AI assistant running on the user's own machine.
I have access to tools: web browsing, terminal commands, file management,
web search, and any skills installed in the workspace.
"""
    (workspace_dir / "SOUL.md").write_text(content)


def _write_user(name: str, context: str, workspace_dir: Path) -> None:
    content = f"""# About the User

## Name
{name}

## Context
{context}

## Preferences
- Prefers concise answers unless asked for detail.
- Trusts the assistant to make reasonable decisions autonomously.
- Wants to be notified of anything HIGH or CRITICAL risk before execution.

## Notes
(The assistant will add notes here as it learns about the user over time.)
"""
    (workspace_dir / "USER.md").write_text(content)


def _write_heartbeat(agent_name: str, workspace_dir: Path) -> None:
    content = f"""# Heartbeat Checklist for {agent_name}

Run this checklist every 30 minutes to stay proactive and useful.

## Check
- [ ] Are there any overdue reminders? (personal_reminder_set action=check)
- [ ] Are there any pending tasks that haven't been followed up on?
- [ ] Is there anything in MEMORY.md that should be acted on today?
- [ ] Are there any files or processes that need attention?

## Review
- Scan recent conversation history for unfinished requests.
- If a task was started but not confirmed complete, follow up.

## Distill
- If any important new facts were learned about the user, add them to USER.md.
- If any new workflow or preference was discovered, note it in MEMORY.md.

## Rules
- Only send a Telegram/notification message if there is something genuinely actionable.
- Never spam the user with empty check-ins.
- Keep heartbeat messages short — one or two lines maximum.
"""
    (workspace_dir / "HEARTBEAT.md").write_text(content)


def _write_memory(workspace_dir: Path) -> None:
    content = """# Long-Term Memory

This file contains important facts, preferences, and learned context
about the user and their environment. The assistant updates this file
automatically as it learns. You can also edit it directly.

## Learned Facts
(empty — will be populated as the assistant learns)

## Workflows
(empty — add recurring tasks or automations here)

## Important Credentials & Config Notes
(empty — add notes about your environment, not secrets)
"""
    (workspace_dir / "MEMORY.md").write_text(content)


# ── .env writer ───────────────────────────────────────────────────────────────

def _update_env(env_path: Path, updates: dict[str, str]) -> None:
    """Merge new key=value pairs into .env, preserving existing lines."""
    existing: dict[str, str] = {}
    lines: list[str] = []

    if env_path.exists():
        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key, _, val = stripped.partition("=")
                existing[key.strip()] = val.strip()
            lines.append(line)

    # Update existing keys in-place, append new ones
    written = set()
    new_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in updates:
                new_lines.append(f"{key}={updates[key]}")
                written.add(key)
                continue
        new_lines.append(line)

    for key, val in updates.items():
        if key not in written:
            new_lines.append(f"{key}={val}")

    env_path.write_text("\n".join(new_lines) + "\n")


# ── config.yaml patcher ───────────────────────────────────────────────────────

def _patch_config(config_path: Path, provider: str, model: str, agent_name: str) -> None:
    """Patch config.yaml with the chosen provider, model, and agent name."""
    if not config_path.exists():
        return

    text = config_path.read_text()
    import re

    # Patch agent name
    text = re.sub(
        r'(name:\s*)"[^"]*"',
        f'\\1"{agent_name}"',
        text,
        count=1,
    )

    # Patch default_provider
    text = re.sub(
        r'(default_provider:\s*)"?[^\n"]*"?',
        f'\\1"{provider}"',
        text,
        count=1,
    )

    # Patch default_model
    text = re.sub(
        r'(default_model:\s*)"?[^\n"]*"?',
        f'\\1"{model}"',
        text,
        count=1,
    )

    config_path.write_text(text)


# ── LLM health check ──────────────────────────────────────────────────────────

async def _health_check(provider: str, model: str, api_key: Optional[str]) -> bool:
    """Quick reachability test — instantiate the client and call health_check()."""
    try:
        if provider == "openai":
            from brain.openai_client import OpenAIClient
            client = OpenAIClient(api_key=api_key or "")
        elif provider == "anthropic":
            from brain.anthropic_client import AnthropicClient
            client = AnthropicClient(api_key=api_key or "")
        elif provider == "gemini":
            from brain.gemini_client import GeminiClient
            client = GeminiClient(api_key=api_key or "")
        elif provider == "bytez":
            from brain.bytez_client import BytezClient
            client = BytezClient(api_key=api_key or "")
        elif provider == "ollama":
            from brain.ollama_client import OllamaClient
            client = OllamaClient(api_key="")
        else:
            return False
        return await client.health_check()
    except Exception:
        return False


# ── Main wizard ───────────────────────────────────────────────────────────────

async def run_onboard(skip_health_check: bool = False) -> int:
    """
    Run the full interactive onboard wizard.
    Returns 0 on success, 1 on abort.
    """
    console = Console()
    TOTAL_STEPS = 7

    if _RICH:
        console.print(Panel(
            Text(BANNER, style="bold cyan"),
            subtitle="[dim]Setup Wizard[/dim]",
            border_style="cyan",
        ))
    else:
        print(BANNER)
        print("=" * 60)
        print("  NeuralClaw Setup Wizard")
        print("=" * 60)

    console.print()
    console.print("  Welcome! This wizard sets up NeuralClaw in about 2 minutes.")
    console.print("  Your answers are stored locally — nothing is sent anywhere.")
    console.print()

    collected: dict[str, str] = {}

    # ── Step 1: Persona name ─────────────────────────────────────────────────
    _step(console, 1, TOTAL_STEPS, "Name Your Assistant")
    console.print()
    console.print("  What would you like to call your assistant?")
    console.print("  [dim]Examples: Aria, Claw, Max, Nova, Atlas[/dim]" if _RICH else
                  "  Examples: Aria, Claw, Max, Nova, Atlas")
    console.print()
    agent_name = _ask(console, "Assistant name", default="NeuralClaw")
    if not agent_name.strip():
        agent_name = "NeuralClaw"
    agent_name = agent_name.strip()
    _ok(console, f"Your assistant will be called [bold]{agent_name}[/bold]" if _RICH else
        f"Your assistant will be called {agent_name}")

    console.print()
    console.print("  Describe {0}'s personality in a sentence:".format(agent_name))
    console.print("  [dim]Examples: 'Calm and methodical', 'Enthusiastic and concise'[/dim]"
                  if _RICH else "  Examples: 'Calm and methodical', 'Enthusiastic and concise'")
    personality = _ask(console, "Personality", default="Helpful, direct, and honest")
    collected["AGENT_NAME"] = agent_name
    collected["AGENT_PERSONALITY"] = personality

    # ── Step 2: About the user ───────────────────────────────────────────────
    _step(console, 2, TOTAL_STEPS, "About You")
    console.print()
    your_name = _ask(console, "Your name", default="")
    user_context = _ask(
        console,
        "In one sentence, what do you mainly use your computer for?",
        default="Software development and research",
    )
    collected["USER_NAME"] = your_name
    collected["USER_CONTEXT"] = user_context
    _ok(console, "Got it — {0} will remember this about you.".format(agent_name))

    # ── Step 3: LLM provider ─────────────────────────────────────────────────
    _step(console, 3, TOTAL_STEPS, "Choose LLM Provider")
    console.print()

    provider_list = list(PROVIDERS.keys())
    for i, (p_key, (p_label, _, _)) in enumerate(PROVIDERS.items(), 1):
        console.print(f"  [bold]{i}[/bold]. {p_label}" if _RICH else f"  {i}. {p_label}")

    console.print()
    while True:
        choice_raw = _ask(console, "Provider number", default="1")
        try:
            choice_idx = int(choice_raw) - 1
            if 0 <= choice_idx < len(provider_list):
                break
        except ValueError:
            pass
        _err(console, "Please enter a number between 1 and {0}.".format(len(provider_list)))

    provider_key = provider_list[choice_idx]
    p_label, env_var, default_models = PROVIDERS[provider_key]
    _ok(console, f"Provider: [bold]{p_label}[/bold]" if _RICH else f"Provider: {p_label}")
    collected["LLM_PROVIDER"] = provider_key

    # ── Step 4: API key ───────────────────────────────────────────────────────
    api_key: Optional[str] = None
    if env_var:
        console.print()
        existing_key = os.environ.get(env_var, "")
        if existing_key:
            console.print(f"  [dim]{env_var} is already set in your environment.[/dim]"
                          if _RICH else f"  {env_var} is already set.")
            if _confirm(console, f"Use existing {env_var}?", default=True):
                api_key = existing_key
            else:
                existing_key = ""

        if not existing_key:
            console.print(f"  Enter your [bold]{env_var}[/bold]:" if _RICH else
                          f"  Enter your {env_var}:")
            console.print("  [dim](Input hidden — not stored in shell history)[/dim]"
                          if _RICH else "  (Input hidden)")
            api_key = _ask(console, env_var, password=True)
            if not api_key:
                _warn(console, f"{env_var} left blank. You can add it to .env later.")

        if api_key:
            collected[env_var] = api_key
    else:
        console.print()
        console.print("  [dim]Ollama runs locally — no API key needed.[/dim]"
                      if _RICH else "  Ollama runs locally — no API key needed.")
        console.print("  Make sure Ollama is running: [bold]ollama serve[/bold]"
                      if _RICH else "  Make sure Ollama is running: ollama serve")

    # ── Step 4b: Model ────────────────────────────────────────────────────────
    console.print()
    console.print("  Available models for this provider:")
    for i, m in enumerate(default_models, 1):
        console.print(f"    [bold]{i}[/bold]. {m}" if _RICH else f"    {i}. {m}")
    console.print(f"    [bold]{len(default_models)+1}[/bold]. Enter custom model name"
                  if _RICH else f"    {len(default_models)+1}. Enter custom model name")
    console.print()

    while True:
        model_choice = _ask(console, "Model number or name", default="1")
        try:
            m_idx = int(model_choice) - 1
            if 0 <= m_idx < len(default_models):
                chosen_model = default_models[m_idx]
                break
            elif m_idx == len(default_models):
                chosen_model = _ask(console, "Custom model name", default="")
                if chosen_model:
                    break
                _err(console, "Model name cannot be empty.")
        except ValueError:
            # Typed a model name directly
            chosen_model = model_choice.strip()
            if chosen_model:
                break
        _err(console, "Invalid selection.")

    _ok(console, f"Model: [bold]{chosen_model}[/bold]" if _RICH else f"Model: {chosen_model}")
    collected["LLM_MODEL"] = chosen_model

    # ── Step 5: Telegram (optional) ───────────────────────────────────────────
    _step(console, 5, TOTAL_STEPS, "Telegram Bot (Optional)")
    console.print()
    console.print("  Connecting a Telegram bot lets you message {0} from your phone.".format(agent_name))
    console.print("  Create a bot at [link=https://t.me/BotFather]t.me/BotFather[/link] to get a token."
                  if _RICH else "  Create a bot at https://t.me/BotFather to get a token.")
    console.print()

    if _confirm(console, "Set up Telegram now?", default=False):
        tg_token = _ask(console, "Telegram bot token (from @BotFather)", password=True)
        if tg_token:
            collected["TELEGRAM_BOT_TOKEN"] = tg_token
            tg_user_id = _ask(console, "Your Telegram user ID (from @userinfobot, optional)", default="")
            if tg_user_id:
                collected["TELEGRAM_AUTHORIZED_USER_IDS"] = tg_user_id
            _ok(console, "Telegram configured. Start with: [bold]./run.sh telegram[/bold]"
                if _RICH else "Telegram configured. Start with: ./run.sh telegram")
    else:
        console.print("  [dim]Skipped — you can add TELEGRAM_BOT_TOKEN to .env later.[/dim]"
                      if _RICH else "  Skipped.")

    # ── Step 6: LLM health check ──────────────────────────────────────────────
    _step(console, 6, TOTAL_STEPS, "Testing Connection")
    console.print()

    if skip_health_check:
        _warn(console, "Health check skipped (--skip-health-check).")
    else:
        console.print(f"  Testing connection to [bold]{p_label}[/bold]…"
                      if _RICH else f"  Testing connection to {p_label}…")
        try:
            ok = await asyncio.wait_for(
                _health_check(provider_key, chosen_model, api_key),
                timeout=15.0,
            )
        except asyncio.TimeoutError:
            ok = False

        if ok:
            _ok(console, "Connection successful!")
        else:
            _warn(console, "Could not verify the connection. Check your API key and network.")
            console.print("  [dim]You can still continue — fix the key in .env and restart.[/dim]"
                          if _RICH else "  You can still continue — fix the key in .env and restart.")
            if not _confirm(console, "Continue anyway?", default=True):
                _err(console, "Onboard cancelled.")
                return 1

    # ── Step 7: Write files ───────────────────────────────────────────────────
    _step(console, 7, TOTAL_STEPS, "Writing Configuration")
    console.print()

    # Workspace directory
    WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    _ok(console, f"Workspace directory: [bold]{WORKSPACE_DIR}[/bold]"
        if _RICH else f"Workspace directory: {WORKSPACE_DIR}")

    # Write workspace files
    _write_soul(agent_name, personality, WORKSPACE_DIR)
    _ok(console, f"[bold]SOUL.md[/bold] — {agent_name}'s identity"
        if _RICH else f"SOUL.md — {agent_name}'s identity")

    _write_user(your_name or "User", user_context, WORKSPACE_DIR)
    _ok(console, "[bold]USER.md[/bold] — your context"
        if _RICH else "USER.md — your context")

    _write_heartbeat(agent_name, WORKSPACE_DIR)
    _ok(console, "[bold]HEARTBEAT.md[/bold] — proactive checklist"
        if _RICH else "HEARTBEAT.md — proactive checklist")

    _write_memory(WORKSPACE_DIR)
    _ok(console, "[bold]MEMORY.md[/bold] — long-term memory"
        if _RICH else "MEMORY.md — long-term memory")

    # Write .env
    env_updates = {k: v for k, v in collected.items()
                   if k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BYTEZ_API_KEY",
                            "GEMINI_API_KEY", "TELEGRAM_BOT_TOKEN",
                            "TELEGRAM_AUTHORIZED_USER_IDS")}
    if env_updates:
        _update_env(ENV_FILE, env_updates)
        _ok(console, f"[bold].env[/bold] updated ({len(env_updates)} key(s))"
            if _RICH else f".env updated ({len(env_updates)} key(s))")

    # Patch config.yaml
    try:
        _patch_config(
            CONFIG_FILE,
            provider=collected.get("LLM_PROVIDER", "openai"),
            model=collected.get("LLM_MODEL", "gpt-4o"),
            agent_name=agent_name,
        )
        _ok(console, "[bold]config.yaml[/bold] — provider, model, and name updated"
            if _RICH else "config.yaml — provider, model, and name updated")
    except Exception as e:
        _warn(console, f"Could not patch config.yaml: {e}. Edit it manually.")

    # ── Done ──────────────────────────────────────────────────────────────────
    console.rule()
    console.print()
    if _RICH:
        console.print(Panel(
            f"[bold green]✓ {agent_name} is ready![/bold green]\n\n"
            f"[white]Start the CLI:[/white]        [cyan bold]./run.sh[/cyan bold]\n"
            f"[white]Start Telegram:[/white]       [cyan bold]./run.sh telegram[/cyan bold]\n"
            f"[white]Install a skill:[/white]      [cyan bold]python main.py install web-search[/cyan bold]\n"
            f"[white]Run again anytime:[/white]    [cyan bold]python main.py onboard[/cyan bold]\n\n"
            f"[dim]Workspace: {WORKSPACE_DIR}[/dim]",
            border_style="green",
            title="[bold]Setup Complete[/bold]",
        ))
    else:
        print(f"\n{'=' * 50}")
        print(f"  ✓ {agent_name} is ready!")
        print()
        print(f"  Start the CLI:     ./run.sh")
        print(f"  Start Telegram:    ./run.sh telegram")
        print(f"  Install a skill:   python main.py install web-search")
        print(f"  Workspace:         {WORKSPACE_DIR}")
        print(f"{'=' * 50}\n")

    return 0