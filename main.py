"""
main.py — NeuralClaw Entry Point

Usage:
    python main.py                          # CLI interface, default settings
    python main.py --interface telegram     # Telegram bot
    python main.py --interface cli          # CLI REPL (explicit)
    python main.py --log-level DEBUG        # Verbose logging
    python main.py --config path/to/config.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="neuralclaw",
        description="NeuralClaw — Local-first autonomous AI agent platform",
    )
    parser.add_argument(
        "--interface",
        choices=["cli", "telegram"],
        default="cli",
        help="Interface to start (default: cli)",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default=None,
        help="Override log level from config",
    )
    parser.add_argument(
        "--enable-mcp",
        action="store_true",
        default=False,
        help="Enable MCP server connections on startup",
    )
    parser.add_argument(
        "--enable-scheduler",
        action="store_true",
        default=False,
        help="Enable the task scheduler",
    )
    return parser.parse_args()


def bootstrap(args: argparse.Namespace):
    """
    Load config and set up logging.
    Returns (settings, log) ready for use.
    """
    from config.settings import load_settings
    from observability.logger import setup_logging, get_logger

    settings = load_settings(args.config)

    # CLI --log-level flag overrides config.yaml
    log_level = args.log_level or settings.log_level

    setup_logging(
        level=log_level,
        log_dir=settings.log_dir,
        json_format=settings.log_json_format,
        console_output=settings.log_console_output,
    )

    log = get_logger("neuralclaw.main")
    return settings, log


async def main() -> int:
    args = parse_args()
    settings, log = bootstrap(args)

    log.info(
        "neuralclaw.starting",
        version=settings.agent.get("version", "1.0.0"),
        interface=args.interface,
        llm_provider=settings.default_llm_provider,
        llm_model=settings.default_llm_model,
    )

    # ── Validate secrets for chosen interface ─────────────────────────────────
    missing = settings.validate_required_for_interface(args.interface)
    if missing:
        log.error(
            "neuralclaw.startup_failed",
            reason="Missing required environment variables",
            missing=missing,
        )
        print(
            f"\n❌  Missing required environment variables: {', '.join(missing)}\n"
            f"    Copy .env.example → .env and fill in the values.\n",
            file=sys.stderr,
        )
        return 1

    log.info("neuralclaw.config_loaded", agent_name=settings.agent_name)

    # ── Ensure data directories exist ─────────────────────────────────────────
    for path_str in [
        settings.memory.get("chroma_persist_dir", "./data/chroma"),
        str(Path(settings.memory.get("sqlite_path", "./data/sqlite/episodes.db")).parent),
        str(settings.log_dir),
        "./data/agent_files",
    ]:
        Path(path_str).expanduser().mkdir(parents=True, exist_ok=True)

    log.info("neuralclaw.data_dirs_ready")

    # ── Launch interface ───────────────────────────────────────────────────────
    if args.interface == "cli":
        log.info("neuralclaw.interface_starting", interface="cli")
        await _run_cli(settings, log)
    elif args.interface == "telegram":
        log.info("neuralclaw.interface_starting", interface="telegram")
        await _run_telegram(settings, log)

    return 0


async def _run_cli(settings, log) -> None:
    """
    Full CLI REPL — Phase 5 implementation.
    Delegates to interfaces/cli.py.
    """
    from interfaces.cli import run_cli
    await run_cli(settings, log)


async def _run_telegram(settings, log) -> None:
    """
    Placeholder Telegram runner.
    Phase 7 will replace this with the full aiogram bot.
    """
    log.info("telegram.placeholder", note="Phase 7 will implement full bot")
    print("Telegram interface placeholder — Phase 7 will add the full bot.")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))