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
# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Load environment variables FIRST (CRITICAL FIX)
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project root explicitly
ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

# ─────────────────────────────────────────────────────────────────────────────
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
        default=None,
        help="Path to config.yaml (default: $NEURALCLAW_CONFIG or config/config.yaml)",
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
    Load config, validate it fully, and set up logging.
    Returns (settings, log) ready for use.

    Exits with code 1 (after printing a clear message) if:
      - config.yaml has invalid/unsafe values (Pydantic ValidationError)
      - cross-field problems are found (ConfigError from validate_all())
    """
    from config.settings import load_settings, ConfigError
    from observability.logger import setup_logging, get_logger
    from pydantic import ValidationError

    # -- Load and parse -------------------------------------------------------
    try:
        settings = load_settings(args.config)
    except ValidationError as exc:
        # Pydantic rejected a field value (wrong type, bad value, etc.)
        problems = "\n".join(
            f"  • {e['loc'][-1] if e['loc'] else '?'}: {e['msg']}"
            for e in exc.errors()
        )
        print(
            f"\n❌  Config validation failed:\n\n{problems}\n\n"
            f"    Fix config/config.yaml or your .env file and restart.\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:  # noqa: BLE001 — catches yaml.YAMLError, FileNotFoundError, etc.
        print(
            f"\n❌  Failed to load config: {type(exc).__name__}: {exc}\n",
            file=sys.stderr,
        )
        sys.exit(1)

    # -- Cross-field validation -----------------------------------------------
    try:
        settings.validate_all()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    # -- Logging --------------------------------------------------------------
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
        version=settings.agent.version,
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
        settings.memory.chroma_persist_dir,
        str(Path(settings.memory.sqlite_path).parent),
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
    Launch the real Telegram bot interface.
    Delegates to interfaces/telegram.py
    """

    try:
        from interfaces.telegram import run_telegram
    except (ImportError, ModuleNotFoundError) as e:
        log.exception("telegram.import_failed", error=str(e))
        print("\n❌ Failed to import Telegram interface.\n", file=sys.stderr)
        raise

    log.info("telegram.interface_bootstrap")

    try:
        await run_telegram(settings=settings, log=log)
    except KeyboardInterrupt:
        log.info("telegram.interrupted")
    except (OSError, RuntimeError) as e:
        log.exception("telegram.crashed", error=str(e), error_type=type(e).__name__)
        raise


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))