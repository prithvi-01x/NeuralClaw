"""
main.py — NeuralClaw Entry Point

Usage:
    python main.py                              # CLI interface, default settings
    python main.py --interface telegram         # Telegram bot
    python main.py --interface cli              # CLI REPL (explicit)
    python main.py --interface gateway          # WebSocket gateway server
    python main.py --interface gateway-cli      # CLI over gateway
    python main.py --log-level DEBUG            # Verbose logging
    python main.py --config path/to/config.yaml
"""

from __future__ import annotations
# ─────────────────────────────────────────────────────────────────────────────
# 🔐 Load environment variables FIRST (CRITICAL FIX)
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
from pathlib import Path

# Load .env — search CWD and parent directories
def _find_env_file() -> Path | None:
    """Walk up from CWD looking for .env file."""
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        candidate = d / ".env"
        if candidate.is_file():
            return candidate
    return None

_env_path = _find_env_file()
if _env_path:
    load_dotenv(dotenv_path=_env_path)

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
    # Positional subcommand: onboard / install / skills
    parser.add_argument(
        "subcommand",
        nargs="?",
        default=None,
        help=(
            "'onboard' — interactive setup wizard. "
            "'install <skill>' — install a skill by name, URL, or file. "
            "'skills' — list installable skills. "
            "Omit to start the normal interface."
        ),
    )
    parser.add_argument(
        "subcommand_arg",
        nargs="?",
        default=None,
        help="Argument for subcommand (e.g. skill name for 'install').",
    )
    parser.add_argument(
        "--interface",
        choices=["cli", "telegram", "voice", "voice-app", "gateway", "gateway-cli", "webui"],
        default="cli",
        help=(
            "Interface to start (default: cli). "
            "gateway = WebSocket control plane server. "
            "gateway-cli = lightweight CLI over gateway. "
            "webui = web browser interface."
        ),
    )
    parser.add_argument(
        "--gateway-url",
        default=None,
        help="WebSocket URL for gateway-cli (default: ws://host:port from config)",
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
        "--force",
        action="store_true",
        default=False,
        help="Force overwrite when installing a skill that already exists",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        default=False,
        help="Skip LLM health check during onboard",
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
    from neuralclaw.config.settings import load_settings, ConfigError
    from neuralclaw.observability.logger import setup_logging, get_logger
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
    except (FileNotFoundError, PermissionError, IsADirectoryError, OSError) as exc:
        print(
            f"\n❌  Failed to load config: {type(exc).__name__}: {exc}\n",
            file=sys.stderr,
        )
        sys.exit(1)
    except (ValueError, TypeError) as exc:
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

    # ── Subcommands: no full bootstrap needed ─────────────────────────────────
    if args.subcommand == "onboard":
        from neuralclaw.onboard.wizard import run_onboard
        return await run_onboard(skip_health_check=getattr(args, "skip_health_check", False))

    if args.subcommand == "install":
        skill_arg = args.subcommand_arg
        if not skill_arg:
            print("Usage: python main.py install <skill-name-or-url>", file=sys.stderr)
            print("       python main.py skills   (to list available skills)", file=sys.stderr)
            return 1
        from neuralclaw.onboard.skill_installer import run_install
        return run_install(skill_arg, force=getattr(args, "force", False))

    if args.subcommand == "skills":
        from neuralclaw.onboard.skill_installer import run_list_available
        return run_list_available(filter_str=args.subcommand_arg or "")

    if args.subcommand == "clawhub":
        # Route: python main.py clawhub <action> [arg]
        action = args.subcommand_arg or ""
        # Re-parse remaining args from sys.argv for clawhub
        extra_args = sys.argv[3:] if len(sys.argv) > 3 else []
        settings_for_clawhub, _ = bootstrap(args)
        from neuralclaw.onboard.clawhub_installer import clawhub_command
        return await clawhub_command(
            action=action, args=extra_args,
            settings=settings_for_clawhub,
        )

    # ── Normal agent startup ───────────────────────────────────────────────────
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

    # ── Test LLM API Key validity ─────────────────────────────────────────────
    # Instantiating the factory and running health_check ensures the configured
    # key is actually valid with the provider, failing fast if not.
    log.info("neuralclaw.testing_llm", provider=settings.default_llm_provider)
    from neuralclaw.brain import LLMClientFactory
    try:
        client = LLMClientFactory.from_settings(settings)
        is_healthy = await client.health_check()
        if not is_healthy:
            log.error(
                "neuralclaw.llm_health_check_failed",
                provider=settings.default_llm_provider,
            )
            print(
                f"\n❌  LLM health check failed for '{settings.default_llm_provider}'.\n"
                f"    Please double check your API key and network connection.\n",
                file=sys.stderr,
            )
            return 1
    except (ImportError, AttributeError, ValueError) as e:
        log.error("neuralclaw.llm_init_failed", error=str(e), error_type=type(e).__name__)
        print(
            f"\n❌  Failed to initialize LLM provider '{settings.default_llm_provider}': {e}\n",
            file=sys.stderr,
        )
        return 1
    except OSError as e:
        log.error("neuralclaw.llm_init_failed", error=str(e), error_type=type(e).__name__)
        print(
            f"\n❌  Failed to initialize LLM provider '{settings.default_llm_provider}': {e}\n",
            file=sys.stderr,
        )
        return 1

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
    elif args.interface == "voice":
        log.info("neuralclaw.interface_starting", interface="voice")
        await _run_voice(settings, log)
    elif args.interface == "voice-app":
        log.info("neuralclaw.interface_starting", interface="voice-app")
        _run_voice_app(settings, log)
    elif args.interface == "gateway":
        log.info("neuralclaw.interface_starting", interface="gateway")
        await _run_gateway(settings, log)
    elif args.interface == "gateway-cli":
        log.info("neuralclaw.interface_starting", interface="gateway-cli")
        url = args.gateway_url or f"ws://{settings.gateway.host}:{settings.gateway.port}"
        await _run_gateway_cli(settings, log, gateway_url=url)
    elif args.interface == "webui":
        log.info("neuralclaw.interface_starting", interface="webui")
        await _run_webui(settings, log)

    return 0


def _run_voice_app(settings, log) -> None:
    """
    Launch the Qt voice application (Phase H): tray icon + ambient overlay
    + asyncio voice pipeline. This is a synchronous call — Qt's event loop
    blocks until the user quits via the tray menu or Ctrl+C.
    """
    try:
        from neuralclaw.app import run_qt_app
    except ImportError as e:
        log.exception("voice_app.import_failed", error=str(e))
        print(
            "\n❌ PyQt6 not installed. Install with:\n"
            "   pip install PyQt6\n",
            file=sys.stderr,
        )
        raise

    log.info("voice_app.starting")
    try:
        exit_code = run_qt_app(settings=settings)
        log.info("voice_app.exited", exit_code=exit_code)
    except KeyboardInterrupt:
        log.info("voice_app.interrupted")


async def _run_cli(settings, log) -> None:
    """
    Full CLI REPL — Phase 5 implementation.
    Delegates to interfaces/cli.py. Scheduler runs concurrently.
    """
    from neuralclaw.interfaces.cli import run_cli
    await run_cli(settings, log)


async def _run_voice(settings, log) -> None:
    """
    Launch the voice interface (Phase F).
    Delegates to interfaces/voice.py.
    """
    try:
        from neuralclaw.interfaces.voice import run_voice
    except (ImportError, ModuleNotFoundError) as e:
        log.exception("voice.import_failed", error=str(e))
        print("\n❌ Failed to import Voice interface.\n", file=sys.stderr)
        raise

    log.info("voice.interface_bootstrap")
    try:
        await run_voice(settings=settings, log_=log)
    except KeyboardInterrupt:
        log.info("voice.interrupted")
    except (OSError, RuntimeError) as e:
        log.exception("voice.crashed", error=str(e))
        raise


async def _run_telegram(settings, log) -> None:
    """
    Launch the real Telegram bot interface.
    Delegates to interfaces/telegram.py. Scheduler runs concurrently.
    """

    try:
        from neuralclaw.interfaces.telegram import run_telegram
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


async def _run_gateway(settings, log) -> None:
    """
    Launch the WebSocket gateway server.

    Wires up the full agent stack and exposes it over WebSocket.
    All interfaces can connect as thin clients.
    """
    from neuralclaw.kernel.kernel import AgentKernel
    from neuralclaw.gateway.gateway_server import GatewayServer
    from neuralclaw.gateway.session_store import SessionStore

    log.info("gateway.building_kernel")
    kernel = await AgentKernel.build(settings)

    session_store = SessionStore()
    server = GatewayServer(
        orchestrator=kernel.orchestrator,
        session_store=session_store,
        skill_registry=kernel.skill_registry,
        host=settings.gateway.host,
        port=settings.gateway.port,
        auth_token=settings.gateway.auth_token,
        max_connections=settings.gateway.max_connections,
    )

    print(
        f"\n🌐 NeuralClaw Gateway listening on "
        f"ws://{settings.gateway.host}:{settings.gateway.port}\n"
        f"   Press Ctrl+C to stop.\n"
    )

    await server.start()

    try:
        # Keep running until interrupted
        stop = asyncio.Event()
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("gateway.stopping")
    finally:
        await server.shutdown()
        kernel.shutdown()


async def _run_gateway_cli(settings, log, gateway_url: str) -> None:
    """
    Launch the lightweight CLI that connects through the gateway.
    No orchestrator is created — everything goes over WebSocket.
    """
    from neuralclaw.interfaces.gateway_cli import run_gateway_cli

    log.info("gateway_cli.starting", url=gateway_url)
    await run_gateway_cli(
        gateway_url=gateway_url,
        auth_token=settings.gateway.auth_token,
    )


async def _run_webui(settings, log) -> None:
    """
    Launch the Web UI: Gateway WebSocket server + static HTTP file server.

    - Gateway (WebSocket) on settings.gateway.port (default 9090)
    - Web UI (HTTP) on port 8080
    - Auto-opens the browser
    """
    import webbrowser
    from neuralclaw.kernel.kernel import AgentKernel
    from neuralclaw.gateway.gateway_server import GatewayServer
    from neuralclaw.gateway.session_store import SessionStore
    from neuralclaw.gateway.webui_server import start_webui_server

    log.info("webui.building_kernel")
    kernel = await AgentKernel.build(settings)

    session_store = SessionStore()

    # Start WebSocket gateway
    gw_server = GatewayServer(
        orchestrator=kernel.orchestrator,
        session_store=session_store,
        skill_registry=kernel.skill_registry,
        host=settings.gateway.host,
        port=settings.gateway.port,
        auth_token=settings.gateway.auth_token,
        max_connections=settings.gateway.max_connections,
    )
    await gw_server.start()

    # Start HTTP static file server
    http_port = 8080
    httpd = await start_webui_server(
        host=settings.gateway.host,
        port=http_port,
    )

    url = f"http://{settings.gateway.host}:{http_port}"
    print(
        f"\n🌐 NeuralClaw Web UI\n"
        f"   Web UI:    {url}\n"
        f"   Gateway:   ws://{settings.gateway.host}:{settings.gateway.port}\n"
        f"   Press Ctrl+C to stop.\n"
    )

    # Open browser
    try:
        webbrowser.open(url)
    except Exception:
        pass  # No browser available — that's fine

    try:
        stop = asyncio.Event()
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        log.info("webui.stopping")
    finally:
        httpd.shutdown()
        await gw_server.shutdown()
        kernel.shutdown()


def main_sync() -> None:
    """Synchronous entry point for console_scripts (pyproject.toml)."""
    sys.exit(asyncio.run(main()))


if __name__ == "__main__":
    main_sync()