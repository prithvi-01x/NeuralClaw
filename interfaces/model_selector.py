"""
interfaces/model_selector.py — NeuralClaw Interactive Model Selector (v2)

Premium Gemini CLI-style keyboard-driven model picker with live Ollama discovery.

New in v2:
  - Dynamic Ollama model discovery via local API (no hardcoded models)
  - Live metadata: model size, context window, GPU/CPU status, running state
  - Running models visually highlighted
  - Auto (Recommended) prefers local Ollama first, then cloud
  - Non-blocking async prefetch — UI opens instantly, metadata appears on redraw
  - Telegram: inline keyboard shows size/status metadata in button labels
  - Graceful handling when Ollama is offline or no models installed

Terminal UI:
  ╭────────────────────────────────────────────────────────────────╮
  │  🤖  Select Model                             ↻ refreshed 312ms│
  ├────────────────────────────────────────────────────────────────┤
  │  Automatic                                                     │
  │  ● Auto (Recommended)                            [active]      │
  │    Prefers local → cloud                                       │
  │                                                                │
  │  Ollama (Local)                                                │
  │  ◉ qwen2.5:3b             1.9 GB · ctx 32k · 🟢 GPU           │
  │  ○ ministral-3b:8b        6.0 GB · ctx 8k  · idle             │
  │  ○ gemma3:4b              3.3 GB · ctx 8k  · CPU              │
  │                                                                │
  │  OpenAI                                                        │
  │  ○ gpt-4o                 High quality · multimodal           │
  ├────────────────────────────────────────────────────────────────┤
  │  ↑↓ navigate · Enter select · d set default · Esc cancel       │
  ╰────────────────────────────────────────────────────────────────╯
"""

from __future__ import annotations

import asyncio
import json
import sys
import termios
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console
from exceptions import NeuralClawError


# ─────────────────────────────────────────────────────────────────────────────
# Ollama live metadata
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class OllamaModelMeta:
    """Runtime metadata for a single installed Ollama model."""
    model_id: str
    size_bytes: int = 0
    context_length: int = 0
    is_running: bool = False
    uses_gpu: bool = False
    family: str = ""
    parameter_size: str = ""

    @property
    def size_display(self) -> str:
        if not self.size_bytes:
            return ""
        gb = self.size_bytes / 1_073_741_824
        if gb >= 1.0:
            return f"{gb:.1f} GB"
        return f"{self.size_bytes / 1_048_576:.0f} MB"

    @property
    def ctx_display(self) -> str:
        if not self.context_length:
            return ""
        k = self.context_length / 1024
        return f"ctx {k:.0f}k" if k >= 1 else f"ctx {self.context_length}"

    @property
    def status_display(self) -> str:
        if self.is_running:
            return "🟢 GPU" if self.uses_gpu else "🟡 CPU"
        return "idle"

    @property
    def meta_line(self) -> str:
        parts = [p for p in [self.size_display, self.ctx_display, self.status_display] if p]
        return " · ".join(parts)


class OllamaDiscovery:
    """
    Queries the local Ollama REST API to discover installed and running models.

    Endpoints used:
      GET  /api/tags   → all installed models + sizes + details
      GET  /api/ps     → currently loaded models + GPU VRAM info
      POST /api/show   → per-model metadata (context_length, family)
    """

    def __init__(self, base_url: str = "http://localhost:11434"):
        self._base = base_url.rstrip("/")
        self._show_cache: dict[str, int] = {}   # model_id → context_length

    async def fetch_all(self) -> tuple[list[OllamaModelMeta], Optional[str]]:
        """
        Return (list_of_meta, error_string).
        error_string is None on success.
        """
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, self._fetch_all_sync)
            return result
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            return [], f"Ollama error: {e}"

    def _fetch_all_sync(self) -> tuple[list[OllamaModelMeta], Optional[str]]:
        import urllib.request
        import urllib.error

        # 1. Installed models (/api/tags)
        try:
            with urllib.request.urlopen(f"{self._base}/api/tags", timeout=3) as r:
                tags = json.loads(r.read().decode())
        except urllib.error.URLError:
            return [], "Ollama server not running"
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            return [], f"Ollama unreachable: {e}"

        raw_models = tags.get("models", [])
        if not raw_models:
            return [], "No local models installed"

        metas: dict[str, OllamaModelMeta] = {}
        for m in raw_models:
            mid = m.get("model") or m.get("name", "")
            if not mid:
                continue
            details = m.get("details", {})
            ctx = (
                m.get("context_length")
                or details.get("context_length")
                or self._show_cache.get(mid, 0)
            )
            metas[mid] = OllamaModelMeta(
                model_id=mid,
                size_bytes=m.get("size", 0),
                context_length=int(ctx) if ctx else 0,
                family=details.get("family", ""),
                parameter_size=details.get("parameter_size", ""),
            )

        # 2. Running models (/api/ps)
        try:
            with urllib.request.urlopen(f"{self._base}/api/ps", timeout=2) as r:
                ps = json.loads(r.read().decode())
            for rm in ps.get("models", []):
                mid = rm.get("model") or rm.get("name", "")
                if mid in metas:
                    metas[mid].is_running = True
                    metas[mid].uses_gpu = rm.get("size_vram", 0) > 0
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass  # /api/ps may not exist on older Ollama versions

        # 3. Context length from /api/show (best-effort, cached)
        for mid, meta in metas.items():
            if meta.context_length:
                continue
            ctx = self._fetch_show_ctx(mid)
            if ctx:
                meta.context_length = ctx
                self._show_cache[mid] = ctx

        return list(metas.values()), None

    def _fetch_show_ctx(self, model_id: str) -> int:
        import urllib.request
        try:
            body = json.dumps({"model": model_id, "verbose": False}).encode()
            req = urllib.request.Request(
                f"{self._base}/api/show",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2) as r:
                data = json.loads(r.read().decode())
            info = data.get("model_info", {})
            ctx = (
                info.get("llama.context_length")
                or info.get("context_length")
                or 0
            )
            return int(ctx) if ctx else 0
        except (OSError, ValueError, RuntimeError, AttributeError):
            return 0


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ModelOption:
    """A selectable model entry in the picker."""
    key: str
    name: str
    description: str
    provider: str
    model_id: str
    requires_key: str
    ollama_meta: Optional[OllamaModelMeta] = field(default=None, hash=False, compare=False)
    is_running: bool = False


# Static cloud models — Ollama entries are built dynamically each open
_STATIC_CLOUD_OPTIONS: list[ModelOption] = [
    ModelOption("auto", "Auto (Recommended)", "Prefers local Ollama → cloud providers",
                "auto", "auto", ""),
    # OpenAI
    ModelOption("openai:gpt-4o",      "gpt-4o",      "High quality · multimodal",         "openai", "gpt-4o",           "openai_api_key"),
    ModelOption("openai:gpt-4o-mini", "gpt-4o-mini", "Fast and affordable · low cost",     "openai", "gpt-4o-mini",      "openai_api_key"),
    ModelOption("openai:gpt-4-turbo", "gpt-4-turbo", "Powerful · large context window",    "openai", "gpt-4-turbo",      "openai_api_key"),
    ModelOption("openai:o1-mini",     "o1-mini",     "Reasoning model · fast",             "openai", "o1-mini",          "openai_api_key"),
    # Anthropic
    ModelOption("anthropic:claude-3-5-sonnet-20241022", "claude-3-5-sonnet", "Most intelligent · top coding",   "anthropic", "claude-3-5-sonnet-20241022", "anthropic_api_key"),
    ModelOption("anthropic:claude-3-5-haiku-20241022",  "claude-3-5-haiku",  "Fast and compact · low latency",  "anthropic", "claude-3-5-haiku-20241022",  "anthropic_api_key"),
    ModelOption("anthropic:claude-3-opus-20240229",     "claude-3-opus",     "Most capable · complex tasks",    "anthropic", "claude-3-opus-20240229",     "anthropic_api_key"),
    # Gemini
    ModelOption("gemini:gemini-2.0-flash",  "gemini-2.0-flash",  "Latest generation · fast",          "gemini", "gemini-2.0-flash",  "gemini_api_key"),
    ModelOption("gemini:gemini-1.5-pro",    "gemini-1.5-pro",    "Capable multimodal · 2M context",   "gemini", "gemini-1.5-pro",    "gemini_api_key"),
    ModelOption("gemini:gemini-1.5-flash",  "gemini-1.5-flash",  "Fast and efficient · multimodal",   "gemini", "gemini-1.5-flash",  "gemini_api_key"),
    # OpenRouter
    ModelOption("openrouter:openai/gpt-4o",                         "gpt-4o (OpenRouter)",           "GPT-4o via OpenRouter",          "openrouter", "openai/gpt-4o",                         "openrouter_api_key"),
    ModelOption("openrouter:anthropic/claude-3.5-sonnet",           "claude-3.5-sonnet (OpenRouter)", "Claude via OpenRouter",          "openrouter", "anthropic/claude-3.5-sonnet",           "openrouter_api_key"),
    ModelOption("openrouter:meta-llama/llama-3.1-405b-instruct",    "llama-3.1-405b (OpenRouter)",   "Meta's largest open model",      "openrouter", "meta-llama/llama-3.1-405b-instruct",    "openrouter_api_key"),
    # Bytez
    ModelOption("bytez:openai/gpt-5", "gpt-5 (Bytez)", "Bytez API gateway", "bytez", "openai/gpt-5", "bytez_api_key"),
]

_PROVIDER_LABELS: dict[str, str] = {
    "auto":       "Automatic",
    "ollama":     "Ollama (Local)",
    "openai":     "OpenAI",
    "anthropic":  "Anthropic",
    "gemini":     "Google Gemini",
    "openrouter": "OpenRouter",
    "bytez":      "Bytez",
}

_PROVIDER_ORDER = ["auto", "ollama", "openai", "anthropic", "gemini", "openrouter", "bytez"]

# Module-level cache — updated each time /model opens
MODEL_OPTIONS: list[ModelOption] = list(_STATIC_CLOUD_OPTIONS)


# ─────────────────────────────────────────────────────────────────────────────
# Default model persistence
# ─────────────────────────────────────────────────────────────────────────────

_DEFAULT_PATH = Path.home() / ".neuralclaw" / "default_model.json"


def load_default_model() -> Optional[str]:
    try:
        if _DEFAULT_PATH.exists():
            return json.loads(_DEFAULT_PATH.read_text())["model_key"]
    except (OSError, ValueError, RuntimeError, AttributeError) as e:
        import logging as _logging
        _logging.getLogger(__name__).debug("model_selector.load_default_failed: %s", e)
    return None


def save_default_model(model_key: str) -> None:
    _DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DEFAULT_PATH.write_text(json.dumps({"model_key": model_key}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Option building helpers
# ─────────────────────────────────────────────────────────────────────────────


def _build_options_with_ollama(
    ollama_models: list[OllamaModelMeta],
    ollama_error: Optional[str],
) -> list[ModelOption]:
    """Merge live Ollama data into a fresh ordered option list."""
    ollama_opts: list[ModelOption] = []

    if ollama_error and not ollama_models:
        # Sentinel — shown as disabled row
        ollama_opts.append(ModelOption(
            key="ollama:__error__",
            name=ollama_error,
            description="",
            provider="ollama",
            model_id="__error__",
            requires_key="",
        ))
    elif not ollama_models:
        # Still loading (placeholder)
        ollama_opts.append(ModelOption(
            key="ollama:__loading__",
            name="Discovering local models…",
            description="",
            provider="ollama",
            model_id="__loading__",
            requires_key="",
        ))
    else:
        # Running models first, then alphabetical
        for meta in sorted(ollama_models, key=lambda m: (not m.is_running, m.model_id)):
            # Look up capability hint from the registry
            try:
                from brain.capabilities import get_capabilities
                caps = get_capabilities("ollama", meta.model_id)
                cap_hint = "tools ✓" if caps.supports_tools else "chat only"
            except (OSError, ValueError, RuntimeError, AttributeError):
                cap_hint = ""
            base_desc = meta.meta_line
            description = f"{base_desc} · {cap_hint}" if base_desc else cap_hint
            ollama_opts.append(ModelOption(
                key=f"ollama:{meta.model_id}",
                name=meta.model_id,
                description=description,
                provider="ollama",
                model_id=meta.model_id,
                requires_key="",
                ollama_meta=meta,
                is_running=meta.is_running,
            ))

    auto_opts  = [o for o in _STATIC_CLOUD_OPTIONS if o.provider == "auto"]
    cloud_opts = [o for o in _STATIC_CLOUD_OPTIONS if o.provider not in ("auto", "ollama")]
    return _sort_by_provider(auto_opts + ollama_opts + cloud_opts)


def _sort_by_provider(options: list[ModelOption]) -> list[ModelOption]:
    order = {p: i for i, p in enumerate(_PROVIDER_ORDER)}
    return sorted(options, key=lambda o: order.get(o.provider, 99))


def _is_sentinel(opt: ModelOption) -> bool:
    return opt.model_id in ("__error__", "__loading__") or not opt.model_id


def validate_model(option: ModelOption, settings) -> Optional[str]:
    if option.provider in ("auto", "ollama"):
        return None
    if option.requires_key:
        if not getattr(settings, option.requires_key, None):
            return f"No {option.requires_key.upper()}"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM client builder
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_auto_model(settings, options: list[ModelOption]) -> Optional[ModelOption]:
    # Prefer running Ollama models first
    for opt in options:
        if opt.provider == "ollama" and opt.is_running and not _is_sentinel(opt):
            return opt
    # Any Ollama model
    for opt in options:
        if opt.provider == "ollama" and not _is_sentinel(opt):
            return opt
    # Cloud fallback
    for provider in ["openai", "anthropic", "gemini", "openrouter", "bytez"]:
        for opt in options:
            if opt.provider == provider and not validate_model(opt, settings):
                return opt
    return None


def build_llm_client_for_model(
    option: ModelOption,
    settings,
    options: Optional[list[ModelOption]] = None,
):
    """Return (ResilientLLMClient, error_str)."""
    from brain import LLMClientFactory, ResilientLLMClient

    if option.provider == "auto":
        resolved = _resolve_auto_model(settings, options or MODEL_OPTIONS)
        if resolved is None:
            return None, "No models available — check API keys or start Ollama."
        option = resolved

    if _is_sentinel(option):
        return None, "Cannot switch to placeholder entry."

    api_key_map = {
        "openai":     getattr(settings, "openai_api_key", None),
        "anthropic":  getattr(settings, "anthropic_api_key", None),
        "ollama":     None,
        "openrouter": getattr(settings, "openrouter_api_key", None),
        "gemini":     getattr(settings, "gemini_api_key", None),
        "bytez":      getattr(settings, "bytez_api_key", None),
    }
    base_url_map = {
        "ollama": getattr(settings, "ollama_base_url", "http://localhost:11434") + "/v1",
    }
    try:
        primary = LLMClientFactory.create(
            provider=option.provider,
            api_key=api_key_map.get(option.provider),
            base_url=base_url_map.get(option.provider),
        )
        return ResilientLLMClient(primary=primary, fallbacks=[]), None
    except (OSError, ValueError, RuntimeError, AttributeError) as e:
        return None, str(e)


def current_model_key(settings) -> str:
    provider = getattr(settings, "default_llm_provider", "openai")
    model    = getattr(settings, "default_llm_model",    "gpt-4o")
    return f"{provider}:{model}"


# ─────────────────────────────────────────────────────────────────────────────
# Terminal raw-mode key reader
# ─────────────────────────────────────────────────────────────────────────────


def _read_key(fd: int) -> str:
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        try:
            seq = sys.stdin.read(2)
            if seq == "[A": return "up"
            if seq == "[B": return "down"
            if seq == "[C": return "right"
            if seq == "[D": return "left"
            # Page up/down
            if seq in ("[5", "[6"):
                sys.stdin.read(1)
                return "pageup" if seq == "[5" else "pagedown"
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass  # non-blocking; ESC without a sequence suffix is fine
        return "esc"
    if ch in ("\r", "\n"): return "enter"
    if ch in ("\x03", "\x04"): return "quit"
    return ch.lower()


async def _read_key_async(fd: int) -> str:
    """
    Read a complete keypress (including multi-byte arrow sequences) without
    blocking the event loop.

    Uses asyncio.get_event_loop().add_reader() to get a callback when the fd
    has data, then reads the full escape sequence synchronously using os.read()
    on the raw file descriptor — no threads, no races.
    """
    import os

    loop = asyncio.get_running_loop()

    # Wait for the fd to become readable via the event loop's I/O selector
    fut: asyncio.Future = loop.create_future()

    def _on_readable() -> None:
        loop.remove_reader(fd)
        if not fut.done():
            fut.set_result(None)

    loop.add_reader(fd, _on_readable)
    try:
        await fut
    finally:
        # Guard: remove reader if still registered (e.g. on cancellation)
        try:
            loop.remove_reader(fd)
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass  # may already be removed on cancellation — safe to ignore

    # fd is readable — read first byte using os.read (non-buffered)
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        # Arrow/function keys: ESC [ X  (3 bytes total)
        # Use a tight non-blocking read to get the rest of the sequence.
        # os.read on a raw tty in raw mode returns immediately if bytes are
        # waiting, so two small reads are enough.
        import select as _select
        r, _, _ = _select.select([fd], [], [], 0.04)
        if r:
            rest = os.read(fd, 4)   # read up to 4 more bytes
            seq = rest.decode("latin-1", errors="replace")
            if seq.startswith("[A"): return "up"
            if seq.startswith("[B"): return "down"
            if seq.startswith("[C"): return "right"
            if seq.startswith("[D"): return "left"
            if seq.startswith("[5"): return "pageup"
            if seq.startswith("[6"): return "pagedown"
            if seq.startswith("[H"): return "home"
            if seq.startswith("[F"): return "end"
        return "esc"

    try:
        c = ch.decode("utf-8")
    except UnicodeDecodeError:
        return ""

    if c in ("\r", "\n"): return "enter"
    if c in ("\x03", "\x04"): return "quit"
    return c.lower()


# ─────────────────────────────────────────────────────────────────────────────
# ANSI palette
# ─────────────────────────────────────────────────────────────────────────────

_R   = "\033[0m"         # reset
_B   = "\033[1m"         # bold
_D   = "\033[2m"         # dim
_CY  = "\033[36m"        # cyan
_BCY = "\033[96m"        # bright cyan
_GR  = "\033[32m"        # green
_BGR = "\033[92m"        # bright green
_YL  = "\033[33m"        # yellow
_BYL = "\033[93m"        # bright yellow
_RD  = "\033[31m"        # red
_BRD = "\033[91m"        # bright red
_WH  = "\033[97m"        # white
_BGS = "\033[48;5;236m"  # bg selected (charcoal)
_BRN = "\033[48;5;22m"   # bg running (deep green)
_FSL = "\033[96m"        # fg selected (bright cyan)

_PC: dict[str, str] = {
    "auto":       "\033[97m",
    "openai":     "\033[92m",
    "anthropic":  "\033[95m",
    "gemini":     "\033[94m",
    "ollama":     "\033[93m",
    "openrouter": "\033[96m",
    "bytez":      "\033[91m",
}

_W = 68  # box width


def _pad(text_raw: str, vis: int) -> str:
    return f"│ {text_raw}{' ' * max(0, _W - 4 - vis)} │"


# ─────────────────────────────────────────────────────────────────────────────
# Renderer
# ─────────────────────────────────────────────────────────────────────────────


def _render_selector(
    options: list[ModelOption],
    selected: int,
    default_key: Optional[str],
    current_key: str,
    settings,
    refresh_ms: Optional[float] = None,
) -> str:
    lines: list[str] = []
    top = f"╭{'─' * (_W - 2)}╮"
    bot = f"╰{'─' * (_W - 2)}╯"
    sep = f"├{'─' * (_W - 2)}┤"

    # Title
    lines.append(_BCY + top + _R)
    title_raw = f"{_B}{_WH}🤖  Select Model{_R}"
    title_vis = 16
    if refresh_ms is not None:
        rs = f"↻ {refresh_ms:.0f}ms"
        rpad = max(0, _W - 4 - title_vis - len(rs))
        lines.append(f"│ {title_raw}{' ' * rpad}{_D}{rs}{_R} │")
    else:
        rpad = max(0, _W - 4 - title_vis - 12)
        lines.append(f"│ {title_raw}{' ' * rpad}{_D}loading…{_R}     │")
    lines.append(_BCY + sep + _R)

    last_provider = ""
    for i, opt in enumerate(options):
        sentinel    = _is_sentinel(opt)
        is_selected = i == selected
        is_current  = opt.key == current_key
        is_default  = opt.key == default_key
        is_running  = opt.is_running
        err         = None if opt.provider in ("auto", "ollama") else validate_model(opt, settings)
        unavailable = err is not None

        # Group header
        if opt.provider != last_provider:
            if last_provider:
                lines.append(_pad("", 0))
            pc = _PC.get(opt.provider, _D)
            lbl = _PROVIDER_LABELS.get(opt.provider, opt.provider.upper())
            lines.append(_pad(f"{pc}{_D}{lbl}{_R}", len(lbl)))
            last_provider = opt.provider

        # Sentinel rows (error / loading)
        if sentinel:
            icon = "⠋" if opt.model_id == "__loading__" else "⚠"
            col  = _D if opt.model_id == "__loading__" else f"{_D}{_RD}"
            lines.append(_pad(f"  {col}{icon}  {opt.name}{_R}", 4 + len(opt.name)))
            continue

        # Bullet
        if is_selected and is_running:
            bullet = f"{_BRN}{_BGR}▶{_R}"
        elif is_selected:
            bullet = f"{_BGS}{_FSL}{_B}●{_R}"
        elif is_current:
            bullet = f"{_BGR}▶{_R}"
        elif is_running:
            bullet = f"{_GR}◉{_R}"
        else:
            bullet = f"{_D}○{_R}"

        # Inline tags
        tags = ""
        tv   = 0
        if is_current:
            tags += f" {_BGR}[active]{_R}"; tv += 8
        if is_default:
            tags += f" {_BYL}[default]{_R}"; tv += 9
        if unavailable:
            errs = f"[{err}]"
            tags += f" {_D}{errs}{_R}"; tv += len(errs) + 1
        # Capability hint — show [chat only] for models without tool support
        if not unavailable and not is_current:
            try:
                from brain.capabilities import get_capabilities
                caps = get_capabilities(opt.provider, opt.model_id)
                if not caps.supports_tools:
                    tags += f" {_DIM}[chat only]{_R}"; tv += 11
            except (NeuralClawError, AttributeError, ValueError) as _cap_err:
                import logging as _logging
                _logging.getLogger(__name__).debug("model_selector.caps_check_failed: %s", _cap_err)

        # Name
        if unavailable:
            nc = _D
        elif is_selected:
            nc = f"{_BGS}{_FSL}{_B}"
        elif is_running:
            nc = f"{_BGR}{_B}"
        else:
            nc = _B

        name_raw = f"{nc}{opt.name}{_R}"
        name_vis = len(opt.name)

        bg_on  = _BGS if is_selected and not is_running else (_BRN if is_selected and is_running else "")
        bg_off = _R   if is_selected else ""

        row1 = f"{bg_on}  {bullet} {name_raw}{tags}{bg_off}"
        lines.append(_pad(row1, 4 + name_vis + tv))

        # Description / metadata
        desc = opt.description
        if desc:
            if unavailable:
                dc = f"{_D}{_RD}"
            elif is_selected:
                dc = f"{_BGS}{_D}"
            else:
                dc = _D
            lines.append(_pad(f"{dc}    {desc}{_R}", 4 + len(desc)))

    # Footer
    lines.append(_pad("", 0))
    lines.append(_BCY + sep + _R)
    hint = f"{_D}↑↓ navigate · Enter select · d set default · Esc cancel{_R}"
    lines.append(_pad(hint, 54))
    lines.append(_BCY + bot + _R)

    return "\r\n".join(lines) + "\r\n"


# ─────────────────────────────────────────────────────────────────────────────
# Navigation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _find_index(options: list[ModelOption], key: str) -> int:
    for i, o in enumerate(options):
        if o.key == key:
            return i
    # Land on first selectable
    for i, o in enumerate(options):
        if not _is_sentinel(o):
            return i
    return 0


def _next_sel(options: list[ModelOption], cur: int) -> int:
    n = len(options)
    for d in range(1, n + 1):
        idx = (cur + d) % n
        if not _is_sentinel(options[idx]):
            return idx
    return cur


def _prev_sel(options: list[ModelOption], cur: int) -> int:
    n = len(options)
    for d in range(1, n + 1):
        idx = (cur - d) % n
        if not _is_sentinel(options[idx]):
            return idx
    return cur


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI selector
# ─────────────────────────────────────────────────────────────────────────────


async def run_model_selector(
    console: Console,
    settings,
    current_key: Optional[str] = None,
) -> Optional[tuple[str, str, bool]]:
    """
    Open the interactive model selector.

    Flow:
      1. Opens immediately with a "Discovering…" placeholder for Ollama.
      2. Fetches Ollama models in a background task.
      3. On completion, redraws with full metadata.
      4. Accepts keyboard input throughout.

    Returns (provider, model_id, set_as_default) or None if cancelled.
    """
    if not sys.stdin.isatty():
        console.print("[yellow]⚠ /model requires an interactive terminal.[/]")
        return None

    if current_key is None:
        current_key = current_model_key(settings)
    default_key = load_default_model()
    ollama_base = getattr(settings, "ollama_base_url", "http://localhost:11434")
    discovery   = OllamaDiscovery(base_url=ollama_base)

    # Initial option list with loading placeholder
    options  = _build_options_with_ollama([], None)
    selected = _find_index(options, current_key)
    refresh_ms: Optional[float] = None
    ollama_ready = False

    ollama_task: asyncio.Task
    watcher_task: asyncio.Task

    fd       = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    num_lines = 0

    def clear_ui() -> None:
        nonlocal num_lines
        if num_lines > 0:
            sys.stdout.write(f"\033[{num_lines}A\033[0J")
            sys.stdout.flush()
        num_lines = 0

    def draw_ui() -> None:
        nonlocal num_lines
        rendered = _render_selector(options, selected, default_key, current_key, settings, refresh_ms)
        sys.stdout.write(rendered)
        sys.stdout.flush()
        num_lines = rendered.count("\r\n")

    def flash(msg: str) -> None:
        sys.stdout.write(f"\r\n  {_BRD}✗ {msg}{_R}\r\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        # Set stdin fd to non-blocking so os.read() never hangs after add_reader fires
        import os as _os
        import fcntl
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | _os.O_NONBLOCK)

        sys.stdout.write("\r\n")
        draw_ui()

        t0 = time.monotonic()
        ollama_task: asyncio.Task = asyncio.create_task(discovery.fetch_all())

        async def _ollama_watcher() -> None:
            """Redraws the UI as soon as Ollama discovery finishes."""
            nonlocal options, selected, refresh_ms, ollama_ready
            try:
                ollama_models, ollama_err = await ollama_task
                refresh_ms = (time.monotonic() - t0) * 1000
                options    = _build_options_with_ollama(ollama_models, ollama_err)
                selected   = _find_index(options, current_key)
                ollama_ready = True
                global MODEL_OPTIONS
                MODEL_OPTIONS = list(options)
                clear_ui()
                draw_ui()
            except asyncio.CancelledError:
                pass  # expected on cancellation
            except (OSError, RuntimeError, AttributeError) as _watch_err:
                import logging as _logging
                _logging.getLogger(__name__).debug("model_selector.ollama_watcher_failed: %s", _watch_err)

        watcher_task: asyncio.Task = asyncio.create_task(_ollama_watcher())

        while True:
            # _read_key_async polls select() in 50ms chunks, yielding to the
            # event loop each time so the Ollama watcher can make progress.
            key = await _read_key_async(fd)

            # Redundant check — watcher already redraws, but sync state if needed
            if not ollama_ready and ollama_task.done():
                ollama_ready = True
                try:
                    ollama_models, ollama_err = ollama_task.result()
                    refresh_ms = (time.monotonic() - t0) * 1000
                    options  = _build_options_with_ollama(ollama_models, ollama_err)
                    selected = _find_index(options, current_key)
                    # Update module-level cache
                    global MODEL_OPTIONS
                    MODEL_OPTIONS = list(options)
                except (OSError, RuntimeError, AttributeError) as _refresh_err:
                    import logging as _logging
                    _logging.getLogger(__name__).debug(
                        "model_selector.ollama_refresh_failed: %s", _refresh_err
                    )

            if key in ("esc", "q", "quit"):
                clear_ui()
                return None

            elif key in ("up", "left"):
                selected = _prev_sel(options, selected)
                clear_ui()
                draw_ui()

            elif key in ("down", "right"):
                selected = _next_sel(options, selected)
                clear_ui()
                draw_ui()

            elif key == "pageup":
                for _ in range(5):
                    selected = _prev_sel(options, selected)
                clear_ui()
                draw_ui()

            elif key == "pagedown":
                for _ in range(5):
                    selected = _next_sel(options, selected)
                clear_ui()
                draw_ui()

            elif key in ("enter", "d"):
                opt = options[selected]

                if _is_sentinel(opt):
                    flash("Not selectable — use ↑↓ to navigate")
                    await asyncio.sleep(1.2)
                    clear_ui()
                    draw_ui()
                    continue

                err = validate_model(opt, settings)
                if err:
                    flash(err)
                    await asyncio.sleep(1.2)
                    clear_ui()
                    draw_ui()
                    continue

                set_default = key == "d"
                clear_ui()
                return (opt.provider, opt.model_id, set_default)

    finally:
        # Cancel background tasks if still running
        for task in [t for t in (locals().get("ollama_task"), locals().get("watcher_task")) if t is not None]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass  # cleanup — ignore errors
        # Restore blocking mode then terminal settings
        try:
            import fcntl as _fcntl, os as _os2
            fl = _fcntl.fcntl(fd, _fcntl.F_GETFL)
            _fcntl.fcntl(fd, _fcntl.F_SETFL, fl & ~_os2.O_NONBLOCK)
        except (OSError, ValueError, RuntimeError, AttributeError):
            pass  # non-POSIX environment or already restored
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        sys.stdout.write("\r\n")
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Async fetch helper for CLI pre-loading
# ─────────────────────────────────────────────────────────────────────────────


async def fetch_ollama_options(settings) -> tuple[list[ModelOption], Optional[str]]:
    """
    Fetch live Ollama data and return (merged_options, error).
    Called from Telegram handler to get fresh data before building keyboard.
    """
    ollama_base = getattr(settings, "ollama_base_url", "http://localhost:11434")
    discovery   = OllamaDiscovery(base_url=ollama_base)
    models, err = await discovery.fetch_all()
    opts = _build_options_with_ollama(models, err)
    global MODEL_OPTIONS
    MODEL_OPTIONS = list(opts)
    return opts, err


# ─────────────────────────────────────────────────────────────────────────────
# Telegram support
# ─────────────────────────────────────────────────────────────────────────────


async def build_telegram_model_keyboard_async(settings, current_key: Optional[str] = None):
    """
    Build InlineKeyboardMarkup with live Ollama metadata.
    Preferred over sync wrapper in async Telegram handlers.
    Returns (text_str, InlineKeyboardMarkup).
    """
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    if current_key is None:
        current_key = current_model_key(settings)
    default_key = load_default_model()

    opts, _ = await fetch_ollama_options(settings)
    rows: list[list[InlineKeyboardButton]] = []

    for opt in opts:
        if _is_sentinel(opt) and opt.model_id == "__loading__":
            continue
        if _is_sentinel(opt):
            rows.append([InlineKeyboardButton(f"⚠ {opt.name}", callback_data="model_unavailable:ollama_error")])
            continue

        err = validate_model(opt, settings)
        unavailable = err is not None
        icon = "▶" if opt.key == current_key else ("★" if opt.key == default_key else "○")

        if opt.provider == "ollama" and opt.ollama_meta:
            meta = opt.ollama_meta
            run_i = "🟢" if meta.is_running else "·"
            sz    = f" {meta.size_display}" if meta.size_bytes else ""
            label = f"{icon} {opt.name}{sz} {run_i}"
        else:
            label = f"{icon} {opt.name}" + (" ✗" if unavailable else "")

        cb = f"model_unavailable:{opt.key}" if unavailable else f"model_select:{opt.key}"
        cb = cb[:64]  # Telegram callback_data limit
        rows.append([InlineKeyboardButton(label, callback_data=cb)])

    rows.append([
        InlineKeyboardButton("★ Set as Default", callback_data="model_set_default"),
        InlineKeyboardButton("✗ Cancel",          callback_data="model_cancel"),
    ])
    return InlineKeyboardMarkup(rows)


def format_telegram_model_list(settings, current_key: Optional[str] = None) -> str:
    """Format model list header text for Telegram (uses cached MODEL_OPTIONS)."""
    if current_key is None:
        current_key = current_model_key(settings)
    default_key = load_default_model()

    lines = ["*🤖 Select Model*\n"]
    last_provider = ""

    for opt in MODEL_OPTIONS:
        if opt.provider != last_provider:
            lbl = _PROVIDER_LABELS.get(opt.provider, opt.provider.upper())
            lines.append(f"\n*{lbl}*")
            last_provider = opt.provider

        if _is_sentinel(opt):
            icon = "⠋" if opt.model_id == "__loading__" else "⚠"
            lines.append(f"  {icon} _{opt.name}_")
            continue

        err   = validate_model(opt, settings)
        stat  = "▶" if opt.key == current_key else ("★" if opt.key == default_key else "○")
        extra = ""

        if opt.provider == "ollama" and opt.ollama_meta:
            meta = opt.ollama_meta
            run  = "  🟢" if meta.is_running else ""
            extra = f"  _{opt.description}_{run}"
        elif err:
            extra = f"  ✗ _{err}_"

        lines.append(f"{stat} `{opt.name}`{extra}")

    lines.append("\n_▶ active · ★ default · 🟢 running_")
    return "\n".join(lines)


# Sync wrapper for backward compat
def build_telegram_model_keyboard(settings, current_key: Optional[str] = None):
    """Sync fallback — no live Ollama data. Prefer build_telegram_model_keyboard_async()."""
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    if current_key is None:
        current_key = current_model_key(settings)
    default_key = load_default_model()
    rows: list[list[InlineKeyboardButton]] = []

    for opt in MODEL_OPTIONS:
        if _is_sentinel(opt):
            continue
        err  = validate_model(opt, settings)
        icon = "▶" if opt.key == current_key else ("★" if opt.key == default_key else "○")

        if opt.provider == "ollama" and opt.ollama_meta:
            meta = opt.ollama_meta
            sz   = f" {meta.size_display}" if meta.size_bytes else ""
            run  = " 🟢" if meta.is_running else ""
            label = f"{icon} {opt.name}{sz}{run}"
        else:
            label = f"{icon} {opt.name}" + (" ✗" if err else "")

        cb = f"model_unavailable:{opt.key}" if err else f"model_select:{opt.key}"
        rows.append([InlineKeyboardButton(label, callback_data=cb[:64])])

    rows.append([
        InlineKeyboardButton("★ Set as Default", callback_data="model_set_default"),
        InlineKeyboardButton("✗ Cancel",          callback_data="model_cancel"),
    ])
    return InlineKeyboardMarkup(rows)