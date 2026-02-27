#!/usr/bin/env python3
"""
neuralclaw_cli.py — NeuralClaw CLI Tool

DEPRECATED: This file is superseded by interfaces/cli.py and will be removed
in the next release. Do not add new features here.
Use 'python main.py --interface cli' instead.

Standalone developer-facing CLI for managing skills, configuration, and the
running NeuralClaw agent. Equivalent of `clawhub` from OpenClaw.

Usage:
    neuralclaw <command> [args] [flags]
    python neuralclaw_cli.py <command> [args] [flags]
"""

import warnings
warnings.warn(
    "neuralclaw_cli.py is deprecated. Use 'python main.py --interface cli' instead.",
    DeprecationWarning,
    stacklevel=1,
)

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
import webbrowser
from datetime import datetime, timezone
from difflib import get_close_matches
from pathlib import Path
from typing import Any, Optional

# ── Rich (graceful fallback) ─────────────────────────────────────────────────

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    _RICH = True
except ImportError:
    _RICH = False

    class Console:  # type: ignore[no-redef]
        def __init__(self, **kw):
            self._no_color = kw.get("no_color", False)
        def print(self, *a, **kw):
            from interfaces.branding import safe_print
            safe_print(*a)

    class Panel:  # type: ignore[no-redef]
        def __init__(self, *a, **kw): self._text = str(a[0]) if a else ""
        def __str__(self): return self._text

    class Table:  # type: ignore[no-redef]
        def __init__(self, **kw): self._rows = []; self._cols = []
        def add_column(self, *a, **kw): self._cols.append(a[0] if a else "")
        def add_row(self, *a, **kw): self._rows.append(a)

    class Text:  # type: ignore[no-redef]
        def __init__(self, t="", **kw): self._t = t
        def __str__(self): return self._t


# ── Constants ────────────────────────────────────────────────────────────────

CLI_VERSION = "1.0.0"
REGISTRY_URL = "https://raw.githubusercontent.com/neuralclaw/skills/main/index.json"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_NO_ROOT = 2
EXIT_NETWORK = 3
EXIT_DOCTOR = 4

CATEGORIES = [
    "builtin", "cyber", "developer", "personal",
    "data", "automation", "system", "meta",
]


# ── Project Root Detection ───────────────────────────────────────────────────

def find_project_root(override: Optional[str] = None) -> Optional[Path]:
    """Walk up from CWD looking for main.py + skills/. Fallback to env var."""
    if override:
        p = Path(override).resolve()
        if p.exists():
            return p
    env = os.environ.get("NEURALCLAW_ROOT")
    if env:
        p = Path(env).resolve()
        if p.exists():
            return p
    cwd = Path.cwd()
    for d in [cwd, *cwd.parents]:
        if (d / "main.py").exists() and (d / "skills").is_dir():
            return d
    return None


def require_root(args) -> Path:
    """Return project root or print error and exit."""
    root = find_project_root(getattr(args, "root", None))
    if root is None:
        print("✗  NeuralClaw project root not found.", file=sys.stderr)
        print("   Run this command from inside your NeuralClaw project,", file=sys.stderr)
        print("   or set NEURALCLAW_ROOT or use --root <path>.", file=sys.stderr)
        sys.exit(EXIT_NO_ROOT)
    return root


# ── Metadata Helpers ─────────────────────────────────────────────────────────

def _metadata_path(root: Path) -> Path:
    return root / "skills" / "plugins" / ".metadata.json"

def _disabled_path(root: Path) -> Path:
    return root / "skills" / "plugins" / ".disabled.json"

def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default if default is not None else {}
    try:
        return json.loads(path.read_text("utf-8"))
    except (json.JSONDecodeError, OSError):
        return default if default is not None else {}

def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")

def _read_metadata(root: Path) -> dict:
    return _read_json(_metadata_path(root), {})

def _write_metadata(root: Path, data: dict) -> None:
    _write_json(_metadata_path(root), data)

def _read_disabled(root: Path) -> list:
    return _read_json(_disabled_path(root), [])

def _write_disabled(root: Path, data: list) -> None:
    _write_json(_disabled_path(root), data)


# ── Registry / Network ──────────────────────────────────────────────────────

_BUILTIN_INDEX: dict[str, dict] = {
    "web-search":  {"name": "web_search",              "description": "Search the web using DuckDuckGo",         "source": "builtin",  "file": "web_search.py",              "category": "builtin"},
    "web-fetch":   {"name": "web_fetch",               "description": "Fetch and parse web pages",               "source": "builtin",  "file": "web_fetch.py",               "category": "builtin"},
    "filesystem":  {"name": "filesystem",              "description": "Read and write local files",              "source": "builtin",  "file": "filesystem.py",              "category": "builtin"},
    "terminal":    {"name": "terminal",                "description": "Run terminal commands in sandboxed shell","source": "builtin",  "file": "terminal.py",                "category": "builtin"},
    "weather":     {"name": "personal_weather_fetch",  "description": "Get current weather for a location",     "source": "plugin",   "file": "personal_weather_fetch.py",  "category": "personal"},
    "git-log":     {"name": "dev_git_log",             "description": "View git commit history",                 "source": "plugin",   "file": "dev_git_log.py",             "category": "developer"},
    "git-diff":    {"name": "dev_git_diff",            "description": "Show git diffs for files/commits",       "source": "plugin",   "file": "dev_git_diff.py",            "category": "developer"},
    "git-blame":   {"name": "dev_git_blame",           "description": "Show git blame for a file",              "source": "plugin",   "file": "dev_git_blame.py",           "category": "developer"},
    "port-scan":   {"name": "cyber_port_scan",         "description": "Scan ports on a target host",            "source": "plugin",   "file": "cyber_port_scan.py",         "category": "cyber"},
    "dns-enum":    {"name": "cyber_dns_enum",          "description": "Enumerate DNS records",                  "source": "plugin",   "file": "cyber_dns_enum.py",          "category": "cyber"},
    "disk-usage":  {"name": "system_disk_usage",       "description": "Show disk usage statistics",             "source": "plugin",   "file": "system_disk_usage.py",       "category": "system"},
    "news":        {"name": "personal_news_digest",    "description": "Fetch a personalised news digest",       "source": "plugin",   "file": "personal_news_digest.py",    "category": "personal"},
    "reminder":    {"name": "personal_reminder_set",   "description": "Set and manage personal reminders",      "source": "plugin",   "file": "personal_reminder_set.py",    "category": "personal"},
    "daily-brief": {"name": "meta_daily_assistant",    "description": "Daily briefing and task overview",       "source": "plugin",   "file": "meta_daily_assistant.py",     "category": "meta"},
}


def _fetch_url(url: str, timeout: int = 10) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "NeuralClaw/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_registry() -> dict:
    try:
        data = _fetch_url(REGISTRY_URL, timeout=5)
        return json.loads(data)
    except Exception:
        return dict(_BUILTIN_INDEX)


def _validate_skill_file(path: Path) -> tuple[bool, str]:
    if path.suffix == ".md":
        content = path.read_text()
        if "name:" not in content:
            return False, "Markdown skill must have 'name:' in frontmatter."
        return True, "ok"
    if path.suffix != ".py":
        return False, f"Unsupported file type: {path.suffix}. Expected .py or .md"
    try:
        ast.parse(path.read_text())
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    source = path.read_text()
    if "SkillBase" not in source and "SkillManifest" not in source:
        return False, "No SkillBase or SkillManifest found — not a valid NeuralClaw skill."
    return True, "ok"


# ── YAML Config Helpers ──────────────────────────────────────────────────────

def _load_yaml(path: Path) -> dict:
    import yaml
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _save_yaml(path: Path, data: dict) -> None:
    import yaml
    with path.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

def _get_dotted(data: dict, key: str) -> Any:
    parts = key.split(".")
    for p in parts:
        if isinstance(data, dict) and p in data:
            data = data[p]
        else:
            return None
    return data

def _set_dotted(data: dict, key: str, value: Any) -> None:
    parts = key.split(".")
    for p in parts[:-1]:
        if p not in data or not isinstance(data[p], dict):
            data[p] = {}
        data = data[p]
    # Auto-convert types
    if isinstance(value, str):
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
    data[parts[-1]] = value


# ══════════════════════════════════════════════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_install(args) -> int:
    """Install a skill by slug, URL, or local path."""
    root = require_root(args)
    sys.path.insert(0, str(root))
    try:
        from onboard.skill_installer import run_install
        return run_install(args.slug, force=args.force)
    except ImportError:
        print("✗  Could not import skill installer.", file=sys.stderr)
        return EXIT_ERROR


def cmd_remove(args) -> int:
    """Remove an installed plugin skill."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    slug = args.slug
    metadata = _read_metadata(root)
    registry = _fetch_registry()
    plugins_dir = root / "skills" / "plugins"

    # Resolve filename
    entry = metadata.get(slug) or registry.get(slug)
    if entry:
        filename = entry.get("file", f"{slug.replace('-', '_')}.py")
    else:
        filename = f"{slug.replace('-', '_')}.py"

    filepath = plugins_dir / filename

    # Check if it's a builtin
    builtin_entry = registry.get(slug, {})
    if builtin_entry.get("source") == "builtin":
        console.print(f"[yellow]⚠[/yellow]  '{slug}' is a built-in skill and cannot be removed." if _RICH
                       else f"⚠  '{slug}' is a built-in skill and cannot be removed.")
        console.print("   Use 'neuralclaw disable' instead." if not _RICH else
                       "   Use [bold]neuralclaw disable[/bold] instead.")
        return EXIT_OK

    if not filepath.exists():
        console.print(f"[yellow]⚠[/yellow]  Skill '{slug}' is not installed." if _RICH
                       else f"⚠  Skill '{slug}' is not installed.")
        return EXIT_OK

    # Confirm
    if not args.yes:
        answer = input(f"Remove '{slug}' ({filepath.name})? [y/N] ").strip().lower()
        if answer not in ("y", "yes"):
            console.print("Cancelled.")
            return EXIT_OK

    filepath.unlink()
    if slug in metadata:
        del metadata[slug]
        _write_metadata(root, metadata)

    # Also remove from disabled list
    disabled = _read_disabled(root)
    if slug in disabled:
        disabled.remove(slug)
        _write_disabled(root, disabled)

    console.print(f"[green]✓[/green]  Removed '{slug}' ({filename})" if _RICH
                   else f"✓  Removed '{slug}' ({filename})")
    return EXIT_OK


def cmd_update(args) -> int:
    """Update installed skills to latest remote versions."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    metadata = _read_metadata(root)
    registry = _fetch_registry()
    plugins_dir = root / "skills" / "plugins"

    slugs = []
    if args.all:
        slugs = list(metadata.keys())
    elif args.slug:
        slugs = [args.slug]
    else:
        console.print("Usage: neuralclaw update <slug> or neuralclaw update --all")
        return EXIT_ERROR

    updated = skipped = failed = 0
    for slug in slugs:
        entry = registry.get(slug, metadata.get(slug, {}))
        if entry.get("source") == "builtin":
            if not args.quiet:
                console.print(f"  ⊘  {slug} — builtin, skipped")
            skipped += 1
            continue
        url = entry.get("url")
        if not url:
            if not args.quiet:
                console.print(f"  ⊘  {slug} — no remote URL, skipped")
            skipped += 1
            continue
        filename = entry.get("file", f"{slug.replace('-', '_')}.py")
        dest = plugins_dir / filename

        if args.dry_run:
            console.print(f"  →  Would update {slug} from {url}")
            updated += 1
            continue

        try:
            data = _fetch_url(url)
            with tempfile.NamedTemporaryFile(suffix=filename, delete=False) as tmp:
                tmp.write(data)
                tmp_path = Path(tmp.name)
            ok, msg = _validate_skill_file(tmp_path)
            if not ok:
                console.print(f"  [red]✗[/red]  {slug} — validation failed: {msg}" if _RICH
                               else f"  ✗  {slug} — validation failed: {msg}")
                tmp_path.unlink(missing_ok=True)
                failed += 1
                continue
            shutil.move(str(tmp_path), str(dest))
            if slug in metadata:
                metadata[slug]["updated_at"] = datetime.now(timezone.utc).isoformat()
            updated += 1
            console.print(f"  [green]✓[/green]  {slug} updated" if _RICH else f"  ✓  {slug} updated")
        except Exception as e:
            console.print(f"  [red]✗[/red]  {slug} — {e}" if _RICH else f"  ✗  {slug} — {e}")
            failed += 1

    _write_metadata(root, metadata)
    console.print(f"\n{updated} updated, {skipped} skipped, {failed} failed")
    return EXIT_OK if failed == 0 else EXIT_ERROR


def cmd_list(args) -> int:
    """List installed or all available skills."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    metadata = _read_metadata(root)
    disabled = _read_disabled(root)
    registry = _fetch_registry()
    plugins_dir = root / "skills" / "plugins"
    builtin_dir = root / "skills" / "builtin"

    # Build rows
    rows = []
    if args.all:
        # All registry entries
        for slug, entry in sorted(registry.items()):
            cat = entry.get("category", "")
            if args.category and cat != args.category:
                continue
            filename = entry.get("file", "")
            installed = (plugins_dir / filename).exists() or (builtin_dir / filename).exists()
            is_disabled = slug in disabled
            status = "⊘ disabled" if is_disabled else ("✓ active" if installed else "not installed")
            version = metadata.get(slug, {}).get("version", entry.get("version", "—"))
            rows.append((slug, entry.get("name", ""), cat, status, version))
    else:
        # Installed skills from metadata
        for slug, meta in sorted(metadata.items()):
            cat = meta.get("category", "")
            if args.category and cat != args.category:
                continue
            is_disabled = slug in disabled
            status = "⊘ disabled" if is_disabled else "✓ active"
            rows.append((slug, meta.get("file", ""), cat, status, meta.get("version", "—")))
        # Also scan plugins dir for untracked
        for f in sorted(plugins_dir.glob("*.py")):
            if f.name.startswith("_") or f.name.startswith("."):
                continue
            tracked = any(m.get("file") == f.name for m in metadata.values())
            if not tracked:
                slug_guess = f.stem
                is_disabled = slug_guess in disabled
                status = "⊘ disabled" if is_disabled else "✓ active"
                rows.append((slug_guess, f.name, "?", status, "—"))

    if args.json_output:
        print(json.dumps([{"slug": r[0], "name": r[1], "category": r[2], "status": r[3], "version": r[4]} for r in rows], indent=2))
        return EXIT_OK

    if _RICH:
        table = Table(title="NeuralClaw Skills", show_lines=True)
        table.add_column("Slug", style="bold")
        table.add_column("Name")
        table.add_column("Category")
        table.add_column("Status")
        table.add_column("Version")
        for r in rows:
            status_style = "green" if "active" in r[3] else ("yellow" if "disabled" in r[3] else "dim")
            table.add_row(r[0], r[1], r[2], f"[{status_style}]{r[3]}[/{status_style}]", r[4])
        console.print(table)
    else:
        fmt = "{:<20} {:<28} {:<12} {:<12} {:<8}"
        console.print(fmt.format("Slug", "Name", "Category", "Status", "Version"))
        console.print("─" * 82)
        for r in rows:
            console.print(fmt.format(*r))

    active = sum(1 for r in rows if "active" in r[3])
    dis = sum(1 for r in rows if "disabled" in r[3])
    console.print(f"\n{len(rows)} skills  ({active} active, {dis} disabled)")
    return EXIT_OK


def cmd_search(args) -> int:
    """Search the registry by name, description, category."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    registry = _fetch_registry()
    query = args.query.lower()
    plugins_dir = root / "skills" / "plugins"

    results = []
    for slug, entry in sorted(registry.items()):
        if args.category and entry.get("category", "") != args.category:
            continue
        searchable = f"{slug} {entry.get('name', '')} {entry.get('description', '')} {entry.get('category', '')}".lower()
        if query in searchable:
            installed = (plugins_dir / entry.get("file", "")).exists()
            if args.installed and not installed:
                continue
            results.append((slug, entry.get("category", ""), entry.get("description", ""), installed))

    if not results:
        console.print(f'No results for "{args.query}"')
        return EXIT_OK

    console.print(f'Search results for "{args.query}"\n')
    for slug, cat, desc, inst in results:
        marker = " [installed]" if inst else ""
        console.print(f"  {slug:<18} {cat:<12} {desc}{marker}")
    console.print(f"\n{len(results)} results  ·  Install with: neuralclaw install <slug>")
    return EXIT_OK


def cmd_info(args) -> int:
    """Show full details for a skill."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    registry = _fetch_registry()
    metadata = _read_metadata(root)
    slug = args.slug

    entry = registry.get(slug)
    meta = metadata.get(slug, {})
    if not entry and not meta:
        console.print(f"✗  Skill '{slug}' not found in registry.")
        close = get_close_matches(slug, list(registry.keys()), n=3, cutoff=0.5)
        if close:
            console.print(f"   Did you mean: {', '.join(close)}?")
        return EXIT_ERROR

    info = {**(entry or {}), **meta}
    plugins_dir = root / "skills" / "plugins"
    installed = (plugins_dir / info.get("file", "")).exists()

    lines = [
        f"  Name:        {info.get('name', slug)}",
        f"  Category:    {info.get('category', '—')}",
        f"  Version:     {info.get('version', '—')}",
        f"  Risk Level:  {info.get('risk_level', '—')}",
        f"  Status:      {'✓ Installed' if installed else '✗ Not installed'}",
    ]
    if meta.get("installed_at"):
        lines.append(f"  Installed:   {meta['installed_at']}")
    if info.get("file"):
        lines.append(f"  File:        skills/plugins/{info['file']}")
    if info.get("description"):
        lines.append(f"\n  Description:")
        lines.append(f"  {info['description']}")

    if _RICH:
        console.print(Panel("\n".join(lines), title=slug, border_style="cyan"))
    else:
        console.print(f"── {slug} ──")
        for l in lines:
            console.print(l)
    return EXIT_OK


def cmd_create(args) -> int:
    """Scaffold a new skill from a template."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    slug = args.slug
    name = slug.replace("-", "_")
    class_name = "".join(w.capitalize() for w in slug.split("-")) + "Skill"

    # Determine type
    if args.python:
        skill_type = "python"
    elif args.markdown:
        skill_type = "markdown"
    else:
        answer = input("Skill type? [p]ython / [m]arkdown: ").strip().lower()
        skill_type = "markdown" if answer.startswith("m") else "python"

    desc = input("Description: ").strip() if not args.quiet else f"A custom {name} skill"
    category = input(f"Category ({', '.join(CATEGORIES)}): ").strip() if not args.quiet else "personal"

    output_dir = Path(args.output) if args.output else root / "skills" / "plugins"
    output_dir.mkdir(parents=True, exist_ok=True)

    if skill_type == "python":
        ext = ".py"
        content = textwrap.dedent(f'''\
            """
            {name} — NeuralClaw Skill
            {desc}
            """
            from skills.base import SkillBase
            from skills.types import SkillManifest, SkillResult, RiskLevel

            class {class_name}(SkillBase):
                manifest = SkillManifest(
                    name="{name}",
                    version="1.0.0",
                    description="{desc}",
                    category="{category}",
                    risk_level=RiskLevel.LOW,
                    capabilities=frozenset(),
                    parameters={{
                        "type": "object",
                        "properties": {{
                            "input": {{"type": "string", "description": "The input to process"}}
                        }},
                        "required": ["input"],
                    }},
                )

                async def execute(self, input: str, **kwargs) -> SkillResult:
                    skill_call_id = kwargs.get("_skill_call_id", "")
                    try:
                        result = f"Processed: {{input}}"
                        return SkillResult.ok(self.manifest.name, skill_call_id, result)
                    except Exception as e:
                        return SkillResult.fail(self.manifest.name, skill_call_id, str(e))
        ''')
    else:
        ext = ".md"
        content = textwrap.dedent(f"""\
            ---
            name: {name}
            version: 1.0.0
            description: {desc}
            category: {category}
            risk_level: LOW
            parameters:
              type: object
              properties:
                input:
                  type: string
              required: [input]
            ---

            You are a helpful assistant. Process the following input:

            {{{{input}}}}
        """)

    out_file = output_dir / f"{name}{ext}"
    if out_file.exists():
        console.print(f"[yellow]⚠[/yellow]  {out_file} already exists." if _RICH
                       else f"⚠  {out_file} already exists.")
        return EXIT_ERROR

    out_file.write_text(content, encoding="utf-8")
    console.print(f"[green]✓[/green]  Created {out_file.relative_to(root)}" if _RICH
                   else f"✓  Created {out_file.relative_to(root)}")
    return EXIT_OK


def cmd_enable(args) -> int:
    """Re-enable a previously disabled skill."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    disabled = _read_disabled(root)
    slug = args.slug
    if slug not in disabled:
        console.print(f"'{slug}' is not disabled.")
        return EXIT_OK
    disabled.remove(slug)
    _write_disabled(root, disabled)
    console.print(f"[green]✓[/green]  Enabled '{slug}'" if _RICH else f"✓  Enabled '{slug}'")
    return EXIT_OK


def cmd_disable(args) -> int:
    """Disable a skill without deleting it."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    disabled = _read_disabled(root)
    slug = args.slug
    if slug in disabled:
        console.print(f"'{slug}' is already disabled.")
        return EXIT_OK
    disabled.append(slug)
    _write_disabled(root, disabled)
    console.print(f"[green]✓[/green]  Disabled '{slug}'" if _RICH else f"✓  Disabled '{slug}'")
    return EXIT_OK


def cmd_sync(args) -> int:
    """Sync installed skills metadata with registry."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    registry = _fetch_registry()
    metadata = _read_metadata(root)
    plugins_dir = root / "skills" / "plugins"

    tracked = unrecognized = 0
    for f in sorted(plugins_dir.glob("*.py")):
        if f.name.startswith("_") or f.name.startswith("."):
            continue
        # Find matching registry entry
        match_slug = None
        for slug, entry in registry.items():
            if entry.get("file") == f.name:
                match_slug = slug
                break
        if match_slug:
            if match_slug not in metadata or args.all:
                entry = registry[match_slug]
                metadata[match_slug] = {
                    "file": f.name,
                    "installed_at": metadata.get(match_slug, {}).get("installed_at", datetime.now(timezone.utc).isoformat()),
                    "version": entry.get("version", "1.0.0"),
                    "source": entry.get("source", "plugin"),
                    "category": entry.get("category", ""),
                }
            console.print(f"  [green]✓[/green]  {match_slug:<18} → already tracked (v{metadata[match_slug].get('version', '?')})" if _RICH
                           else f"  ✓  {match_slug:<18} → already tracked")
            tracked += 1
        else:
            console.print(f"  ?  {f.name:<18} → not in registry (unrecognized)")
            unrecognized += 1

    # Also check .md skills
    for f in sorted(plugins_dir.glob("*.md")):
        if f.name.startswith("."):
            continue
        console.print(f"  ?  {f.name:<18} → markdown skill (unrecognized)")
        unrecognized += 1

    _write_metadata(root, metadata)
    console.print(f"\nSync complete. {tracked} tracked, {unrecognized} unrecognized.")
    return EXIT_OK


def cmd_publish(args) -> int:
    """Submit a skill to the community registry via GitHub issue."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    filepath = Path(args.file)
    if not filepath.exists():
        filepath = root / "skills" / "plugins" / args.file
    if not filepath.exists():
        console.print(f"✗  File not found: {args.file}")
        return EXIT_ERROR

    ok, msg = _validate_skill_file(filepath)
    if not ok:
        console.print(f"✗  Validation failed: {msg}")
        return EXIT_ERROR

    source = filepath.read_text()
    name = filepath.stem
    title = f"[Skill Submission] {name}"
    body = f"## Skill: {name}\n\n**File:** {filepath.name}\n\n```python\n{source[:2000]}\n```"

    if args.dry_run:
        console.print(f"Would create GitHub issue:\n  Title: {title}\n  Body preview: {body[:200]}...")
        return EXIT_OK

    url = f"https://github.com/neuralclaw/skills/issues/new?title={urllib.request.quote(title)}&body={urllib.request.quote(body[:2000])}"
    console.print(f"Opening browser to submit skill '{name}'...")
    webbrowser.open(url)
    return EXIT_OK


def cmd_doctor(args) -> int:
    """Diagnose NeuralClaw installation issues."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    warnings = errors = 0

    def ok(msg):
        console.print(f"  [green]✓[/green]  {msg}" if _RICH else f"  ✓  {msg}")
    def warn(msg):
        nonlocal warnings; warnings += 1
        console.print(f"  [yellow]⚠[/yellow]  {msg}" if _RICH else f"  ⚠  {msg}")
    def fail(msg):
        nonlocal errors; errors += 1
        console.print(f"  [red]✗[/red]  {msg}" if _RICH else f"  ✗  {msg}")

    console.print(f"NeuralClaw Doctor  v{CLI_VERSION}")
    console.print("─" * 45)

    # Python version
    v = sys.version_info
    if v >= (3, 11):
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        fail(f"Python {v.major}.{v.minor}.{v.micro} (requires >= 3.11)")

    # Project root
    ok(f"Project root: {root}")

    # .env
    env_file = root / ".env"
    if env_file.exists() and env_file.stat().st_size > 0:
        ok(".env file present")
    elif env_file.exists():
        warn(".env file is empty")
    else:
        fail(".env file not found")

    # Config
    config_path = root / "config" / "config.yaml"
    if config_path.exists():
        cfg = _load_yaml(config_path)
        provider = cfg.get("llm", {}).get("default_provider", "?")
        ok(f"LLM provider: {provider}")

        # Provider-specific checks
        if provider == "ollama":
            try:
                urllib.request.urlopen("http://localhost:11434", timeout=3)
                ok("Ollama reachable at http://localhost:11434")
            except Exception:
                fail("Ollama not reachable at http://localhost:11434")
        elif provider in ("openai", "anthropic", "gemini", "bytez", "openrouter"):
            key_env = f"{provider.upper()}_API_KEY"
            if os.environ.get(key_env):
                ok(f"{key_env} is set")
            else:
                fail(f"{key_env} not set")
    else:
        fail("config/config.yaml not found")

    # skills/plugins/
    plugins_dir = root / "skills" / "plugins"
    if plugins_dir.exists() and os.access(plugins_dir, os.W_OK):
        skill_count = len(list(plugins_dir.glob("*.py")))
        ok(f"skills/plugins/ writable ({skill_count} skills)")
    else:
        fail("skills/plugins/ missing or not writable")

    # Plugin syntax check
    bad_plugins = 0
    for f in plugins_dir.glob("*.py"):
        if f.name.startswith("_"):
            continue
        try:
            ast.parse(f.read_text())
        except SyntaxError:
            bad_plugins += 1
    if bad_plugins == 0:
        ok("All plugin files valid Python")
    else:
        fail(f"{bad_plugins} plugin file(s) have syntax errors")

    # Data dirs
    for d in ["data/chroma", "data/sqlite"]:
        dp = root / d
        if dp.exists():
            ok(f"{d}/ exists")
        else:
            warn(f"{d}/ does not exist")

    # Telegram token
    if os.environ.get("TELEGRAM_BOT_TOKEN"):
        ok("TELEGRAM_BOT_TOKEN is set")
    else:
        warn("TELEGRAM_BOT_TOKEN not set (only needed for telegram interface)")

    console.print(f"\n{warnings} warning(s) · {errors} error(s)")
    return EXIT_OK if errors == 0 else EXIT_DOCTOR


def cmd_status(args) -> int:
    """Show agent runtime status."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    config_path = root / "config" / "config.yaml"
    cfg = _load_yaml(config_path)
    pid_file = root / "data" / "agent.pid"

    console.print("NeuralClaw Status")
    console.print("─" * 45)

    # Process
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            console.print(f"  Process:     ● Running  (PID {pid})")
        except (ValueError, OSError):
            console.print("  Process:     ○ Not running (stale PID file)")
    else:
        console.print("  Process:     ○ Not running")
        console.print("  Run: python main.py")

    # Config info
    provider = cfg.get("llm", {}).get("default_provider", "?")
    model = cfg.get("llm", {}).get("default_model", "?")
    console.print(f"  Provider:    {provider} / {model}")

    # Skills count
    plugins_dir = root / "skills" / "plugins"
    disabled = _read_disabled(root)
    skill_files = [f for f in plugins_dir.glob("*.py") if not f.name.startswith("_")]
    active = len(skill_files) - len(disabled)
    console.print(f"\n  Skills")
    console.print(f"    {len(skill_files)} installed  ({active} active, {len(disabled)} disabled)")

    # Memory stats
    sqlite_path = root / cfg.get("memory", {}).get("sqlite_path", "data/sqlite/episodes.db")
    chroma_dir = root / cfg.get("memory", {}).get("chroma_persist_dir", "data/chroma")
    console.print(f"\n  Memory")
    if sqlite_path.exists():
        size_mb = sqlite_path.stat().st_size / (1024 * 1024)
        console.print(f"    SQLite: {size_mb:.1f} MB ({sqlite_path.name})")
    else:
        console.print("    SQLite: not present")
    if chroma_dir.exists():
        console.print(f"    ChromaDB: {chroma_dir}")
    else:
        console.print("    ChromaDB: not present")

    # Logs
    log_dir = root / cfg.get("logging", {}).get("log_dir", "data/logs")
    if log_dir.exists():
        log_files = list(log_dir.glob("*.log"))
        if log_files:
            latest = max(log_files, key=lambda f: f.stat().st_mtime)
            size_mb = latest.stat().st_size / (1024 * 1024)
            console.print(f"\n  Logs")
            console.print(f"    {latest.relative_to(root)}  ({size_mb:.1f} MB)")

    return EXIT_OK


def cmd_config(args) -> int:
    """Read or write config values."""
    root = require_root(args)
    console = Console(no_color=args.no_color)
    config_path = root / "config" / "config.yaml"
    cfg = _load_yaml(config_path)

    if not args.key:
        # Print full config
        if args.json_output:
            print(json.dumps(cfg, indent=2, default=str))
        else:
            import yaml
            console.print(yaml.dump(cfg, default_flow_style=False, sort_keys=False))
        return EXIT_OK

    if not args.value:
        # Read single key
        val = _get_dotted(cfg, args.key)
        if val is None:
            console.print(f"✗  Key '{args.key}' not found in config.")
            return EXIT_ERROR
        if args.raw:
            print(val)
        else:
            console.print(f"{args.key} = {val}")
        return EXIT_OK

    # Write
    if "api_key" in args.key.lower():
        console.print("✗  API keys belong in .env, not config.yaml.")
        return EXIT_ERROR

    _set_dotted(cfg, args.key, args.value)
    _save_yaml(config_path, cfg)
    console.print(f"[green]✓[/green]  Set {args.key} = {_get_dotted(cfg, args.key)}" if _RICH
                   else f"✓  Set {args.key} = {_get_dotted(cfg, args.key)}")
    return EXIT_OK


def cmd_logs(args) -> int:
    """View or stream agent log output."""
    root = require_root(args)
    config_path = root / "config" / "config.yaml"
    cfg = _load_yaml(config_path)
    log_dir = root / cfg.get("logging", {}).get("log_dir", "data/logs")

    if not log_dir.exists():
        print(f"✗  Log directory not found: {log_dir}", file=sys.stderr)
        return EXIT_ERROR

    log_files = sorted(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)
    if not log_files:
        print("✗  No log files found.", file=sys.stderr)
        return EXIT_ERROR

    log_file = log_files[0]
    tail_n = args.tail or 50
    level_filter = args.level.upper() if args.level else None

    if args.follow:
        cmd = ["tail", "-f", "-n", str(tail_n), str(log_file)]
        try:
            proc = subprocess.Popen(cmd)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return EXIT_OK

    # Read last N lines
    lines = log_file.read_text(errors="replace").splitlines()
    lines = lines[-tail_n:]
    if level_filter:
        lines = [l for l in lines if level_filter in l.upper()]
    if args.session:
        lines = [l for l in lines if args.session in l]
    for line in lines:
        if args.json_output:
            print(line)
        else:
            print(line)
    return EXIT_OK


def cmd_version(args) -> int:
    """Print version information."""
    root = find_project_root(getattr(args, "root", None))
    console = Console(no_color=args.no_color)

    core_version = "—"
    skills_count = 0
    config_path_str = "—"

    if root:
        cfg_path = root / "config" / "config.yaml"
        config_path_str = str(cfg_path)
        if cfg_path.exists():
            cfg = _load_yaml(cfg_path)
            core_version = cfg.get("agent", {}).get("version", "1.0.0")
        plugins = root / "skills" / "plugins"
        if plugins.exists():
            skills_count = len([f for f in plugins.glob("*.py") if not f.name.startswith("_")])

    console.print(f"neuralclaw CLI       v{CLI_VERSION}")
    console.print(f"NeuralClaw core      v{core_version}")
    console.print(f"Python               {platform.python_version()}")
    console.print(f"Platform             {sys.platform} / {platform.machine()}")
    console.print(f"Skills               {skills_count} installed")
    console.print(f"Config               {config_path_str}")
    return EXIT_OK


# ══════════════════════════════════════════════════════════════════════════════
# ARGUMENT PARSER
# ══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralclaw",
        description="NeuralClaw — Developer CLI for managing skills, config, and agent.",
    )
    # Global flags
    parser.add_argument("--no-color", action="store_true", help="Disable colored output")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Machine-readable JSON output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    parser.add_argument("--verbose", "-v", action="store_true", help="Extra debug info")
    parser.add_argument("--root", default=None, help="Override project root detection")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # install
    p = sub.add_parser("install", help="Install a skill")
    p.add_argument("slug", help="Skill slug, URL, or local path")
    p.add_argument("--force", action="store_true", help="Overwrite if exists")
    p.add_argument("--dev", action="store_true", help="Symlink instead of copy")
    p.add_argument("--dry-run", action="store_true", help="Show what would happen")

    # remove
    p = sub.add_parser("remove", help="Remove an installed skill")
    p.add_argument("slug", help="Skill slug")
    p.add_argument("--yes", "-y", action="store_true", help="Skip confirmation")

    # update
    p = sub.add_parser("update", help="Update installed skills")
    p.add_argument("slug", nargs="?", default=None, help="Skill slug (or --all)")
    p.add_argument("--all", action="store_true", help="Update all installed skills")
    p.add_argument("--dry-run", action="store_true", help="Show what would be updated")

    # list
    p = sub.add_parser("list", help="List installed skills")
    p.add_argument("--all", action="store_true", help="Show all available skills")
    p.add_argument("--category", choices=CATEGORIES, help="Filter by category")

    # search
    p = sub.add_parser("search", help="Search the skill registry")
    p.add_argument("query", help="Search query")
    p.add_argument("--category", choices=CATEGORIES, help="Filter by category")
    p.add_argument("--installed", action="store_true", help="Only show installed")

    # info
    p = sub.add_parser("info", help="Show skill details")
    p.add_argument("slug", help="Skill slug")

    # create
    p = sub.add_parser("create", help="Scaffold a new skill")
    p.add_argument("slug", help="Skill slug")
    p.add_argument("--python", action="store_true", help="Python template")
    p.add_argument("--markdown", action="store_true", help="Markdown template")
    p.add_argument("--output", default=None, help="Custom output path")

    # enable
    p = sub.add_parser("enable", help="Re-enable a disabled skill")
    p.add_argument("slug", help="Skill slug")

    # disable
    p = sub.add_parser("disable", help="Disable a skill without removing")
    p.add_argument("slug", help="Skill slug")

    # sync
    p = sub.add_parser("sync", help="Sync skill metadata with registry")
    p.add_argument("--all", action="store_true", help="Re-sync all skills")
    p.add_argument("--publish", action="store_true", help="Submit unrecognized to registry")

    # publish
    p = sub.add_parser("publish", help="Submit a skill to the registry")
    p.add_argument("file", help="Skill file path")
    p.add_argument("--token", default=None, help="GitHub token for API submission")
    p.add_argument("--dry-run", action="store_true", help="Show issue body without opening")

    # doctor
    sub.add_parser("doctor", help="Diagnose installation issues")

    # status
    sub.add_parser("status", help="Show agent runtime status")

    # config
    p = sub.add_parser("config", help="Read/write config values")
    p.add_argument("key", nargs="?", default=None, help="Dotted config key")
    p.add_argument("value", nargs="?", default=None, help="Value to set")
    p.add_argument("--raw", action="store_true", help="Print raw value")

    # logs
    p = sub.add_parser("logs", help="View agent logs")
    p.add_argument("--tail", type=int, default=50, help="Number of lines (default: 50)")
    p.add_argument("--follow", "-f", action="store_true", help="Stream logs in real-time")
    p.add_argument("--level", default=None, help="Filter by log level")
    p.add_argument("--session", default=None, help="Filter by session ID")

    # version
    sub.add_parser("version", help="Show version info")

    return parser


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

_DISPATCH = {
    "install": cmd_install,
    "remove": cmd_remove,
    "update": cmd_update,
    "list": cmd_list,
    "search": cmd_search,
    "info": cmd_info,
    "create": cmd_create,
    "enable": cmd_enable,
    "disable": cmd_disable,
    "sync": cmd_sync,
    "publish": cmd_publish,
    "doctor": cmd_doctor,
    "status": cmd_status,
    "config": cmd_config,
    "logs": cmd_logs,
    "version": cmd_version,
}


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return EXIT_OK

    handler = _DISPATCH.get(args.command)
    if handler is None:
        parser.print_help()
        return EXIT_ERROR

    # Ensure json_output is always available
    if not hasattr(args, "json_output"):
        args.json_output = False
    if not hasattr(args, "no_color"):
        args.no_color = False
    if not hasattr(args, "quiet"):
        args.quiet = False

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
