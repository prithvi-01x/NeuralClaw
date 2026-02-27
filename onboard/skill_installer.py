"""
onboard/skill_installer.py — NeuralClaw Skill Installer

Installs skills from:
    1. The built-in skill registry (GitHub-backed index)
    2. A direct GitHub URL (any public repo)
    3. A local .py or .md file path

Usage:
    python main.py install web-search
    python main.py install github.com/someone/my-skill
    python main.py install ./my_skill.py

Skills are dropped into skills/plugins/ and are live on next restart.
The installer validates the skill file before installing (syntax check +
manifest presence check) and refuses to overwrite existing skills unless
--force is passed.

Registry index: https://raw.githubusercontent.com/neuralclaw/skills/main/index.json
(Falls back to a built-in minimal index if the registry is unreachable.)
"""

from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    _RICH = True
except ImportError:
    _RICH = False

    class Console:  # type: ignore[no-redef]
        def print(self, *a, **kw):
            from interfaces.branding import safe_print
            safe_print(*a)

    class Progress:  # type: ignore[no-redef]
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def add_task(self, *a, **kw): return 0
        def update(self, *a, **kw): pass

    class SpinnerColumn: pass  # type: ignore[no-redef]
    class TextColumn:  # type: ignore[no-redef]
        def __init__(self, *a): pass


PROJECT_ROOT  = Path(__file__).parent.parent
PLUGINS_DIR   = PROJECT_ROOT / "skills" / "plugins"
REGISTRY_URL  = "https://raw.githubusercontent.com/neuralclaw/skills/main/index.json"

# ── Built-in fallback index ───────────────────────────────────────────────────
# A minimal set of well-known skills available without network access to
# the registry. Keys are the install name; values are GitHub raw URLs.

_BUILTIN_INDEX: dict[str, dict] = {
    "web-search": {
        "name": "web_search",
        "description": "Search the web using DuckDuckGo",
        "source": "builtin",
        "file": "web_search.py",
    },
    "web-fetch": {
        "name": "web_fetch",
        "description": "Fetch and parse web pages",
        "source": "builtin",
        "file": "web_fetch.py",
    },
    "filesystem": {
        "name": "filesystem",
        "description": "Read and write local files",
        "source": "builtin",
        "file": "filesystem.py",
    },
    "terminal": {
        "name": "terminal",
        "description": "Run terminal commands in a sandboxed shell",
        "source": "builtin",
        "file": "terminal.py",
    },
    "reminder": {
        "name": "personal_reminder_set",
        "description": "Set and manage personal reminders",
        "source": "plugin",
        "file": "personal_reminder_set.py",
    },
    "daily-brief": {
        "name": "meta_daily_assistant",
        "description": "Daily briefing and task overview",
        "source": "plugin",
        "file": "meta_daily_assistant.py",
    },
    "news": {
        "name": "personal_news_digest",
        "description": "Fetch a personalised news digest",
        "source": "plugin",
        "file": "personal_news_digest.py",
    },
    "weather": {
        "name": "personal_weather_fetch",
        "description": "Get current weather for a location",
        "source": "plugin",
        "file": "personal_weather_fetch.py",
    },
    "git-log": {
        "name": "dev_git_log",
        "description": "View git commit history",
        "source": "plugin",
        "file": "dev_git_log.py",
    },
    "disk-usage": {
        "name": "system_disk_usage",
        "description": "Show disk usage statistics",
        "source": "plugin",
        "file": "system_disk_usage.py",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ok(c: Console, msg: str) -> None:
    c.print(f"[green]✓[/green] {msg}" if _RICH else f"✓ {msg}")

def _err(c: Console, msg: str) -> None:
    c.print(f"[red]✗[/red] {msg}" if _RICH else f"✗ {msg}")

def _info(c: Console, msg: str) -> None:
    c.print(f"[dim]{msg}[/dim]" if _RICH else msg)


def _fetch_url(url: str, timeout: int = 10) -> bytes:
    """Fetch a URL and return bytes. Raises urllib.error.URLError on failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "NeuralClaw/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _fetch_registry() -> dict:
    """Try to fetch the remote registry index. Falls back to _BUILTIN_INDEX."""
    try:
        data = _fetch_url(REGISTRY_URL, timeout=5)
        import json
        return json.loads(data)
    except Exception:
        return _BUILTIN_INDEX


def _validate_skill_file(path: Path) -> tuple[bool, str]:
    """
    Basic validation of a skill file before installing:
      1. Must be valid Python syntax
      2. Must contain 'SkillBase' (subclass) or be a markdown skill

    Returns (ok, message).
    """
    if path.suffix == ".md":
        content = path.read_text()
        if "name:" not in content:
            return False, "Markdown skill file must have a 'name:' in its frontmatter."
        return True, "ok"

    if path.suffix != ".py":
        return False, f"Unsupported file type: {path.suffix}. Expected .py or .md"

    try:
        source = path.read_text()
        ast.parse(source)
    except SyntaxError as e:
        return False, f"Syntax error in skill file: {e}"

    if "SkillBase" not in source and "SkillManifest" not in source:
        return False, (
            "File does not appear to be a NeuralClaw skill "
            "(no SkillBase or SkillManifest found)."
        )

    return True, "ok"


def _resolve_source(skill_name: str, registry: dict) -> Optional[tuple[str, str]]:
    """
    Resolve a skill name or URL to (download_url_or_path, dest_filename).

    Priority:
        1. Local file path
        2. Registry lookup (built-in alias)
        3. GitHub shorthand: username/repo
        4. Raw URL

    Returns None if unresolvable.
    """
    # 1. Local file
    local = Path(skill_name).expanduser()
    if local.exists() and local.is_file():
        return str(local), local.name

    # 2. Registry
    entry = registry.get(skill_name)
    if entry:
        source = entry.get("source", "")
        filename = entry.get("file", f"{skill_name.replace('-', '_')}.py")

        if source == "builtin":
            builtin_path = PROJECT_ROOT / "skills" / "builtin" / filename
            if builtin_path.exists():
                return str(builtin_path), filename
            return None, f"Built-in skill '{filename}' not found in skills/builtin/"

        if source == "plugin":
            plugin_path = PROJECT_ROOT / "skills" / "plugins" / filename
            if plugin_path.exists():
                return str(plugin_path), filename

        # Has a URL
        if "url" in entry:
            return entry["url"], filename

        return None, f"Registry entry for '{skill_name}' has no resolvable source."

    # 3. GitHub shorthand
    if "/" in skill_name and not skill_name.startswith("http"):
        parts = skill_name.strip("/").split("/")
        if len(parts) == 2:
            user, repo = parts
            url = f"https://raw.githubusercontent.com/{user}/{repo}/main/skill.py"
            filename = f"{repo.replace('-', '_')}.py"
            return url, filename
        if len(parts) >= 3:
            # github.com/user/repo/path/to/file.py
            if parts[0] in ("github.com", "raw.githubusercontent.com"):
                parts = parts[1:]
            if len(parts) >= 3:
                user, repo = parts[0], parts[1]
                file_path = "/".join(parts[2:])
                url = f"https://raw.githubusercontent.com/{user}/{repo}/main/{file_path}"
                filename = Path(file_path).name
                return url, filename

    # 4. Raw URL
    if skill_name.startswith("http://") or skill_name.startswith("https://"):
        filename = Path(skill_name.split("?")[0]).name or "custom_skill.py"
        return skill_name, filename

    return None, f"Cannot resolve skill source for '{skill_name}'."


# ── Main install command ──────────────────────────────────────────────────────

def run_install(skill_arg: str, force: bool = False) -> int:
    """
    Install a skill by name, path, or URL.
    Returns 0 on success, 1 on failure.
    """
    console = Console()

    console.print()
    console.print(f"[bold]Installing skill:[/bold] [cyan]{skill_arg}[/cyan]"
                  if _RICH else f"Installing skill: {skill_arg}")
    console.print()

    # Ensure plugins dir exists
    PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

    # Fetch registry
    _info(console, "Fetching skill registry…")
    registry = _fetch_registry()

    # Resolve source
    result = _resolve_source(skill_arg, registry)
    if result is None or (isinstance(result, tuple) and result[0] is None):
        msg = result[1] if isinstance(result, tuple) else f"Cannot resolve '{skill_arg}'."
        _err(console, msg)
        console.print(
            "\n[dim]Tip: run [bold]python main.py skills[/bold] to see installable skills.[/dim]"
            if _RICH else "\nTip: run 'python main.py skills' to see installable skills."
        )
        return 1

    source_path_or_url, dest_filename = result

    # Determine destination
    dest_path = PLUGINS_DIR / dest_filename
    if dest_path.exists() and not force:
        _err(console, f"Skill '{dest_filename}' already exists in skills/plugins/.")
        console.print(
            "  Use [bold]--force[/bold] to overwrite." if _RICH else "  Use --force to overwrite."
        )
        return 1

    # Download or copy
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_file = Path(tmpdir) / dest_filename

        if source_path_or_url.startswith("http"):
            _info(console, f"Downloading from {source_path_or_url}…")
            try:
                data = _fetch_url(source_path_or_url)
                tmp_file.write_bytes(data)
            except urllib.error.URLError as e:
                _err(console, f"Download failed: {e}")
                return 1
            except Exception as e:
                _err(console, f"Unexpected error during download: {e}")
                return 1
        else:
            # Local file — copy to temp first for validation
            try:
                shutil.copy2(source_path_or_url, tmp_file)
            except (OSError, shutil.Error) as e:
                _err(console, f"Could not read source file: {e}")
                return 1

        # Validate
        _info(console, "Validating skill file…")
        ok, msg = _validate_skill_file(tmp_file)
        if not ok:
            _err(console, f"Validation failed: {msg}")
            return 1

        # Install
        shutil.copy2(tmp_file, dest_path)

    _ok(console, f"Installed: [bold]{dest_path.relative_to(PROJECT_ROOT)}[/bold]"
        if _RICH else f"Installed: {dest_path.relative_to(PROJECT_ROOT)}")

    # Show skill metadata if registry entry found
    entry = registry.get(skill_arg)
    if entry and entry.get("description"):
        console.print(f"  [dim]{entry['description']}[/dim]" if _RICH else f"  {entry['description']}")

    console.print()
    console.print(
        "[dim]Restart NeuralClaw to activate the new skill, "
        "or run [bold]/reload-skills[/bold] in the CLI.[/dim]"
        if _RICH else
        "Restart NeuralClaw to activate the new skill."
    )
    console.print()
    return 0


def run_list_available(filter_str: str = "") -> int:
    """List all available installable skills."""
    console = Console()
    registry = _fetch_registry()

    console.print()
    console.print("[bold]Available Skills[/bold]" if _RICH else "Available Skills")
    console.print("[dim]Install with: python main.py install <name>[/dim]\n"
                  if _RICH else "Install with: python main.py install <name>\n")

    for key, entry in sorted(registry.items()):
        if filter_str and filter_str.lower() not in key.lower():
            continue
        desc = entry.get("description", "")
        installed = (PLUGINS_DIR / entry.get("file", f"{key}.py")).exists()
        status = "[green]installed[/green]" if (installed and _RICH) else ("installed" if installed else "")
        line = f"  [bold]{key:<22}[/bold]  {desc}"
        if status:
            line += f"  [{status}]" if _RICH else f"  ({status})"
        console.print(line if _RICH else line)

    console.print()
    return 0