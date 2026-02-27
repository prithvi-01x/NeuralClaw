"""
onboard/clawhub_installer.py — ClawHub Skill Installer

Handles installing, searching, listing, and removing ClawHub skills.
Used by both the CLI `/clawhub` commands and `neuralclaw clawhub` subcommand.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    _RICH = True
except ImportError:
    _RICH = False

try:
    from observability.logger import get_logger as _get_logger
    _log_raw = _get_logger(__name__)
    _STRUCTLOG = True
except ImportError:
    import logging as _logging
    _log_raw = _logging.getLogger(__name__)
    _STRUCTLOG = False


def _log(level: str, event: str, **kwargs) -> None:
    if _STRUCTLOG:
        getattr(_log_raw, level)(event, **kwargs)
    else:
        extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
        getattr(_log_raw, level)("%s %s", event, extra)


# ─────────────────────────────────────────────────────────────────────────────
# Lock file management
# ─────────────────────────────────────────────────────────────────────────────

def _lock_path(settings) -> Path:
    """Path to the ClawHub lock file."""
    skills_dir = Path(settings.clawhub.skills_dir)
    return skills_dir.parent / "lock.json"


def _read_lock(settings) -> dict:
    """Read the lock.json file, or return default structure."""
    lp = _lock_path(settings)
    if not lp.exists():
        return {"version": 1, "skills": {}}
    try:
        return json.loads(lp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "skills": {}}


def _write_lock(settings, lock_data: dict) -> None:
    """Write the lock.json file."""
    lp = _lock_path(settings)
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(json.dumps(lock_data, indent=2), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# ClawHub API interaction
# ─────────────────────────────────────────────────────────────────────────────

async def _http_get(url: str) -> tuple[int, bytes]:
    """Shared HTTP GET helper. Returns (status_code, body_bytes)."""
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"User-Agent": "NeuralClaw/1.0"})
            return resp.status_code, resp.content
    except ImportError:
        import urllib.request
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "NeuralClaw/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                return resp.status, resp.read()
        except Exception as e:
            _log("warning", "clawhub.http_error", error=str(e))
            return 0, b""
    except Exception as e:
        _log("warning", "clawhub.http_error", error=str(e))
        return 0, b""


async def _fetch_skill_from_registry(slug: str, registry_url: str) -> dict | None:
    """
    Fetch skill metadata from the ClawHub registry API.
    Uses the v1 REST endpoint: GET /api/v1/skills/{slug}
    Returns the parsed JSON response, or None on failure.
    """
    base = registry_url.rstrip("/")
    url = f"{base}/api/v1/skills/{slug}"
    status, body = await _http_get(url)
    if status == 200:
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            _log("warning", "clawhub_install.json_decode_error", url=url)
            return None
    elif status == 404:
        return None
    else:
        _log("warning", "clawhub_install.api_error", status=status, url=url)
        return None


async def _fetch_skill_file(slug: str, path: str, registry_url: str) -> str | None:
    """
    Fetch a raw file from an installed ClawHub skill.
    Uses: GET /api/v1/skills/{slug}/file?path={path}
    Returns raw text content, or None on failure.
    """
    base = registry_url.rstrip("/")
    url = f"{base}/api/v1/skills/{slug}/file?path={path}"
    status, body = await _http_get(url)
    if status == 200:
        return body.decode("utf-8", errors="replace")
    return None


async def _search_registry(query: str, registry_url: str) -> list[dict]:
    """
    Search the ClawHub registry for skills matching a query.
    Uses the v1 semantic search endpoint: GET /api/v1/search?q={query}&limit=20
    Results are under the 'results' key, each with: score, slug, displayName, summary, version
    Falls back to browse endpoint if search returns nothing.
    """
    import urllib.parse
    base = registry_url.rstrip("/")
    encoded = urllib.parse.quote_plus(query)

    # Primary: semantic/vector search
    url = f"{base}/api/v1/search?q={encoded}&limit=20"
    status, body = await _http_get(url)
    if status == 200:
        try:
            data = json.loads(body)
            results = data.get("results", data if isinstance(data, list) else [])
            if results:
                # Normalise field names to match what the rest of the code expects
                normalised = []
                for r in results:
                    normalised.append({
                        "name": r.get("slug", r.get("name", r.get("displayName", "?"))),
                        "description": r.get("summary", r.get("description", "")),
                        "version": r.get("version", ""),
                        "score": r.get("score"),
                    })
                return normalised
        except json.JSONDecodeError:
            pass

    # Fallback: browse by downloads and filter client-side
    url = f"{base}/api/v1/skills?limit=50&sort=downloads"
    status, body = await _http_get(url)
    if status == 200:
        try:
            data = json.loads(body)
            items = data.get("items", data if isinstance(data, list) else [])
            q = query.lower()
            return [
                {
                    "name": r.get("slug", r.get("name", "?")),
                    "description": r.get("summary", r.get("description", "")),
                    "version": r.get("version", ""),
                }
                for r in items
                if q in r.get("slug", "").lower()
                or q in r.get("displayName", "").lower()
                or q in r.get("summary", "").lower()
                or q in r.get("description", "").lower()
            ][:20]
        except json.JSONDecodeError:
            pass

    return []


# ─────────────────────────────────────────────────────────────────────────────
# Install / Remove operations
# ─────────────────────────────────────────────────────────────────────────────

async def install_from_clawhub(
    slug: str,
    settings,
    force: bool = False,
    console: Optional["Console"] = None,
) -> int:
    """
    Install a ClawHub skill by slug.

    1. Fetch skill metadata from ClawHub API
    2. Create skill folder and write files
    3. Parse with bridge_parser → validate
    4. Check requirements
    5. Update lock.json
    6. Print summary

    Returns 0 on success, 1 on failure.
    """
    c = console or (Console() if _RICH else None)

    def _print(msg: str) -> None:
        if c:
            c.print(msg)
        else:
            print(msg)

    skills_dir = Path(settings.clawhub.skills_dir)
    skills_dir.mkdir(parents=True, exist_ok=True)
    skill_dir = skills_dir / slug

    # Check if already installed
    lock = _read_lock(settings)
    if slug in lock["skills"] and not force:
        _print(f"[yellow]⚠  Skill '{slug}' is already installed. Use --force to reinstall.[/yellow]"
               if _RICH else f"⚠  Skill '{slug}' is already installed. Use --force to reinstall.")
        return 1

    # Fetch from registry
    _print(f"[cyan]Fetching '{slug}' from ClawHub...[/cyan]"
           if _RICH else f"Fetching '{slug}' from ClawHub...")

    data = await _fetch_skill_from_registry(slug, settings.clawhub.registry_url)
    if data is None:
        _print(f"[red]✗  Skill '{slug}' not found on ClawHub.[/red]"
               if _RICH else f"✗  Skill '{slug}' not found on ClawHub.")
        return 1

    # Create skill folder
    if skill_dir.exists() and force:
        shutil.rmtree(skill_dir)
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    files = data.get("files", [])
    if not files:
        # Try fetching SKILL.md directly via the v1 file endpoint
        raw_skill_md = await _fetch_skill_file(slug, "SKILL.md", settings.clawhub.registry_url)
        if raw_skill_md:
            skill_md_content = raw_skill_md
        else:
            # Last resort: build from whatever the API returned
            content = data.get("content", data.get("body", ""))
            fm = data.get("frontmatter", {})
            if fm:
                try:
                    import yaml
                    skill_md_content = f"---\n{yaml.dump(fm, default_flow_style=False)}---\n\n{content}"
                except ImportError:
                    skill_md_content = content
            else:
                skill_md_content = content
        (skill_dir / "SKILL.md").write_text(skill_md_content, encoding="utf-8")
    else:
        for f in files:
            file_path = skill_dir / f.get("path", f.get("name", "SKILL.md"))
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(f.get("content", ""), encoding="utf-8")

    # Parse and validate
    try:
        from skills.clawhub.bridge_parser import parse_clawhub_skill_md
        manifest = parse_clawhub_skill_md(skill_dir)
    except Exception as e:
        _print(f"[red]✗  Failed to parse skill: {e}[/red]"
               if _RICH else f"✗  Failed to parse skill: {e}")
        shutil.rmtree(skill_dir, ignore_errors=True)
        return 1

    # Check requirements
    missing_env = [v for v in manifest.requires.env if not os.environ.get(v)]
    from skills.clawhub.dependency_checker import check_bins
    bins_ok, missing_bins = check_bins(manifest.requires)

    tier_labels = {1: "Prompt-only", 2: "API/HTTP", 3: "Binary"}
    tier_label = tier_labels.get(manifest.execution_tier, "Unknown")

    # Update lock file
    lock["skills"][slug] = {
        "version": manifest.version,
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "skill_dir": str(skill_dir),
        "execution_tier": manifest.execution_tier,
        "requires_env": manifest.requires.env,
        "requires_bins": manifest.requires.bins,
        "env_satisfied": len(missing_env) == 0,
        "bins_satisfied": bins_ok,
    }
    _write_lock(settings, lock)

    # Print summary
    _print(f"\n[green]✓  Installed {slug} (Tier {manifest.execution_tier} — {tier_label})[/green]"
           if _RICH else f"\n✓  Installed {slug} (Tier {manifest.execution_tier} — {tier_label})")

    if missing_env:
        _print(f"   [yellow]Requires env: {', '.join(missing_env)}[/yellow]"
               if _RICH else f"   Requires env: {', '.join(missing_env)}")
        _print("   Add to your .env:")
        for v in missing_env:
            _print(f"     {v}=your_value_here")

    if missing_bins:
        _print(f"   [yellow]Missing binaries: {', '.join(missing_bins)}[/yellow]"
               if _RICH else f"   Missing binaries: {', '.join(missing_bins)}")

    _print("")
    _print(f"   [dim]⚠  ClawHub skills are community-built. Review the SKILL.md before use:[/dim]"
           if _RICH else "   ⚠  ClawHub skills are community-built. Review the SKILL.md before use:")
    _print(f"      {skill_dir / 'SKILL.md'}")
    _print("   Restart NeuralClaw or run /reload-skills to activate.")

    return 0


def list_installed(settings, console: Optional["Console"] = None) -> int:
    """List all installed ClawHub skills."""
    c = console or (Console() if _RICH else None)
    lock = _read_lock(settings)
    skills = lock.get("skills", {})

    if not skills:
        msg = "No ClawHub skills installed."
        if c and _RICH:
            c.print(f"[dim]{msg}[/dim]")
        else:
            print(msg)
        return 0

    if c and _RICH:
        table = Table(title="Installed ClawHub Skills", show_lines=True)
        table.add_column("Skill", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Tier")
        table.add_column("Env ✓", justify="center")
        table.add_column("Bins ✓", justify="center")
        for slug, info in sorted(skills.items()):
            tier = str(info.get("execution_tier", "?"))
            env_ok = "✓" if info.get("env_satisfied", False) else "✗"
            bins_ok = "✓" if info.get("bins_satisfied", True) else "✗"
            table.add_row(slug, info.get("version", "?"), tier, env_ok, bins_ok)
        c.print(table)
    else:
        print("\nInstalled ClawHub Skills:")
        for slug, info in sorted(skills.items()):
            print(f"  {slug} v{info.get('version', '?')} (Tier {info.get('execution_tier', '?')})")

    return 0


def show_info(slug: str, settings, console: Optional["Console"] = None) -> int:
    """Show info about an installed ClawHub skill."""
    c = console or (Console() if _RICH else None)
    lock = _read_lock(settings)
    info = lock.get("skills", {}).get(slug)

    if info is None:
        msg = f"Skill '{slug}' is not installed."
        if c and _RICH:
            c.print(f"[red]{msg}[/red]")
        else:
            print(msg)
        return 1

    # Try to parse the skill for more detail
    skill_dir = Path(info.get("skill_dir", ""))
    try:
        from skills.clawhub.bridge_parser import parse_clawhub_skill_md
        manifest = parse_clawhub_skill_md(skill_dir)
        desc = manifest.description
        body_preview = manifest.body[:200] + "..." if len(manifest.body) > 200 else manifest.body
    except Exception:
        desc = "Unable to parse"
        body_preview = ""

    if c and _RICH:
        lines = [
            f"[bold]{slug}[/bold] v{info.get('version', '?')}",
            f"Tier: {info.get('execution_tier', '?')}",
            f"Description: {desc}",
            f"Requires env: {', '.join(info.get('requires_env', [])) or 'none'}",
            f"Requires bins: {', '.join(info.get('requires_bins', [])) or 'none'}",
            f"Installed: {info.get('installed_at', '?')}",
            f"Path: {info.get('skill_dir', '?')}",
        ]
        c.print(Panel("\n".join(lines), title=f"ClawHub: {slug}"))
    else:
        print(f"\n{slug} v{info.get('version', '?')}")
        print(f"  Tier: {info.get('execution_tier', '?')}")
        print(f"  Description: {desc}")
        print(f"  Path: {info.get('skill_dir', '?')}")

    return 0


def remove_skill(slug: str, settings, console: Optional["Console"] = None) -> int:
    """Remove an installed ClawHub skill."""
    c = console or (Console() if _RICH else None)

    def _print(msg: str) -> None:
        if c and _RICH:
            c.print(msg)
        else:
            print(msg)

    lock = _read_lock(settings)
    info = lock.get("skills", {}).get(slug)

    if info is None:
        _print(f"[red]Skill '{slug}' is not installed.[/red]"
               if _RICH else f"Skill '{slug}' is not installed.")
        return 1

    # Remove skill directory
    skill_dir = Path(info.get("skill_dir", ""))
    if skill_dir.exists():
        shutil.rmtree(skill_dir, ignore_errors=True)

    # Remove from lock
    del lock["skills"][slug]
    _write_lock(settings, lock)

    _print(f"[green]✓  Removed '{slug}'[/green]"
           if _RICH else f"✓  Removed '{slug}'")
    _print("   Restart NeuralClaw to fully unload the skill.")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI dispatcher
# ─────────────────────────────────────────────────────────────────────────────

async def clawhub_command(
    action: str,
    args: list[str],
    settings=None,
    console: Optional["Console"] = None,
) -> int:
    """
    Dispatch a clawhub subcommand.

    Actions: search, install, list, info, remove
    """
    if settings is None:
        from config.settings import get_settings
        settings = get_settings()

    if action == "install" and args:
        return await install_from_clawhub(
            slug=args[0],
            settings=settings,
            force="--force" in args,
            console=console,
        )
    elif action == "list":
        return list_installed(settings, console=console)
    elif action == "info" and args:
        return show_info(args[0], settings, console=console)
    elif action == "remove" and args:
        return remove_skill(args[0], settings, console=console)
    elif action == "search" and args:
        results = await _search_registry(
            query=" ".join(args),
            registry_url=settings.clawhub.registry_url,
        )
        if not results:
            msg = "No results found."
            if console and _RICH:
                console.print(f"[dim]{msg}[/dim]")
            else:
                print(msg)
            return 0
        if console and _RICH:
            table = Table(title="ClawHub Search Results")
            table.add_column("Slug", style="cyan")
            table.add_column("Description")
            for r in results[:20]:
                table.add_row(
                    r.get("name", r.get("slug", "?")),
                    r.get("description", "")[:80],
                )
            console.print(table)
        else:
            for r in results[:20]:
                print(f"  {r.get('name', '?')} — {r.get('description', '')[:80]}")
        return 0
    else:
        msg = (
            "Usage:\n"
            "  /clawhub search <query>   — Search ClawHub registry\n"
            "  /clawhub install <slug>   — Install a ClawHub skill\n"
            "  /clawhub list             — List installed ClawHub skills\n"
            "  /clawhub info <slug>      — Show skill details\n"
            "  /clawhub remove <slug>    — Remove an installed skill"
        )
        if console and _RICH:
            console.print(msg)
        else:
            print(msg)
        return 0