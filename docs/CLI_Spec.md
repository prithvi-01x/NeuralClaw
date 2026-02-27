# `neuralclaw` CLI — Command Specification

> **What this document is:** A complete spec for a standalone `neuralclaw` CLI tool — the NeuralClaw equivalent of `clawhub`. This covers every command, its flags, behavior, output, error handling, and implementation notes. Build this as `neuralclaw_cli.py` + an entry point so users can run `neuralclaw <command>` from their terminal anywhere inside the project.

---

## Overview

The `neuralclaw` CLI is a developer-facing tool for managing skills, configuration, and the running agent. It mirrors the `clawhub` experience from OpenClaw but is purpose-built for NeuralClaw's architecture (skills live in `skills/plugins/`, config is `config/config.yaml`, metadata tracked in `.metadata.json`).

### How to invoke it

```bash
# After symlinking or adding to PATH:
neuralclaw <command> [args] [flags]

# Or directly without install:
python neuralclaw_cli.py <command> [args] [flags]

# Or via python main.py (existing hook, already partially works):
python main.py install <slug>
python main.py skills
```

The goal is a proper standalone `neuralclaw` binary that people can put in their `$PATH`.

---

## Installation of the CLI itself

### `scripts/install_cli.sh` (to include)
```bash
#!/usr/bin/env bash
# Symlinks neuralclaw_cli.py to ~/bin/neuralclaw so it's available everywhere
chmod +x neuralclaw_cli.py
mkdir -p ~/bin
ln -sf "$(pwd)/neuralclaw_cli.py" ~/bin/neuralclaw
echo "✓ neuralclaw CLI installed. Make sure ~/bin is in your PATH."
```

Alternatively, expose it as a `pyproject.toml` entry point:
```toml
[project.scripts]
neuralclaw = "neuralclaw_cli:main"
```

---

## Project Root Detection

The CLI must auto-detect the NeuralClaw project root by walking up from `CWD` looking for a directory that contains both `main.py` and `skills/`. Falls back to `NEURALCLAW_ROOT` env var. If not found, most commands should print a clear error and exit 1.

---

## Metadata Files

Two hidden JSON files live in `skills/plugins/` to track state:

| File | Purpose |
|---|---|
| `skills/plugins/.metadata.json` | Per-slug install timestamps, versions, source URLs, categories |
| `skills/plugins/.disabled.json` | List of disabled slug names (skill file stays on disk, just flagged off) |

---

## Full Command Reference

---

### `neuralclaw install <slug>`

Installs a skill by slug name, GitHub URL, or local file path.

**Behavior:**
1. Fetch registry index from `https://raw.githubusercontent.com/neuralclaw/skills/main/index.json` — fall back to built-in index if unreachable
2. Resolve the slug to a source (registry entry → URL, local file, GitHub shorthand)
3. Download/copy to a temp file
4. Validate: Python syntax check + presence of `SkillBase` or `SkillManifest` (for `.py`), or `name:` frontmatter (for `.md`)
5. Copy to `skills/plugins/<filename>`
6. Write entry to `.metadata.json`
7. Print success + remind user to restart or `/reload-skills`

**Input formats:**
```bash
neuralclaw install weather                        # by slug
neuralclaw install github.com/alice/my-skill      # GitHub shorthand
neuralclaw install https://raw.github.../skill.py # raw URL
neuralclaw install ./my_local_skill.py            # local file
```

**Flags:**
| Flag | Description |
|---|---|
| `--force` | Overwrite if already installed |
| `--dev` | Symlink instead of copy (for skill authors developing locally) |
| `--dry-run` | Show what would happen without writing any files |

**Output (success):**
```
╭─────────────────────────────────╮
│  Installing skill: weather      │
╰─────────────────────────────────╯
   Fetching registry…
   Validating skill file…
✓  Installed weather → skills/plugins/personal_weather_fetch.py
   Get current weather for a location

   Restart NeuralClaw or run /reload-skills to activate.
```

**Output (already installed):**
```
⚠  Skill 'weather' is already installed.
   Use --force to reinstall.
```

**Output (not found + fuzzy suggestion):**
```
✗  Skill 'wether' not found in registry.
   Did you mean: weather, web-fetch?
   Run 'neuralclaw search weather' to explore available skills.
```

**Error cases to handle:**
- Slug not in registry → fuzzy match suggestion + point to `search`
- Download fails (network error) → show URL + error message
- Validation fails (bad Python/missing manifest) → show exact reason
- Already installed without `--force` → warn, don't overwrite silently
- Built-in skill (source = "builtin") → print "already available, no install needed"

---

### `neuralclaw remove <slug>`

Removes an installed plugin skill.

**Behavior:**
1. Look up slug in registry to get filename
2. Check the file exists in `skills/plugins/`
3. Confirm with user (interactive prompt)
4. Delete the file
5. Remove from `.metadata.json`

**Flags:**
| Flag | Description |
|---|---|
| `--yes` / `-y` | Skip confirmation prompt |

**Notes:**
- Built-in skills (in `skills/builtin/`) cannot be removed. Suggest `disable` instead.
- If the slug is not installed, warn and exit 0 (not an error)

---

### `neuralclaw update [slug] [--all]`

Updates one or all installed skills to their latest remote versions.

**Behavior:**
1. If `--all`: iterate all entries in `.metadata.json`
2. For each: fetch the remote URL from registry, validate, overwrite in place
3. Update `version` and add `updated_at` timestamp in `.metadata.json`
4. Report how many were updated, skipped (no URL), or failed

**Flags:**
| Flag | Description |
|---|---|
| `--all` | Update every installed skill that has a remote URL |
| `--dry-run` | Show what would be updated without doing it |

**Notes:**
- Skills with `source: builtin` are skipped (they update with the repo)
- Skills with no `url` in registry are skipped with a note
- Failed validations after download → keep old version, warn

---

### `neuralclaw list`

Lists all installed skills in a formatted table.

**Flags:**
| Flag | Description |
|---|---|
| `--all` | Show all available skills from registry (not just installed ones) |
| `--category <name>` | Filter by category (cyber, developer, personal, data, automation, system, meta, builtin) |
| `--json` | Output raw JSON (useful for scripting) |

**Output (installed skills):**
```
┌──────────────────┬────────────────────────────┬────────────┬───────────┬──────────┐
│ Slug             │ Name                       │ Category   │ Status    │ Version  │
├──────────────────┼────────────────────────────┼────────────┼───────────┼──────────┤
│ weather          │ personal_weather_fetch     │ personal   │ ✓ active  │ 1.0.0    │
│ git-log          │ dev_git_log                │ developer  │ ✓ active  │ 1.0.0    │
│ port-scan        │ cyber_port_scan            │ cyber      │ ⊘ disabled│ 1.0.0    │
└──────────────────┴────────────────────────────┴────────────┴───────────┴──────────┘
4 skills installed  (3 active, 1 disabled)
```

**Output (--all):** Same table but includes all registry entries, with `not installed` status for ones not yet installed.

---

### `neuralclaw search <query>`

Searches the registry index by name, description, and category.

**Flags:**
| Flag | Description |
|---|---|
| `--category <name>` | Filter results to a specific category |
| `--installed` | Only show skills you've already installed |

**Output:**
```
Search results for "git"

  git-log       developer   View git commit history
  git-diff      developer   Show git diffs for files/commits
  git-blame     developer   Show git blame for a file
  pr-summary    developer   Summarize a pull request (uses git)

4 results  ·  Install with: neuralclaw install <slug>
```

---

### `neuralclaw info <slug>`

Shows full details for a skill from the registry.

**Output:**
```
╭──────────────────────────────────────────────╮
│  weather                                     │
├──────────────────────────────────────────────┤
│  Name:        personal_weather_fetch         │
│  Category:    personal                       │
│  Version:     1.0.0                          │
│  Risk Level:  LOW                            │
│  Status:      ✓ Installed                    │
│  Installed:   2025-07-01 14:23               │
│  File:        skills/plugins/personal_...    │
│                                              │
│  Description:                                │
│  Get current weather for a location.         │
│  Returns temperature, conditions, wind.      │
│                                              │
│  Parameters:                                 │
│    location  (string, required)              │
│    units     (string) "metric" | "imperial"  │
╰──────────────────────────────────────────────╯
```

---

### `neuralclaw create <slug>`

Scaffolds a new skill from a template — the fastest way to build a custom skill.

**Behavior:**
1. Ask user: Python skill or Markdown skill? (interactive prompt)
2. Ask for: description, category, risk level
3. Generate file from template into `skills/plugins/<slug>.py` (or `.md`)
4. Print the file contents and path

**Flags:**
| Flag | Description |
|---|---|
| `--python` | Force Python template (skip prompt) |
| `--markdown` | Force Markdown template (skip prompt) |
| `--output <path>` | Write to a custom path instead of `skills/plugins/` |

**Python template output** (written to `skills/plugins/my_skill.py`):
```python
"""
my_skill — NeuralClaw Skill
<description>
"""
from skills.base import SkillBase
from skills.types import SkillManifest, SkillResult, RiskLevel

class MySkill(SkillBase):
    manifest = SkillManifest(
        name="my_skill",
        version="1.0.0",
        description="<description>",
        category="<category>",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset(),
        parameters={
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "The input to process"}
            },
            "required": ["input"],
        },
    )

    async def execute(self, input: str, **kwargs) -> SkillResult:
        skill_call_id = kwargs.get("_skill_call_id", "")
        try:
            result = f"Processed: {input}"
            return SkillResult.ok(self.manifest.name, skill_call_id, result)
        except Exception as e:
            return SkillResult.fail(self.manifest.name, skill_call_id, str(e))
```

**Markdown template** (for prompt-based skills):
```markdown
---
name: my_skill
version: 1.0.0
description: <description>
category: <category>
risk_level: LOW
parameters:
  type: object
  properties:
    input:
      type: string
  required: [input]
---

You are a helpful assistant. Process the following input:

{{input}}
```

---

### `neuralclaw enable <slug>`

Re-enables a previously disabled skill.

**Behavior:** Remove the slug from `.disabled.json`. File stays in `skills/plugins/` untouched.

---

### `neuralclaw disable <slug>`

Disables a skill without deleting it.

**Behavior:** Add the slug to `.disabled.json`. The file stays on disk. When NeuralClaw loads skills, any name in the disabled list has `manifest.enabled = False`, so it won't appear in tool schemas sent to the LLM.

**This is the right way to disable built-in skills** (which cannot be removed).

---

### `neuralclaw sync`

Scans installed skills and syncs their metadata with the registry.

**Behavior:**
1. Read all `.py` and `.md` files in `skills/plugins/`
2. For each file, try to match it to a registry entry by filename
3. Update `.metadata.json` with any missing entries (fills in category, version, source)
4. Report any unrecognized files (skills not in the registry)
5. Optionally: push unrecognized skills to registry via `--publish` flag

**Flags:**
| Flag | Description |
|---|---|
| `--all` | Sync all skills including ones already tracked |
| `--publish` | Open a GitHub issue to submit unrecognized skills to the registry |

**Output:**
```
Syncing skills…
✓  weather       → already tracked (v1.0.0)
✓  git-log       → already tracked (v1.0.0)
?  my_custom.py  → not in registry (unrecognized)

Sync complete. 2 tracked, 1 unrecognized.
Run 'neuralclaw publish my_custom.py' to submit to the registry.
```

---

### `neuralclaw publish <file>`

Submits a skill file to the community registry via a GitHub issue.

**Behavior:**
1. Validate the skill file (syntax + manifest check)
2. Read the manifest metadata (name, description, category, risk_level)
3. Open a pre-filled GitHub issue URL in the browser with the skill metadata template
4. OR (with `--token`): create the issue directly via GitHub API

**Flags:**
| Flag | Description |
|---|---|
| `--token <github_token>` | GitHub personal access token for API submission |
| `--dry-run` | Show the issue body without opening browser |

---

### `neuralclaw doctor`

Validates the entire NeuralClaw installation and diagnoses common issues.

**Checks to run:**

| Check | What it validates |
|---|---|
| Python version | `>= 3.11` |
| Project root found | `main.py` + `skills/` both present |
| `requirements.txt` satisfied | All packages importable |
| `.env` file present | Exists and is not empty |
| LLM provider config | `config.yaml` `llm.default_provider` is set |
| API key present | Env var for configured provider is set (e.g. `OPENAI_API_KEY`) |
| `skills/plugins/` exists | Directory present and writable |
| All plugins valid | Each `.py` file in `skills/plugins/` passes syntax check |
| ChromaDB data dir | `./data/chroma/` exists |
| SQLite data dir | `./data/sqlite/` exists |
| Ollama reachable | If provider is `ollama`, check `http://localhost:11434` responds |
| Telegram token | If interface is `telegram`, `TELEGRAM_BOT_TOKEN` is set |

**Output:**
```
NeuralClaw Doctor  v1.0.0
─────────────────────────────────────────
✓  Python 3.11.9
✓  Project root: /home/user/neuralclaw
✓  .env file present
✓  LLM provider: ollama
✓  Ollama reachable at http://localhost:11434
✓  Model available: qwen3:8b
✓  skills/plugins/ writable (52 skills)
✓  All plugin files valid Python
✓  data/chroma/ exists
✓  data/sqlite/ exists
⚠  TELEGRAM_BOT_TOKEN not set (only needed for telegram interface)
✗  OPENAI_API_KEY not set (needed if using openai provider)

2 warnings · 1 error
Run 'neuralclaw config llm.default_provider ollama' to stay on Ollama.
```

Exit code: `0` if no errors, `1` if any errors found.

---

### `neuralclaw status`

Shows the current runtime status of the agent (if it's running).

**What to show:**
- Is the agent process running? (check PID file at `./data/agent.pid`)
- Active interface (cli / telegram / voice)
- Current LLM provider and model
- Number of active sessions
- Memory stats: SQLite turn count, ChromaDB vector count
- Installed skills count (active / disabled)
- Log file location and size
- Uptime

**Output:**
```
NeuralClaw Status
─────────────────────────────────────────
  Process:     ● Running  (PID 12345, uptime 4h 32m)
  Interface:   cli
  Provider:    ollama / qwen3:8b
  Sessions:    1 active

  Memory
    Short-term: 12 turns in SQLite
    Long-term:  847 vectors in ChromaDB

  Skills
    52 installed  (51 active, 1 disabled)

  Logs
    ./data/logs/neuralclaw.log  (2.3 MB)
```

If not running:
```
  Process:     ○ Not running
  Run: python main.py
```

---

### `neuralclaw config [key] [value]`

Read or write values in `config/config.yaml` from the command line.

**Behavior:**
- No args → print full config (pretty-printed YAML)
- One arg → print the value at that dotted key
- Two args → set the value and save the file

**Examples:**
```bash
neuralclaw config                                    # print everything
neuralclaw config llm.default_provider               # print: "ollama"
neuralclaw config llm.default_provider openai        # set provider to openai
neuralclaw config llm.default_model gpt-4o           # set model
neuralclaw config agent.default_trust_level medium   # set trust level
neuralclaw config safety.require_confirmation_for    # show list
neuralclaw config llm.temperature 0.2                # set temperature
```

**Flags:**
| Flag | Description |
|---|---|
| `--raw` | Print value without formatting (useful for scripting) |

**Notes:**
- Use dotted-path notation matching the YAML hierarchy: `llm.default_provider`, `agent.max_iterations_per_turn`, `memory.max_short_term_turns`, etc.
- After writing, validate the config against Pydantic schema and warn if invalid
- Never write `api_key` values into `config.yaml` — those belong in `.env`

---

### `neuralclaw logs [--tail N] [--follow] [--level LEVEL]`

View or stream the agent's structured log output.

**Flags:**
| Flag | Description |
|---|---|
| `--tail N` | Show last N lines (default: 50) |
| `--follow` / `-f` | Stream new log lines in real-time (like `tail -f`) |
| `--level <level>` | Filter by log level: DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `--json` | Print raw JSON log lines without formatting |
| `--session <id>` | Filter to a specific session ID |

**Output (formatted):**
```
[14:23:01] INFO     orchestrator.turn_start          session=abc123
[14:23:01] INFO     skill_bus.dispatch               skill=web_search
[14:23:02] INFO     skill_bus.result                 skill=web_search duration_ms=834
[14:23:02] INFO     orchestrator.turn_done           status=success ms=1203
[14:23:15] WARNING  orchestrator.max_iter_reached    iterations=10
```

---

### `neuralclaw version`

Prints version info for the CLI, NeuralClaw core, and Python environment.

**Output:**
```
neuralclaw CLI       v1.0.0
NeuralClaw core      v1.0.0  (from config/config.yaml agent.version)
Python               3.11.9
Platform             linux / x86_64
Skills               52 installed
Config               /home/user/neuralclaw/config/config.yaml
```

---

## Exit Codes

| Code | Meaning |
|---|---|
| `0` | Success |
| `1` | General error (missing args, not found, validation failed) |
| `2` | Project root not found |
| `3` | Network error (registry unreachable) |
| `4` | Doctor found errors |

---

## Global Flags (apply to all commands)

| Flag | Description |
|---|---|
| `--no-color` | Disable Rich colored output |
| `--json` | Output machine-readable JSON where supported |
| `--quiet` / `-q` | Suppress all output except errors |
| `--verbose` / `-v` | Show extra debug info |
| `--root <path>` | Override project root detection |

---

## Implementation Notes

### File structure for the CLI
```
neuralclaw_cli.py          ← the single-file CLI (entry point)
scripts/
  install_cli.sh           ← symlinks neuralclaw_cli.py to ~/bin/neuralclaw
```

### Dependencies
The CLI should run with only what's already in `requirements.txt`. Specifically:
- `rich` — already a dependency, use for all output formatting
- `pyyaml` — already a dependency, use for config read/write
- `pydantic` — already a dependency, optional use for config validation
- No new packages needed

### What `main.py` already handles (don't duplicate)
`main.py` already handles `python main.py install <slug>` and `python main.py skills`. The new `neuralclaw_cli.py` should be a **separate standalone file** that does NOT import from `main.py` — it's a standalone tool. It can import from `onboard/skill_installer.py` for the install logic to avoid duplication.

### Registry metadata format (`.metadata.json`)
```json
{
  "weather": {
    "file": "personal_weather_fetch.py",
    "installed_at": "2025-07-01T14:23:01",
    "updated_at": "2025-07-02T09:15:00",
    "version": "1.0.0",
    "source": "plugin",
    "category": "personal"
  }
}
```

### Disabled skills format (`.disabled.json`)
```json
["port-scan", "sql-query"]
```

### How `disable`/`enable` hooks into the loader
When `skills/loader.py` loads a skill, it should check if `skills/plugins/.disabled.json` contains that skill's slug. If yes, set `manifest.enabled = False`. This means the skill is loaded into the registry but excluded from tool schemas sent to the LLM — it won't be called but also won't break on reload.

---

## Categories Reference

These are the valid category values used in `--category` filters and skill manifests:

| Category | Skills it contains |
|---|---|
| `builtin` | web-search, web-fetch, filesystem, terminal |
| `cyber` | port-scan, dns-enum, whois, ssl-check, cve-lookup, subdomain-enum, banner-grab, http-probe, tech-fingerprint, vuln-report |
| `developer` | git-log, git-diff, git-blame, lint, test-runner, pr-summary, code-search, dep-audit, readme-gen, file-summarize |
| `personal` | weather, calendar, email-summary, news, note, tasks, reminder |
| `data` | csv-parse, json-transform, sql-query, chart-gen, doc-summarize, diff-docs |
| `automation` | script-runner, cron-create, file-watcher, webhook, slack-notify, report-render |
| `system` | disk-usage, proc-list, service-status, log-tail, backup, pkg-update |
| `meta` | daily-brief, research, recon-pipeline, sys-maintenance, repo-audit |

---

## Quick Reference Card

```
neuralclaw install <slug>          Install a skill
neuralclaw install <slug> --force  Reinstall/overwrite
neuralclaw remove <slug>           Remove a skill
neuralclaw update --all            Update all installed skills
neuralclaw list                    List installed skills
neuralclaw list --all              List all available skills
neuralclaw list --category cyber   Filter by category
neuralclaw search <query>          Search the skill registry
neuralclaw info <slug>             Show skill details
neuralclaw create <slug>           Scaffold a new skill
neuralclaw enable <slug>           Re-enable a disabled skill
neuralclaw disable <slug>          Disable without removing
neuralclaw sync                    Sync metadata with registry
neuralclaw publish <file>          Submit skill to community registry
neuralclaw doctor                  Diagnose installation issues
neuralclaw status                  Show agent runtime status
neuralclaw config                  View full config
neuralclaw config <key> <value>    Set a config value
neuralclaw logs --tail 100         View last 100 log lines
neuralclaw logs --follow           Stream logs in real-time
neuralclaw version                 Show version info
```
