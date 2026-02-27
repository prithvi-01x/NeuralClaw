# ClawHub → NeuralClaw Bridge Adapter
## Implementation Spec

> **Goal:** Let NeuralClaw install and run any of the 5,700+ skills from clawhub.ai without
> manually rewriting them. A bridge adapter reads the OpenClaw `SKILL.md` format natively
> and executes it inside NeuralClaw's existing `SkillBus` + `SafetyKernel` pipeline.

---

## 1. Understanding the Two Formats

Before building anything, we need to understand exactly what we're bridging.

### ClawHub Skill Format (`SKILL.md`)

Every ClawHub skill is a **folder** containing a `SKILL.md` file plus optional support files.
The `SKILL.md` is markdown with YAML frontmatter:

```yaml
---
name: todoist-cli
description: Manage Todoist tasks, projects, and labels from the command line.
version: 1.2.0
metadata:
  openclaw:
    requires:
      env:
        - TODOIST_API_KEY       # env vars the skill needs
      bins:
        - curl                  # CLI binaries that must exist
      anyBins:
        - node                  # at least one of these must exist
    install:
      - kind: brew              # auto-install instructions
        formula: jq
        bins: [jq]
      - kind: node
        package: typescript
        bins: [tsc]
    primaryEnv: TODOIST_API_KEY # the main credential env var
    emoji: "✅"
    homepage: https://github.com/example/todoist-cli
---

## What this skill does

Explain the skill's purpose and usage here. This is injected directly into the
LLM's system prompt as plain instructions. The LLM reads these instructions and
decides how/when to invoke them.

## Commands

Use `todoist add "Buy milk" --due tomorrow` to create a task.
Use `todoist list --project Work` to list tasks in a project.
```

**Key insight:** The body of the `SKILL.md` (everything after the frontmatter `---`) is
**raw text injected into the agent's system prompt**. It's not code. OpenClaw's LLM reads
it as instructions and then issues shell commands (`bash` tool) to call the CLI binaries.

### NeuralClaw Skill Format

NeuralClaw skills are Python `SkillBase` subclasses:

```python
class MySkill(SkillBase):
    manifest = SkillManifest(name="my_skill", ...)
    async def execute(self, **kwargs) -> SkillResult: ...
```

Or simple `.md` files that `md_loader.py` wraps in an auto-generated `SkillBase` class.

---

## 2. The Bridge Strategy

There are **three tiers** of compatibility, handled by three different bridge approaches:

### Tier 1 — Prompt-only skills (no `bins`, no `install`) → `~40% of ClawHub`
Pure instructions. No binaries needed. The LLM just reads the instructions and knows what
to do using NeuralClaw's existing built-in tools (web_fetch, filesystem, terminal).

**Bridge approach:** Pass the SKILL.md body directly into NeuralClaw's system prompt
via a new `ClawhubMdSkill` class. Zero conversion needed.

### Tier 2 — Skills with `requires.env` only (API key skills) → `~35% of ClawHub`
Skills that call REST APIs via `curl` or HTTP. They declare env vars but no special binaries.

**Bridge approach:** Wrap in a `ClawhubHttpSkill` class that uses NeuralClaw's existing
`httpx` client instead of shelling out to `curl`. Env vars are injected from `.env`.

### Tier 3 — Skills with `requires.bins` or `install` directives → `~25% of ClawHub`
Skills that need `jq`, `gh` (GitHub CLI), `brew` packages, Node.js tools, etc.

**Bridge approach:** Route through NeuralClaw's existing `terminal_exec` built-in skill
(sandboxed shell). Auto-check if the binary exists. If missing, run the `install`
directives first (via `brew install`, `npm install -g`, etc.) inside the sandbox.

---

## 3. New Files To Create

```
skills/
  clawhub/                        ← new directory for all ClawHub bridge code
    __init__.py
    bridge_loader.py              ← discovers & loads ClawHub skill folders
    bridge_parser.py              ← parses SKILL.md frontmatter + body
    bridge_executor.py            ← the three execution tiers
    dependency_checker.py         ← checks bins exist, runs install directives
    env_injector.py               ← validates & injects required env vars
    clawhub_skill.py              ← the ClawhubSkill SkillBase subclass

onboard/
  clawhub_installer.py            ← `neuralclaw install <clawhub-slug>` command

config/
  config.yaml                     ← add `clawhub:` section (see below)

data/
  clawhub/
    skills/                       ← installed ClawHub skill folders live here
    lock.json                     ← tracks installed versions (mirrors .clawhub/lock.json)
```

---

## 4. Config Changes (`config/config.yaml`)

Add a new top-level `clawhub:` section:

```yaml
clawhub:
  enabled: true

  # Where ClawHub skill folders are installed on disk
  skills_dir: "./data/clawhub/skills"

  # ClawHub registry API base URL
  registry_url: "https://clawhub.ai"

  # Execution tier settings
  execution:
    # Tier 3: allow skills to shell out to CLI binaries via terminal_exec
    allow_binary_skills: true

    # Tier 3: auto-run install directives (brew, npm, etc.) when a binary is missing
    auto_install_deps: false        # default false — require explicit user opt-in

    # Sandbox binary execution in Docker (strongly recommended for Tier 3)
    sandbox_binary_skills: true

  # Env var handling
  env:
    # If a skill requires env vars that aren't set, block execution and warn
    block_on_missing_env: true

    # Show which env vars a skill needs when it's installed
    show_env_requirements_on_install: true

  # Risk mapping — ClawHub has no risk levels, so we assign them
  risk_defaults:
    prompt_only: "LOW"         # Tier 1
    api_http: "LOW"            # Tier 2
    binary_execution: "HIGH"   # Tier 3 (requires user confirmation)
    install_directive: "HIGH"  # auto-installing packages
```

---

## 5. Core Implementation

### 5.1 `skills/clawhub/bridge_parser.py` — Parse `SKILL.md`

This file is the foundation. It reads a ClawHub `SKILL.md` and returns a structured
`ClawhubSkillManifest` dataclass.

**What it must parse from the frontmatter:**

```python
@dataclass
class ClawhubRequires:
    env: list[str]       # env vars that must ALL be set
    bins: list[str]      # CLI binaries that must ALL exist
    any_bins: list[str]  # CLI binaries where at least ONE must exist
    configs: list[str]   # config file paths the skill reads

@dataclass
class ClawhubInstallDirective:
    kind: str            # "brew", "node", "go", "uv"
    formula: str         # for brew
    package: str         # for node/go/uv
    bins: list[str]      # binaries this directive provides

@dataclass
class ClawhubSkillManifest:
    # From frontmatter
    name: str
    description: str
    version: str
    primary_env: str | None
    emoji: str
    homepage: str | None
    requires: ClawhubRequires
    install_directives: list[ClawhubInstallDirective]

    # From parsing the body
    body: str            # raw markdown body (the prompt instructions)
    skill_dir: Path      # path to the skill folder on disk
    extra_files: dict[str, str]  # other files in the folder (e.g. references/*.md)

    # Derived
    execution_tier: int  # 1, 2, or 3 — determined by bridge_parser
```

**Tier detection logic** (run during parse):
```
if requires.bins is empty AND install_directives is empty:
    tier = 1  (prompt-only)
elif requires.bins is empty AND only uses curl/wget/http:
    tier = 2  (HTTP/API skill)
else:
    tier = 3  (binary skill)
```

**Handling the `metadata` key aliases:** ClawHub accepts `metadata.openclaw`,
`metadata.clawdbot`, and `metadata.clawdis` as equivalent. The parser must check all three.

**Handling `references/` subfolder:** Many ClawHub skills have a `references/` directory
with extra `.md` files (API docs, examples). These should be read and appended to `body`
so the LLM has the full context.

---

### 5.2 `skills/clawhub/clawhub_skill.py` — The Bridge SkillBase Class

This is the single `SkillBase` subclass that wraps any ClawHub skill.
One class, different behavior based on `execution_tier`.

```python
class ClawhubSkill(SkillBase):
    """
    A NeuralClaw SkillBase wrapper for a ClawHub SKILL.md skill.
    
    Dynamically created per-skill by bridge_loader.py.
    The manifest is built from the parsed ClawhubSkillManifest.
    The execute() method dispatches to the appropriate tier executor.
    """

    # These are set per-instance by bridge_loader (not ClassVar)
    _clawhub_manifest: ClawhubSkillManifest
    _executor: "BridgeExecutor"

    # manifest (ClassVar) is constructed dynamically — see bridge_loader.py

    async def execute(self, **kwargs) -> SkillResult:
        return await self._executor.run(self._clawhub_manifest, kwargs)
```

**How the `SkillManifest` is constructed dynamically:**

```python
def build_neuralclaw_manifest(cm: ClawhubSkillManifest, settings) -> SkillManifest:
    risk_map = {
        1: RiskLevel.LOW,
        2: RiskLevel.LOW,
        3: RiskLevel.HIGH,
    }
    return SkillManifest(
        name=_sanitize_name(cm.name),   # "todoist-cli" → "todoist_cli"
        version=cm.version or "1.0.0",
        description=cm.description,
        category="clawhub",             # all bridge skills get this category
        risk_level=risk_map[cm.execution_tier],
        capabilities=_derive_capabilities(cm),
        parameters={
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "What you want the skill to do"
                }
            },
            "required": ["request"],
        },
        requires_confirmation=(cm.execution_tier == 3),
        enabled=True,
    )
```

Note: ClawHub skills expose a single `request` parameter — a free-text string — because
the LLM reads the skill instructions and formulates the right command itself. This is
intentionally different from NeuralClaw's structured parameter schemas.

---

### 5.3 `skills/clawhub/bridge_executor.py` — The Three Tier Executors

#### Tier 1: `PromptOnlyExecutor`

No tool calls. Just inject the skill's instructions + the user's request into a new
LLM call, get the response, return it as a `SkillResult`.

```
execute(request):
    1. Build prompt:
       system = cm.body          ← the SKILL.md instructions
       user   = request          ← what the user asked
    2. Call LLM via llm_client.generate(messages, config, tools=None)
    3. Return SkillResult.ok(output=llm_response.text)
```

This is already basically what NeuralClaw's `md_skill.py` does. Reuse it.

#### Tier 2: `HttpApiExecutor`

The skill wants to call a REST API. Instead of shelling out to `curl`, we call it
via `httpx` directly inside Python.

```
execute(request):
    1. Check required env vars are set (env_injector.py)
    2. Build prompt:
       system = cm.body + "\n\nIMPORTANT: Do not use curl or shell commands.
                Instead, output a JSON object with:
                {method, url, headers, body} for the HTTP call to make."
       user   = request
    3. Call LLM → parse JSON tool call from response
    4. Execute the HTTP call via httpx (with env vars injected as headers/auth)
    5. Feed HTTP response back to LLM for final answer
    6. Return SkillResult.ok(output=final_answer)
```

**Why this works:** Most API skills just tell the LLM "use curl -X POST https://api.example.com
with Authorization: Bearer $API_KEY". When we tell the LLM to output a JSON HTTP spec
instead of a curl command, it naturally converts it. The LLM already knows the API
from the skill instructions.

#### Tier 3: `BinaryExecutor`

The skill needs real CLI binaries. Route through NeuralClaw's `terminal_exec` built-in.

```
execute(request):
    1. Check required env vars (env_injector.py)
    2. Check required bins exist (dependency_checker.py)
       → If missing AND auto_install_deps=true: run install directives first
       → If missing AND auto_install_deps=false: return SkillResult.fail() with
         helpful message listing missing bins and how to install them
    3. Build prompt:
       system = cm.body
       user   = request + "\n\nIMPORTANT: Use shell commands to accomplish this.
                Available tools: terminal_exec. Output commands as terminal_exec calls."
    4. Call LLM with tools=[terminal_exec_schema] 
    5. Route resulting tool calls through the existing SkillBus (terminal_exec)
       → This goes through SafetyKernel, gets risk-scored, requires confirmation if HIGH
    6. Feed terminal output back to LLM
    7. Return final SkillResult
```

**Security note:** Tier 3 execution goes through NeuralClaw's full safety pipeline — the
same `SafetyKernel` + whitelist checks that native terminal skills use. No special-casing.

---

### 5.4 `skills/clawhub/dependency_checker.py` — Binary & Env Checks

**Binary check:**
```python
def check_bins(requires: ClawhubRequires) -> tuple[bool, list[str]]:
    """
    Returns (all_ok, list_of_missing_bins).
    Checks PATH for each required binary using shutil.which().
    For anyBins, passes if at least one is found.
    """
```

**Install directives:**
```python
async def run_install_directive(directive: ClawhubInstallDirective, bus: SkillBus) -> bool:
    """
    Runs a ClawHub install directive via terminal_exec.
    Supported kinds:
      brew → "brew install <formula>"
      node → "npm install -g <package>"
      go   → "go install <package>"
      uv   → "uv tool install <package>"
    Returns True if install succeeded.
    """
```

**Env var check:**
```python
def check_env(requires: ClawhubRequires) -> tuple[bool, list[str]]:
    """
    Returns (all_ok, list_of_missing_vars).
    Checks os.environ for each required env var.
    """
```

---

### 5.5 `skills/clawhub/bridge_loader.py` — Discovery & Registration

This is called at NeuralClaw startup alongside the existing `SkillLoader`.

```python
class ClawhubBridgeLoader:
    """
    Discovers ClawHub skill folders in `clawhub.skills_dir` and registers them
    as ClawhubSkill instances in the shared SkillRegistry.
    """

    def load_all(self, skills_dir: Path, registry: SkillRegistry, settings) -> SkillRegistry:
        """
        Walk skills_dir. For each subdirectory containing a SKILL.md:
          1. Parse with bridge_parser.py → ClawhubSkillManifest
          2. Build SkillManifest via build_neuralclaw_manifest()
          3. Instantiate ClawhubSkill with the manifest
          4. Register in SkillRegistry
          5. Log: "clawhub_bridge.loaded skill=todoist_cli tier=2"
        
        Skips dirs with no SKILL.md (warns but doesn't crash).
        Skips skills with names that clash with existing registered skills (warns).
        """
```

**Hook this into NeuralClaw startup** in `kernel/kernel.py` or wherever `SkillLoader`
is currently called:

```python
# After existing SkillLoader runs:
if settings.clawhub.enabled:
    from skills.clawhub.bridge_loader import ClawhubBridgeLoader
    clawhub_loader = ClawhubBridgeLoader()
    clawhub_loader.load_all(
        skills_dir=Path(settings.clawhub.skills_dir),
        registry=registry,
        settings=settings,
    )
    log.info("clawhub_bridge.ready", 
             skills_dir=settings.clawhub.skills_dir)
```

---

### 5.6 `onboard/clawhub_installer.py` — Install from ClawHub Registry

This handles `neuralclaw install <slug>` for ClawHub skills specifically.

```python
async def install_from_clawhub(slug: str, settings, force: bool = False) -> int:
    """
    1. Fetch skill metadata from ClawHub API:
       GET https://clawhub.ai/api/skills/{slug}
       → returns: {name, description, version, files: [{path, content}]}
    
    2. Create skill folder:
       data/clawhub/skills/{slug}/
         SKILL.md
         references/   (if present)
         scripts/      (if present)
    
    3. Parse with bridge_parser → validate the skill
    
    4. Check requirements:
       - List required env vars that aren't set yet
       - List required bins that aren't installed
       - Print warnings for missing requirements
    
    5. Write to data/clawhub/lock.json
    
    6. Print summary:
       ✓ Installed todoist-cli (Tier 2 — API skill)
         Requires env: TODOIST_API_KEY
         Add to your .env: TODOIST_API_KEY=your_key_here
         Restart NeuralClaw or run /reload-skills to activate.
    """
```

**ClawHub API endpoints to use:**
- `GET https://clawhub.ai/api/skills/{slug}` — skill metadata + file list
- `GET https://clawhub.ai/api/skills?sort=downloads&page=1` — browse/search
- `GET https://clawhub.ai/api/skills?q={query}` — search

*(Note: these are inferred from the ClawHub architecture — verify actual endpoints
by checking the `packages/schema` directory in the clawhub GitHub repo)*

---

## 6. Changes to Existing Files

### `config/settings.py`
Add `ClawhubSettings` Pydantic model and include it in the root `Settings` class:

```python
class ClawhubExecutionSettings(BaseModel):
    allow_binary_skills: bool = True
    auto_install_deps: bool = False
    sandbox_binary_skills: bool = True

class ClawhubSettings(BaseModel):
    enabled: bool = True
    skills_dir: str = "./data/clawhub/skills"
    registry_url: str = "https://clawhub.ai"
    execution: ClawhubExecutionSettings = ClawhubExecutionSettings()
    block_on_missing_env: bool = True
    show_env_requirements_on_install: bool = True

class Settings(BaseSettings):
    ...
    clawhub: ClawhubSettings = ClawhubSettings()
```

### `skills/md_loader.py`
Extend `MarkdownSkillLoader` to also accept ClawHub frontmatter format.
Currently it only handles NeuralClaw's own `.md` format — add a fallback that
detects and parses the ClawHub `metadata.openclaw` structure.

Specifically, add `_is_clawhub_format(frontmatter: dict) -> bool` and if true,
delegate to `bridge_parser.py` instead.

### `interfaces/cli.py`
Add `/clawhub` command to the REPL:

```
/clawhub search <query>       Search ClawHub registry
/clawhub install <slug>       Install a ClawHub skill
/clawhub list                 List installed ClawHub skills
/clawhub info <slug>          Show skill details + requirements
/clawhub remove <slug>        Remove an installed ClawHub skill
```

### `main.py`
Extend existing subcommand handling to route `clawhub` skills:

```python
if args.subcommand == "clawhub":
    from onboard.clawhub_installer import clawhub_command
    return await clawhub_command(args.subcommand_arg, args)
```

---

## 7. The `data/clawhub/lock.json` File

Mirrors ClawHub's own `.clawhub/lock.json` format so skills installed via
the official `clawhub` CLI are also recognized by NeuralClaw:

```json
{
  "version": 1,
  "skills": {
    "todoist-cli": {
      "version": "1.2.0",
      "installed_at": "2025-07-01T14:23:01Z",
      "skill_dir": "data/clawhub/skills/todoist-cli",
      "execution_tier": 2,
      "requires_env": ["TODOIST_API_KEY"],
      "requires_bins": [],
      "env_satisfied": true,
      "bins_satisfied": true
    }
  }
}
```

---

## 8. Safety Considerations

ClawHub skills come from the public internet. Many are community-built and
ClawHub only provides best-effort security analysis. NeuralClaw must apply
its own safety layer on top.

### What the bridge enforces

| Risk | Bridge Behavior |
|---|---|
| Skills injecting malicious shell commands | All Tier 3 commands go through `SafetyKernel` + whitelist |
| Prompt injection in SKILL.md body | LLM sees it as instructions, not trusted code — same as any user input |
| Skills exfiltrating env vars | `env_injector.py` only injects the vars declared in `requires.env` |
| Skills accessing files outside allowed paths | `filesystem` skill's `Path.is_relative_to()` guard still applies |
| Tier 3 skills running arbitrary installs | `auto_install_deps: false` by default — user must opt in |
| Unknown/risky skills | `block_on_missing_env: true` forces explicit setup before a skill can run |

### Recommended config for untrusted skills

```yaml
clawhub:
  execution:
    allow_binary_skills: false    # disable Tier 3 entirely
    auto_install_deps: false
    sandbox_binary_skills: true
```

### What to tell users

When installing any ClawHub skill, print a warning:
```
⚠  ClawHub skills are community-built. Review the SKILL.md before use:
   data/clawhub/skills/todoist-cli/SKILL.md
   ClawHub security report: https://clawhub.ai/skills/todoist-cli
```

---

## 9. Build Order (What To Do First)

Build in this order — each step is independently testable:

1. **`bridge_parser.py`** — Write tests against real `SKILL.md` examples from GitHub. This is pure parsing, no NeuralClaw dependencies. Get this solid first.

2. **`dependency_checker.py`** — Pure stdlib. Check bins via `shutil.which()`, check env via `os.environ`. Unit test with mocked env.

3. **`clawhub_skill.py` + `build_neuralclaw_manifest()`** — Build the SkillManifest from a parsed manifest. Unit test the name sanitization and tier-to-risk mapping.

4. **`bridge_executor.py` Tier 1** — Start with PromptOnlyExecutor. Integration test: install a simple prompt-only ClawHub skill and ask it a question.

5. **`bridge_loader.py`** — Hook into startup. Test that ClawHub skills appear in `/tools` in the CLI.

6. **`bridge_executor.py` Tier 2** — HttpApiExecutor. Test with a skill that calls a public API (weather, news).

7. **`onboard/clawhub_installer.py`** — Download + install from ClawHub API. Test the full `neuralclaw install todoist-cli` flow.

8. **`bridge_executor.py` Tier 3** — BinaryExecutor. Test with a skill that uses `curl` as its only binary (it'll already be installed). Then test the missing-binary warning path.

9. **Config validation** — Add `clawhub:` to `config.yaml` schema, update Pydantic Settings.

10. **CLI `/clawhub` commands** — Wire up `interfaces/cli.py` last, after the core is working.

---

## 10. Testing a Real ClawHub Skill End-to-End

Once the bridge is built, this is the full flow to verify it works:

```bash
# 1. Install a simple Tier 1 skill (no env, no bins needed)
neuralclaw install apple-notes

# 2. Start NeuralClaw CLI
python main.py

# 3. Check it loaded
/tools
# → should show "apple_notes" in the list with category=clawhub

# 4. Use it
/ask Create a note called "Test from NeuralClaw" with content "Bridge adapter works!"

# 5. Install a Tier 2 skill (API key needed)
neuralclaw install todoist-cli
# → shows: Requires env: TODOIST_API_KEY
# → add to .env, restart

# 6. Test Tier 2
/ask Add a task "Test NeuralClaw bridge" to my Todoist inbox due tomorrow

# 7. Check the SKILL.md was used
/status
# → should show ClawHub skills count in status output
```

---

## 11. What This Unlocks

Once the bridge is working, NeuralClaw gains access to the entire ClawHub ecosystem:

| Category | Example Skills Available |
|---|---|
| Productivity | Notion, Todoist, Linear, Obsidian, Apple Notes, Reminders |
| Developer | GitHub CLI, GitLab, Jira, Sentry, Vercel, Railway, Fly.io |
| Communication | Slack, Discord, Telegram bots, email management |
| AI/ML | Replicate, Hugging Face, Perplexity, browsing agents |
| Finance | Stripe, crypto monitoring, stock alerts |
| Smart Home | Home Assistant, Philips Hue, Apple HomeKit |
| Security | CVE monitoring, Shodan, security audits |
| Data | Airtable, Supabase, PlanetScale, Cloudflare R2 |
| Blockchain | Ethereum, Base, Solana, DeFi protocols |

Total: **5,700+ skills** available the day the bridge ships.
