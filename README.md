# NeuralClaw 🦾

**Local-first autonomous AI agent platform for power developers.**

NeuralClaw is personal AI infrastructure — it runs on your machine, remembers everything across sessions, executes tools on your behalf (terminal, filesystem, web search), and can be directed via CLI or Telegram. All data stays local. No SaaS.

```
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██████╗██╗      █████╗ ██╗    ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██║     ██╔══██╗██║    ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██║     ██║     ███████║██║ █╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║     ██╔══██║██║███╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗███████╗██║  ██║╚███╔███╔╝
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
```

---

## Table of Contents

1. [Features](#features)
2. [Architecture Overview](#architecture-overview)
3. [Project Structure](#project-structure)
4. [Quick Start](#quick-start)
5. [Configuration Reference](#configuration-reference)
6. [CLI Commands](#cli-commands)
7. [Trust Levels](#trust-levels)
8. [Tool System](#tool-system)
9. [Memory System](#memory-system)
10. [Safety System](#safety-system)
11. [LLM Providers](#llm-providers)
12. [MCP Integration](#mcp-integration)
13. [Telegram Interface](#telegram-interface)
14. [Adding a Custom Tool](#adding-a-custom-tool)
15. [Bug Fixes & Improvements (v1.0.1)](#bug-fixes--improvements-v101)
16. [Running Tests](#running-tests)
17. [Roadmap](#roadmap)
18. [License](#license)

---

## Features

| Capability | Status |
|---|---|
| Multi-provider LLM brain (OpenAI, Anthropic, Ollama, Gemini, Bytez, OpenRouter) | ✅ |
| 3-tier memory (short-term, ChromaDB long-term, SQLite episodic) | ✅ |
| Tool execution: filesystem, terminal, web search, web fetch | ✅ |
| Safety kernel with risk scoring, whitelist, and trust levels | ✅ |
| Interactive CLI REPL with Rich UI | ✅ |
| Autonomous multi-step task planner (`/run` mode) | ✅ |
| Task scheduler (cron + interval) | ✅ |
| Telegram bot interface | ✅ |
| MCP (Model Context Protocol) server integration | ✅ |
| Docker sandbox for terminal isolation | 🔧 Phase 9 |
| Prometheus metrics + Grafana dashboard | 🔧 Phase 10 |

---

## Architecture Overview

```
╔══════════════════════════════════════════════════════════════════════════╗
║                      NEURALCLAW AGENT PLATFORM                          ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                      INTERFACE LAYER                            │    ║
║  │         Telegram Bot          │         CLI (Rich REPL)         │    ║
║  └──────────────────────────────┬──────────────────────────────────┘    ║
║                                 ▼                                        ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    AGENT ORCHESTRATOR                           │    ║
║  │    Context Build → LLM Call → Tool Dispatch → Reflect          │    ║
║  │    ┌───────────┐  ┌────────────┐  ┌──────────────────────┐     │    ║
║  │    │  Planner  │  │  Reasoner  │  │ Response Synthesizer │     │    ║
║  │    └───────────┘  └────────────┘  └──────────────────────┘     │    ║
║  └──────────────────────────────┬──────────────────────────────────┘    ║
║              ┌───────────────────┼──────────────────┐                   ║
║              ▼                   ▼                  ▼                    ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐    ║
║  │    LLM BRAIN     │  │  MEMORY ENGINE   │  │     TOOL BUS       │    ║
║  │  OpenAI          │  │  Short-term buf  │  │  filesystem        │    ║
║  │  Anthropic       │  │  ChromaDB vector │  │  terminal          │    ║
║  │  Ollama          │  │  SQLite episodic │  │  web_search        │    ║
║  │  Gemini / Bytez  │  └──────────────────┘  │  web_fetch / MCP   │    ║
║  └──────────────────┘                        └────────────────────┘    ║
║                                                        │                 ║
║  ┌─────────────────────────────────────────────────────┘                ║
║  │                     SAFETY KERNEL                                    ║
║  │  Permission Check → Risk Score → Whitelist → Confirm Gate           ║
║  └──────────────────────────────────────────────────────────────────────╝
```

### Agent Loop

The agent operates in a continuous **observe → context build → think → act → reflect** cycle:

1. **Observe** — receive user input or scheduled trigger
2. **Context Build** — pull short-term history, semantic long-term memories, active plan
3. **Think** — call LLM with context + tool schemas; receive text response or tool call(s)
4. **Act** — route tool calls through safety kernel, execute, collect results
5. **Reflect** — feed results back to LLM, repeat until done, then commit episode to memory

Maximum iterations per turn is configurable (default 10). Each turn has a max timeout of 5 minutes.

---

## Project Structure

```
NeuralClaw-fixed/
├── main.py                        # Entry point
├── pyproject.toml
├── requirements.txt
├── conftest.py
│
├── agent/
│   ├── context_builder.py         # Builds LLM prompt from memory + session
│   ├── orchestrator.py            # Core observe→think→act→reflect loop
│   ├── planner.py                 # Goal decomposition for /run mode
│   ├── reasoner.py                # Pre-flight chain-of-thought for risky actions
│   ├── response_synthesizer.py    # Formats agent output for UI
│   └── session.py                 # Per-session state (trust, history, plan)
│
├── brain/
│   ├── llm_client.py              # Abstract base + LLMClientFactory
│   ├── types.py                   # Message, LLMResponse, ToolCall, LLMConfig
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── ollama_client.py
│   ├── openrouter_client.py
│   ├── gemini_client.py
│   └── bytez_client.py
│
├── config/
│   ├── config.yaml                # All non-secret configuration
│   └── settings.py                # Pydantic settings (merges yaml + .env)
│
├── interfaces/
│   ├── cli.py                     # Rich-powered interactive REPL
│   └── telegram.py                # Telegram bot interface
│
├── mcp/
│   ├── manager.py                 # Multi-server MCP manager
│   ├── types.py                   # MCP protocol types
│   └── transports/
│       ├── base.py
│       ├── http.py
│       └── stdio.py
│
├── memory/
│   ├── memory_manager.py          # Unified facade — use this
│   ├── short_term.py              # In-memory sliding conversation window
│   ├── long_term.py               # ChromaDB vector store
│   ├── episodic.py                # SQLite: episodes, tool calls, reflections
│   └── embedder.py                # sentence-transformers wrapper
│
├── observability/
│   └── logger.py                  # Structured JSON logging (structlog)
│
├── safety/
│   ├── safety_kernel.py           # Gatekeeper: APPROVED | BLOCKED | CONFIRM_NEEDED
│   ├── whitelist.py               # Command + path allowlists
│   └── risk_scorer.py             # Heuristic risk level assignment
│
├── scheduler/
│   └── scheduler.py               # Cron + interval task scheduler
│
├── tools/
│   ├── tool_registry.py           # Self-registration via @registry.register
│   ├── tool_bus.py                # Dispatcher: registry → safety → execution
│   ├── types.py                   # ToolCall, ToolResult, ToolSchema, RiskLevel
│   ├── filesystem.py              # file_read, file_write, file_append, list_dir
│   ├── terminal.py                # terminal_exec (async subprocess)
│   ├── search.py                  # web_search
│   ├── web_fetch.py               # web_fetch
│   └── plugins/                   # Drop custom tools here
│
├── tests/
│   └── unit/
│       ├── test_brain.py
│       ├── test_cli.py
│       ├── test_memory.py
│       ├── test_safety.py
│       └── test_tool_bus.py
│
└── docs/
    ├── Architecture.md
    ├── Mvp_Tech.md
    ├── PRD.md
    └── System_Design.md
```

**Total: 75 files** (60 Python + config, docs, docker, tests)

---

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) for local models **or** an API key from any supported provider

### Install

```bash
git clone https://github.com/your-username/neuralclaw
cd neuralclaw
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For browser automation (optional):
```bash
playwright install chromium
```

### Configure

```bash
cp .env.example .env
```

Edit `.env` with your key(s). At minimum, set one of:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
BYTEZ_API_KEY=...
GEMINI_API_KEY=...
OPENROUTER_API_KEY=...
# Or leave all empty and run Ollama locally
```

Then set the matching provider in `config/config.yaml`:

```yaml
llm:
  default_provider: "anthropic"   # openai | anthropic | ollama | gemini | bytez | openrouter
  default_model: "claude-sonnet-4-5"
```

### Run

```bash
# Start the CLI (default)
python main.py

# Verbose logging
python main.py --log-level DEBUG

# Custom config file
python main.py --config /path/to/config.yaml

# Telegram interface
python main.py --interface telegram
```

---

## Configuration Reference

`config/config.yaml` is the single non-secret config file. Secrets go in `.env`.

```yaml
agent:
  name: "NeuralClaw"
  version: "1.0.0"
  max_iterations_per_turn: 10       # Max tool-call loops per user message
  max_turn_timeout_seconds: 300     # Hard timeout per turn
  default_trust_level: "low"        # low | medium | high

llm:
  default_provider: "bytez"         # openai | anthropic | ollama | gemini | bytez | openrouter
  default_model: "openai/gpt-5"
  temperature: 0.4
  max_tokens: 4096

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/sqlite/episodes.db"
  embedding_model: "BAAI/bge-small-en-v1.5"   # Any sentence-transformers model
  max_short_term_turns: 20
  relevance_threshold: 0.85

tools:
  terminal:
    working_dir: "./data/agent_files"
    default_timeout_seconds: 30
    docker_sandbox: false
    whitelist_extra: []               # Extra allowed commands beyond defaults
  filesystem:
    allowed_paths:
      - "./data/agent_files"          # Agent can only read/write here

safety:
  require_confirmation_for:
    - "HIGH"
    - "CRITICAL"

mcp:
  servers:
    blender:
      transport: stdio
      command: "uvx"
      args: ["blender-mcp"]
      enabled: false

scheduler:
  timezone: "UTC"
  max_concurrent_tasks: 3

logging:
  level: "INFO"
  log_dir: "./data/logs"
  json_format: true
```

Environment variables always override `config.yaml` values. Use `__` as the nested delimiter (e.g. `LLM__DEFAULT_PROVIDER=openai`).

---

## CLI Commands

| Command | Description |
|---|---|
| Just type anything | Send a message (same as `/ask`) |
| `/ask <message>` | Interactive turn — agent responds with text and/or tool calls |
| `/run <goal>` | Autonomous multi-step task execution |
| `/status` | Show session stats, memory usage, current settings |
| `/memory <query>` | Semantic search over long-term memory |
| `/tools` | List all registered tools with risk levels |
| `/trust <low\|medium\|high>` | Set the session trust level |
| `/clear` | Clear conversation history |
| `/cancel` | Cancel the currently running task |
| `/help` | Show help |
| `exit` / `quit` / Ctrl+D | Exit cleanly (flushes all memory connections) |

---

## Trust Levels

| Level | Auto-approves | Requires confirmation |
|---|---|---|
| `low` (default) | LOW + MEDIUM risk | HIGH + CRITICAL |
| `medium` | LOW + MEDIUM + HIGH risk | CRITICAL only |
| `high` | Everything | Nothing — use with care |

---

## Tool System

Tools self-register via a decorator. The tool bus routes every LLM tool call through:

1. **Registry lookup** — is the tool registered?
2. **Parameter validation** — required fields present + type checking against JSON schema
3. **Safety kernel** — risk score, path/command allowlist, confirmation gate
4. **Execution** — async handler with timeout enforcement
5. **Result normalization** — always returns a `ToolResult` (never raises)

### Built-in Tools

| Tool | Risk | Description |
|---|---|---|
| `file_read` | LOW | Read a file (max 10 MB cap) |
| `file_write` | MEDIUM | Write/overwrite a file |
| `file_append` | MEDIUM | Append to a file |
| `list_dir` | LOW | List directory contents |
| `terminal_exec` | HIGH | Execute a shell command (whitelist enforced) |
| `web_search` | LOW | Search the web via SerpAPI |
| `web_fetch` | LOW | Fetch and parse a URL |

### Tool Risk Levels

| Risk | Colour | Default Behaviour |
|---|---|---|
| `LOW` | 🟢 | Auto-execute |
| `MEDIUM` | 🟡 | Auto-execute |
| `HIGH` | 🟠 | Confirm at `low` trust |
| `CRITICAL` | 🔴 | Confirm at `low` + `medium` trust |

---

## Memory System

Three complementary tiers work together transparently via `MemoryManager`:

**Short-term** (`memory/short_term.py`) — in-process sliding window of the last N conversation turns. Eviction counts by user messages (one per turn) so tool-result and assistant messages mid-turn don't cause premature context loss.

**Long-term** (`memory/long_term.py`) — ChromaDB vector store for semantic search over past interactions, tool results, and agent knowledge. Uses sentence-transformers embeddings locally (default: `BAAI/bge-small-en-v1.5`). Supports multiple named collections: `conversations`, `knowledge`, `tool_results`, `plans`.

**Episodic** (`memory/episodic.py`) — SQLite store for structured records of completed episodes, individual tool call logs, and agent reflections/lessons-learned. Searchable by session, date, outcome, and LIKE-escaped keyword.

```python
# All memory access goes through MemoryManager
memory = MemoryManager.from_settings(settings)
await memory.init()

# Store
await memory.store("The user prefers Python 3.11", collection="knowledge")

# Search
results = await memory.search_all("python async patterns", n_per_collection=3)

# Episode tracking
episode_id = await memory.episodic.start_episode(session_id, goal="Summarise PRs")
await memory.episodic.finish_episode(episode_id, outcome="success", summary="...")

# Always close cleanly (flushes SQLite + shuts down thread pools)
await memory.close()
```

---

## Safety System

Every tool call passes through three layers before execution:

**1. Risk Scoring** (`safety/risk_scorer.py`) — assigns `LOW / MEDIUM / HIGH / CRITICAL` based on tool name and arguments.

**2. Path & Command Whitelist** (`safety/whitelist.py`) — filesystem tools check the path against `allowed_paths`; terminal tools check the command against an allowlist. Write operations (`write`, `create`, `append`, `delete`, `move`, `copy`) are checked against write-path restrictions.

**3. Confirmation Gate** (`safety/safety_kernel.py`) — actions above the session's trust threshold are held and a confirmation request is sent to the UI (CLI prompt or Telegram inline keyboard). Unresolved confirmations time out after 120 seconds and are denied. Timed-out futures are cleaned up immediately to prevent memory leaks.

---

## LLM Providers

| Provider | Key Env Var | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | GPT-4o, GPT-4-turbo, etc. |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 3.5 Sonnet, Opus, Haiku |
| `ollama` | _(none — runs locally)_ | Set `OLLAMA_BASE_URL` if non-default |
| `gemini` | `GEMINI_API_KEY` | Gemini 1.5 Pro, Flash |
| `bytez` | `BYTEZ_API_KEY` | Default provider in config |
| `openrouter` | `OPENROUTER_API_KEY` | Access any model via OpenRouter |

All providers implement the same `BaseLLMClient` interface. Switching provider is a one-line config change.

Startup validates the required API key for the configured provider and exits cleanly with a descriptive error rather than crashing mid-run.

---

## MCP Integration

NeuralClaw supports the [Model Context Protocol](https://modelcontextprotocol.io) for connecting external tool servers. MCP tools are discovered automatically on connect and registered in the global tool registry with a namespaced prefix (e.g. `blender::create_mesh`).

Configure servers in `config.yaml`:

```yaml
mcp:
  servers:
    blender:
      transport: stdio        # stdio | http
      command: "uvx"
      args: ["blender-mcp"]
      enabled: true
    my_server:
      transport: http
      command: "python"
      args: ["-m", "my_mcp_server"]
      enabled: true
```

Supported transports: `stdio` (local subprocess) and `http`.

---

## Telegram Interface

```bash
# Set in .env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_USER_ID=your_user_id   # Allowlist your chat ID
```

```bash
python main.py --interface telegram
```

### Telegram Commands

| Command | Description |
|---|---|
| `/ask <message>` | Interactive turn |
| `/run <goal>` | Autonomous task |
| `/status` | Session stats |
| `/memory <query>` | Search memory |
| `/tools` | List tools |
| `/trust <level>` | Set trust level |
| `/cancel` | Cancel running task |
| `/clear` | Clear history |
| `/confirm` | Approve a pending high-risk action |

---

## Adding a Custom Tool

Drop a new file in `tools/plugins/` or anywhere in `tools/`:

```python
# tools/my_tool.py
from tools.tool_registry import registry
from tools.types import RiskLevel

@registry.register(
    name="my_tool",
    description="What this tool does — shown to the LLM",
    category="custom",
    risk_level=RiskLevel.LOW,
    parameters={
        "type": "object",
        "properties": {
            "input": {"type": "string", "description": "The input value"},
        },
        "required": ["input"],
    },
)
async def my_tool(input: str) -> str:
    return f"Processed: {input}"
```

Then import it in `interfaces/cli.py` so it self-registers at startup:

```python
import tools.my_tool  # noqa: F401
```

---

## Bug Fixes & Improvements (v1.0.1)

This release patches **12 bugs** and implements **5 improvements** across **15 files**.

### 🔴 High Priority Bug Fixes

**Bug #1 — EpisodicMemory crash after `close()`** (`memory/episodic.py`, `memory/memory_manager.py`)
All public `EpisodicMemory` methods now call `_require_db()` which raises a clear `RuntimeError` instead of `AttributeError: 'NoneType' has no attribute 'execute'`. `MemoryManager.close()` resets `_initialized = False` so the guard works correctly on re-use after close.

**Bug #10 — `file_append` bypassed write-safety checks** (`safety/safety_kernel.py`)
Operation detection previously only matched `"write"` and `"create"`. Tools named `file_append`, `file_delete`, `file_move`, and `file_copy` were all evaluated as read operations and skipped path-write restrictions entirely. Fixed by checking against a full keyword tuple: `("write", "create", "delete", "remove", "append", "move", "copy")`.

**Bug #24 — Scheduler FAILED tasks never retried** (`scheduler/scheduler.py`)
After a task failure, `_advance_schedule()` scheduled the next run but left `status = FAILED`. Since `_check_and_fire()` only fires `status == IDLE` tasks, any failed task was silently abandoned forever. Fixed by resetting `task.status = TaskStatus.IDLE` after scheduling the next run.

### 🟠 Medium Priority Bug Fixes

**Bug #2 — `assert` in production code** (`agent/orchestrator.py`)
`assert all(r is not None for r in results)` was silently stripped when Python ran with `-O`. Replaced with an explicit `RuntimeError` that survives optimized execution.

**Bug #5 — No file-size cap on `file_read`** (`tools/filesystem.py`)
Reading a multi-GB file into the LLM context caused OOM crashes. `file_read` now enforces a **10 MB hard cap** and returns a descriptive message directing the LLM to use `terminal_exec` with `head`, `tail`, or `grep` for large files.

**Bug #14 — Race condition in parallel confirmation callbacks** (`tools/tool_bus.py`, `agent/orchestrator.py`)
LOW-risk tool calls ran in parallel via `asyncio.gather`. Each call temporarily swapped `self._bus.on_confirm_needed`, creating a race where two simultaneous confirmations would overwrite each other's callback. Fixed by adding a `per_call on_confirm_needed` parameter to `ToolBus.dispatch()` — the callback is passed directly instead of mutating shared state.

**Bug #15 — Conversation buffer eviction miscounts** (`memory/short_term.py`)
Eviction fired when `len(messages) > max_turns * 2`, assuming each turn produces exactly 2 messages. Tool-result and assistant-with-tool-calls messages add 3+ messages per turn, causing premature eviction. Fixed by counting only `USER` messages (one per turn) for eviction decisions.

### 🟡 Low Priority Bug Fixes

**Bug #8 — Scheduler first tick delayed** (`scheduler/scheduler.py`)
`_tick_loop` slept *before* the first check. A task due immediately after startup waited a full `tick_interval` (default 30s). Fixed by checking first, then sleeping.

**Bug #12 — `get_settings()` / `load_settings()` could diverge** (`config/settings.py`)
`get_settings()` was `lru_cache`d with the default config path. If `main.py` called `load_settings("custom.yaml")` first, any other module calling `get_settings()` got a different instance loaded from the default path. Fixed with an explicit `_singleton` module variable. `load_settings()` registers itself as the singleton on first call; `get_settings()` always returns it.

**Bug #13 — Recovery step indices out of sync** (`agent/orchestrator.py`)
After inserting recovery steps mid-plan, existing steps after the insertion point kept their old `index` values. Fixed by renumbering all steps after insertion: `for j, s in enumerate(plan.steps): s.index = j`.

**Bug #17 — SQL `LIKE` injection in episode search** (`memory/episodic.py`)
User queries containing `%` or `_` were treated as SQL wildcards, returning unexpected results. Fixed by escaping `\`, `%`, and `_` before building the LIKE pattern and adding `ESCAPE '\\'` to the SQL.

**Bug #23 — Confirmation future leaks on timeout** (`agent/orchestrator.py`)
When a 120-second confirmation timeout expired, the code returned `False` but left the future in `session._pending_confirmations`. Over many timeouts this dict grew unboundedly. Fixed by calling `session._pending_confirmations.pop(tool_call_id, None)` in the timeout handler.

### 🔵 Improvements

**#11 — ThreadPoolExecutor shutdown** (`memory/long_term.py`, `memory/embedder.py`, `memory/memory_manager.py`)
`LongTermMemory` and `Embedder` each create a `ThreadPoolExecutor` that was never shut down, causing `RuntimeError: cannot schedule new futures after interpreter shutdown` warnings on exit. Both now have `close()` methods calling `executor.shutdown(wait=True)`, wired through `MemoryManager.close()`.

**#19 — Sub-config classes converted to Pydantic models** (`config/settings.py` + call sites)
`AgentConfig`, `LLMConfig`, `MemoryConfig`, `ToolsConfig`, `SafetyConfig`, `MCPConfig`, `TelegramConfig`, `SchedulerConfig`, and `LoggingConfig` were plain Python classes with type annotations but no validation — never actually instantiated. All converted to `pydantic.BaseModel` subclasses. `Settings` fields changed from `dict[str, Any]` to the typed models. Field validators coerce raw YAML dicts at load time. All 30+ `.get("key", default)` call sites across `main.py`, `orchestrator.py`, `memory_manager.py`, `cli.py`, and `telegram.py` updated to direct attribute access. Bad config values now raise a `ValidationError` at startup rather than crashing mid-run.

**#21 — ToolBus argument type validation** (`tools/tool_bus.py`)
`_validate_args()` previously only checked that required fields were present. It now also validates the JSON Schema `type` of every provided argument (`string`, `integer`, `number`, `boolean`, `array`, `object`) with special-casing for `bool`-as-`int` in Python. LLM-generated type mismatches are caught before the tool handler runs.

**#25 — Anthropic `health_check` used a hardcoded deprecated model** (`brain/anthropic_client.py`)
`health_check()` called `messages.create(model="claude-3-haiku-20240307", ...)` — consuming tokens and risking breakage if that model is retired. Replaced with `client.models.list()`: no tokens consumed, no hardcoded model string.

**#26 — CLI memory leak on exit** (`interfaces/cli.py`)
`start()` called `_repl_loop()` with no cleanup guarantee. Ctrl+D or an exception left the aiosqlite connection and thread pool executors open. Fixed by wrapping `_repl_loop()` in `try/finally: await self._cleanup()`.

### Files Changed

| File | Changes |
|---|---|
| `agent/orchestrator.py` | Assert → RuntimeError; recovery step renumbering; confirmation race fix; timeout future cleanup |
| `brain/anthropic_client.py` | health_check uses `models.list()` |
| `config/settings.py` | Sub-models → Pydantic; singleton loader fix |
| `interfaces/cli.py` | `try/finally` cleanup; attribute access for settings |
| `interfaces/telegram.py` | Attribute access for all settings sub-fields |
| `main.py` | Attribute access for settings |
| `memory/embedder.py` | `close()` method for executor shutdown |
| `memory/episodic.py` | `_require_db()` guard on all public methods; LIKE escape |
| `memory/long_term.py` | `close()` method for executor shutdown |
| `memory/memory_manager.py` | `_initialized = False` on close; call `long_term.close()` + `embedder.close()` |
| `memory/short_term.py` | Eviction counts USER messages only |
| `safety/safety_kernel.py` | Write keyword tuple includes append/delete/move/copy |
| `scheduler/scheduler.py` | Check before sleep; reset FAILED → IDLE after rescheduling |
| `tools/filesystem.py` | 10 MB file-read cap |
| `tools/tool_bus.py` | Per-call confirmation callback; JSON schema type validation |

---

## Running Tests

```bash
pytest tests/unit/ -v
pytest tests/unit/test_safety.py -v          # Safety kernel only
pytest tests/unit/test_tool_bus.py -v        # Tool bus only
pytest tests/unit/ --cov=. --cov-report=html # Coverage report
```

---

## Roadmap

- **Phase 9** — Docker sandbox for terminal isolation
- **Phase 10** — Prometheus metrics + Grafana dashboard
- **Future** — LLM streaming support for real-time output
- **Future** — Scheduler task persistence across restarts
- **Future** — Retry/backoff for LLM rate limit errors
- **Future** — Tiktoken-based token counting (replace character proxy)

---

## License

MIT — personal project, use freely.
