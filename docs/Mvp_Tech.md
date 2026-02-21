# Mvp_Tech.md — NeuralClaw MVP Technical Specification

> **Version:** 1.0.0  
> **Goal:** Working prototype in 8-10 weeks (solo developer)  
> **Philosophy:** Build the simplest thing that actually works end-to-end, then expand

---

## Table of Contents

1. [Exact Tech Stack](#1-exact-tech-stack)
2. [Why Each Tool Was Chosen](#2-why-each-tool-was-chosen)
3. [MVP Scope Definition](#3-mvp-scope-definition)
4. [Step-by-Step Build Order](#4-step-by-step-build-order)
5. [Dev Environment Setup](#5-dev-environment-setup)
6. [Dependency List](#6-dependency-list)
7. [Repo Structure](#7-repo-structure)
8. [Timeline to Build MVP](#8-timeline-to-build-mvp)
9. [What NOT to Build Initially](#9-what-not-to-build-initially)
10. [Testing Plan](#10-testing-plan)
11. [Configuration Reference](#11-configuration-reference)
12. [First Run Checklist](#12-first-run-checklist)

---

## 1. Exact Tech Stack

### Language & Runtime

| Component | Technology | Version |
|---|---|---|
| Primary language | Python | 3.11+ |
| Async runtime | asyncio (stdlib) | — |
| Package management | pip + pyproject.toml | — |
| Container | Docker + Docker Compose | 24+ |

### LLM Integration

| Component | Technology | Version |
|---|---|---|
| OpenAI client | openai | 1.40+ |
| Anthropic client | anthropic | 0.30+ |
| Local LLM | ollama (external process) | latest |
| Ollama Python | ollama | 0.3+ |
| Token counting | tiktoken | 0.7+ |

### Memory & Storage

| Component | Technology | Version |
|---|---|---|
| Vector database | chromadb | 0.5+ |
| Embedding model | sentence-transformers | 3.0+ |
| Embedding backend | BAAI/bge-small-en-v1.5 | — |
| Relational store | SQLite via aiosqlite | 0.20+ |
| Key-value config | PyYAML + python-dotenv | — |

### Tools

| Component | Technology | Version |
|---|---|---|
| Browser automation | playwright | 1.45+ |
| Browser engine | Chromium (via Playwright) | — |
| HTTP client | httpx | 0.27+ |
| HTML parsing | beautifulsoup4 + lxml | 4.12+ |
| Safe Python exec | RestrictedPython | 7.0+ |

### MCP Integration

| Component | Technology | Version |
|---|---|---|
| MCP protocol | mcp (official SDK) | 1.0+ |
| stdio transport | asyncio subprocess | — |
| HTTP/SSE transport | httpx | 0.27+ |

### Interface

| Component | Technology | Version |
|---|---|---|
| Telegram bot | aiogram | 3.10+ |
| CLI interface | aioconsole + rich | — |
| Rich terminal UI | rich | 13.7+ |
| Input validation | pydantic | 2.7+ |

### Scheduling & Tasks

| Component | Technology | Version |
|---|---|---|
| Task scheduler | APScheduler | 4.0+ |
| Async task queue | asyncio.Queue (stdlib) | — |
| Job persistence | APScheduler SQLAlchemy | — |

### Safety & Security

| Component | Technology | Version |
|---|---|---|
| Config secrets | python-dotenv | 1.0+ |
| JSON Schema validation | jsonschema | 4.23+ |
| Restricted execution | RestrictedPython | 7.0+ |
| Path validation | pathlib (stdlib) | — |

### Observability

| Component | Technology | Version |
|---|---|---|
| Structured logging | structlog | 24.2+ |
| Log formatting | structlog JSON renderer | — |
| Error tracking | Python traceback (stdlib) | — |

### Testing

| Component | Technology | Version |
|---|---|---|
| Test framework | pytest | 8.2+ |
| Async tests | pytest-asyncio | 0.23+ |
| Mocking | pytest-mock + unittest.mock | — |
| Coverage | pytest-cov | 5.0+ |

### Dev Tools

| Component | Technology | Version |
|---|---|---|
| Code formatting | ruff | 0.5+ |
| Type checking | mypy | 1.10+ |
| Pre-commit hooks | pre-commit | 3.7+ |
| Dependency audit | pip-audit | 2.7+ |

---

## 2. Why Each Tool Was Chosen

### Python 3.11+
The natural choice for AI/ML tooling. All major LLM SDKs are Python-first. 3.11+ brings significant performance improvements over 3.10 and earlier. `asyncio` is mature and well-supported.

### openai SDK
The de-facto standard for LLM API clients. The SDK is also used for any OpenAI-compatible endpoint (Ollama, LM Studio, Together AI). Using the official SDK means automatic retry handling, streaming support, and type safety.

### anthropic SDK
Official Anthropic Python client. Claude 3.5 Sonnet is frequently the best-performing model for agentic tasks with tool use, making Anthropic support non-negotiable.

### ChromaDB
Embedded vector database that runs entirely in-process with zero server setup. Data persists to disk automatically. Python-native, excellent docs, actively maintained. Alternative (Qdrant, Weaviate) require running a separate server — unnecessary complexity for MVP.

### sentence-transformers
Local text embedding with `all-MiniLM-L6-v2` or `BAAI/bge-small-en-v1.5`. This means we never send memory content to an external API for embedding — crucial for privacy and offline capability. Fast enough on CPU for personal use.

### aiosqlite
Async wrapper around SQLite. Perfect for episodic memory and structured data on a single machine. Zero admin overhead. Well-tested, reliable, and the data is a plain file you can inspect with any SQLite tool.

### Playwright (Python)
The best browser automation library available. Supports Chromium, Firefox, WebKit. Handles modern SPAs (JavaScript-heavy sites) that Selenium/requests can't reach. Official Python async API (`playwright.async_api`) works natively with asyncio.

### aiogram 3.x
The most capable async Telegram bot framework for Python. Handles polling and webhooks, has built-in FSM (finite state machine) for multi-step conversations, supports inline keyboards for confirmation dialogs, and has a clean middleware architecture.

### mcp SDK
The official MCP Python SDK from Anthropic. Provides the correct wire protocol implementation, prevents protocol bugs, and will track spec updates. Preferred over a custom implementation.

### APScheduler 4.x
Mature Python scheduling library with async support, persistent job stores (SQLAlchemy/SQLite), and multiple trigger types (cron, interval, date). The 4.x async rewrite makes it ideal for our asyncio architecture.

### structlog
Structured logging with JSON output. Every log entry is a dictionary (not a plain string), which makes automated parsing, searching, and analysis trivial. Much better than Python's `logging` module for production use.

### Pydantic v2
Used for all config parsing, API response validation, and data models. Pydantic v2 is 5-50x faster than v1 and provides excellent type coercion and validation error messages.

### rich
Makes CLI output beautiful with colors, tables, progress bars, and markdown rendering. Dramatically improves developer experience during development and debugging.

### ruff
Blazingly fast Python linter and formatter (replaces flake8, black, isort). Single tool for all code quality enforcement.

---

## 3. MVP Scope Definition

### In Scope (MVP)

These features MUST work before v1.0 is called "done":

```
✅ LLM Brain
   - OpenAI API (GPT-4o)
   - Anthropic API (Claude 3.5 Sonnet)
   - Tool calling (both providers)
   - Context window management (auto-truncation)

✅ Memory
   - Short-term: last 20 conversation turns
   - Long-term: ChromaDB with local embeddings
   - Episodic: SQLite episode records
   - Memory search command

✅ Telegram Interface
   - /ask, /run, /status, /cancel, /help commands
   - Inline keyboard confirmations
   - File sending to user
   - Authorized user ID restriction

✅ CLI Interface
   - Interactive REPL
   - All core commands work in CLI

✅ Browser Tool
   - Navigate to URL + extract text
   - Screenshot
   - Click + fill forms

✅ Terminal Tool
   - Execute whitelisted commands
   - Capture stdout/stderr
   - Configurable timeout

✅ Filesystem Tool
   - Read/write text files
   - List directories
   - Restricted to allowed paths

✅ Web Search Tool
   - DuckDuckGo search (free, no API key)
   - Return top-5 results with snippets

✅ Safety Kernel
   - Command whitelist (terminal)
   - Path restriction (filesystem)
   - Risk scoring
   - Confirmation gate (HIGH risk → Telegram confirm)

✅ MCP Support
   - Blender MCP connection (stdio)
   - Tool auto-discovery
   - Namespaced tool registration

✅ Plugin System
   - Auto-load from tools/plugins/ directory
   - Decorator-based registration
   - Example plugin included

✅ Logging
   - Structured JSON logs (all tool calls, LLM calls)
   - Log file rotation

✅ Configuration
   - config.yaml + .env for secrets
   - All behavior configurable without code changes

✅ Docker
   - Dockerfile for the main agent
   - docker-compose.yml with volume mounts
```

### Out of Scope (MVP)

Deferred to v1.1+:

```
❌ Web dashboard (UI)
❌ Multi-agent system
❌ Voice interface
❌ Agent-initiated notifications (proactive monitoring — MVP does reactive only)
❌ Docker sandboxing for terminal (subprocess sandbox is MVP)
❌ Self-improvement / plugin auto-generation
❌ REST API for external integrations
❌ Remote/VPS-specific deployment automation
❌ Advanced re-planning on failure (simple error reporting is MVP)
❌ SSE MCP transport (stdio + HTTP is MVP)
❌ Streaming LLM responses to Telegram (polling is MVP)
```

---

## 4. Step-by-Step Build Order

Build in this exact order. Each phase produces a working, testable artifact.

---

### Phase 1: Project Foundation (Day 1-2)

**Goal:** Running Python project with config, logging, types

**Steps:**
1. Create repo, init git, create `.gitignore`
2. Set up `pyproject.toml` with project metadata and dependencies
3. Create virtual environment and install core deps
4. Create `config/settings.py` (Pydantic BaseSettings)
5. Create `config/config.yaml` with all defaults
6. Create `observability/logger.py` (structlog setup)
7. Create `main.py` entry point (just starts logger and prints config)

**Test:** `python main.py` — should start, log to console, exit cleanly

```bash
$ python main.py
{"event": "NeuralClaw starting", "version": "1.0.0", "level": "info"}
{"event": "Config loaded", "provider": "openai", "level": "info"}
{"event": "Startup complete", "level": "info"}
```

---

### Phase 2: LLM Brain (Day 2-4)

**Goal:** Agent can call LLM and get a response

**Steps:**
1. Create `brain/types.py` (Message, LLMResponse, LLMConfig, ToolCall)
2. Create `brain/llm_client.py` (abstract base)
3. Create `brain/openai_client.py` (OpenAI implementation)
4. Create `brain/anthropic_client.py` (Anthropic implementation)
5. Create `brain/__init__.py` with `LLMClientFactory`
6. Write unit tests for both clients (mocked API calls)

**Test:**
```python
# test: python -c "import asyncio; from brain import LLMClientFactory; ..."
client = LLMClientFactory.create("openai", {"api_key": "..."})
response = asyncio.run(client.generate([Message(role=SYSTEM, content="You are helpful"), 
                                        Message(role=USER, content="Say hello")]))
assert response.content is not None
assert response.input_tokens > 0
```

---

### Phase 3: Tool System Foundation (Day 4-6)

**Goal:** Tool registry, bus, safety kernel, basic tools

**Steps:**
1. Create `tools/types.py` (ToolCall, ToolResult, ToolSchema, RiskLevel)
2. Create `tools/tool_registry.py` (registration, lookup, schema storage)
3. Create `safety/permissions.py` (RiskLevel enum, TrustLevel)
4. Create `safety/whitelist.py` (command and path whitelists)
5. Create `safety/risk_scorer.py` (simple rule-based scoring)
6. Create `safety/safety_kernel.py` (evaluate → APPROVED/BLOCKED/CONFIRM_NEEDED)
7. Create `tools/tool_bus.py` (dispatch → validate → safety → execute → result)
8. Create `tools/filesystem.py` + register tools (file_read, file_write, list_dir)
9. Create `tools/terminal.py` + register tools (terminal_exec)
10. Create `tools/search.py` (DuckDuckGo via httpx) + register
11. Write unit tests for safety kernel and filesystem tool

**Test:**
```python
# test: filesystem tool reads a file
result = asyncio.run(tool_bus.execute(
    ToolCall(id="1", name="file_read", arguments={"path": "/tmp/test.txt"}),
    session=test_session
))
assert result.success
assert "hello" in result.content

# test: terminal whitelist blocks rm
with pytest.raises(SecurityError):
    asyncio.run(tool_bus.execute(
        ToolCall(id="2", name="terminal_exec", arguments={"command": "rm -rf /"}),
        session=test_session
    ))
```

---

### Phase 4: Memory System (Day 6-8)

**Goal:** Short-term and long-term memory working

**Steps:**
1. Create `memory/short_term.py` (ConversationBuffer class)
2. Create `memory/embedder.py` (sentence-transformers wrapper, async via executor)
3. Create `memory/long_term.py` (ChromaDB wrapper with collections)
4. Create `memory/episodic.py` (SQLite via aiosqlite, schema + CRUD)
5. Create `memory/memory_manager.py` (unified interface)
6. Write integration tests (store → search round-trip)

**Test:**
```python
# Round-trip test
asyncio.run(memory.store("Python async programming is great for IO tasks", collection="knowledge"))
results = asyncio.run(memory.search("async python", n=3))
assert len(results) >= 1
assert results[0].distance < 0.5  # Very relevant
```

---

### Phase 5: Agent Orchestrator (Day 8-11)

**Goal:** Full agent loop working end-to-end (CLI only)

**Steps:**
1. Create `agent/context_builder.py` (builds messages list from memory + history)
2. Create `agent/session.py` (Session class, ConversationBuffer, SessionState)
3. Create `agent/orchestrator.py` (main process_message loop)
4. Wire: Orchestrator → LLM → ToolBus → Memory
5. Create `interfaces/cli.py` (simple REPL using rich + aioconsole)
6. Create `main.py` entry with `--interface cli`
7. Manual end-to-end test

**Test (manual):**
```
$ python main.py --interface cli

NeuralClaw> What Python library should I use for async HTTP?
[LLM responds with recommendation]

NeuralClaw> Read the file at /tmp/notes.txt and summarize it
[Tool: file_read → LLM summarizes content]

NeuralClaw> Run: ls -la ~/projects
[Safety: MEDIUM - approved]
[Tool: terminal_exec → output returned to LLM → summarized]
```

---

### Phase 6: Browser Tool (Day 11-13)

**Goal:** Agent can browse the web

**Steps:**
1. `pip install playwright && playwright install chromium`
2. Create `tools/browser.py` (BrowserTool class)
3. Register tools: `browser_navigate`, `browser_screenshot`, `browser_click`, `browser_fill`
4. Wire BrowserTool lifecycle (start on agent startup, cleanup on shutdown)
5. Test with real URLs

**Test (manual):**
```
NeuralClaw> Go to https://news.ycombinator.com and tell me the top 3 post titles

[Tool: browser_navigate → content extracted]
[LLM: "The top 3 HN posts are: 1. ... 2. ... 3. ..."]
```

---

### Phase 7: Telegram Interface (Day 13-16)

**Goal:** Agent controlled from phone

**Steps:**
1. Create Telegram bot via @BotFather, get token
2. `pip install aiogram`
3. Create `interfaces/telegram_bot.py`
4. Implement command handlers: `/ask`, `/status`, `/help`, `/cancel`
5. Implement confirmation inline keyboard flow
6. Implement file sending (for agent-created files)
7. Add authorized user ID check (middleware)
8. Update `main.py` to support `--interface telegram`

**Test:**
```
[Open Telegram, message your bot]
/help → Shows command list
/ask What is 2+2? → Agent responds
/ask Read ~/test.txt and summarize → File read, summary returned
```

---

### Phase 8: MCP Integration (Day 16-18)

**Goal:** Connect to Blender MCP server

**Steps:**
1. `pip install mcp`
2. Create `mcp/mcp_types.py` (MCPTool, MCPResult)
3. Create `mcp/transports/stdio_transport.py`
4. Create `mcp/transports/http_transport.py`
5. Create `mcp/mcp_connection.py`
6. Create `mcp/mcp_manager.py`
7. Wire MCPManager into startup: connect servers, register tools
8. Install Blender MCP: `pip install blender-mcp` or `uvx blender-mcp`
9. Test with Blender running

**Test:**
```
[Start Blender, enable MCP server add-on]
[Start NeuralClaw with MCP config enabled]

NeuralClaw> Create a cube in Blender
[Tool: blender::create_mesh → success]
```

---

### Phase 9: Plugin System (Day 18-19)

**Goal:** Drop a file in plugins/ to add a tool

**Steps:**
1. Create `tools/plugins/__init__.py`
2. Create `tools/plugins/plugin_loader.py` (scans directory, imports modules)
3. Create `tools/plugins/example_plugin.py` (sample tool with decorator)
4. Update tool_registry to be accessible globally for plugins
5. Update startup to call `plugin_loader.load_all()`

**Example plugin:**
```python
# tools/plugins/weather_plugin.py

from tools.tool_registry import tool_registry
from tools.types import ToolResult, RiskLevel
import httpx

@tool_registry.register(
    name="get_weather",
    description="Get current weather for a city",
    risk_level=RiskLevel.LOW,
    schema={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"}
        },
        "required": ["city"]
    }
)
async def get_weather(city: str, **ctx) -> ToolResult:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"https://wttr.in/{city}?format=j1")
        data = resp.json()
        temp = data["current_condition"][0]["temp_C"]
        desc = data["current_condition"][0]["weatherDesc"][0]["value"]
        return ToolResult(
            tool_call_id="",
            content=f"Weather in {city}: {desc}, {temp}°C"
        )
```

---

### Phase 10: Polish + Docker + Tests (Day 19-21)

**Goal:** Production-ready MVP

**Steps:**
1. Write comprehensive tests (see Testing Plan section)
2. Create `Dockerfile` and `docker-compose.yml`
3. Add graceful shutdown handling (SIGTERM → cleanup browser, MCP connections)
4. Add health check endpoint (simple HTTP `/health` with FastAPI or raw http.server)
5. Write `README.md` (setup → first Telegram interaction in <30 minutes)
6. Run `pip-audit` to check for vulnerabilities
7. Set up pre-commit hooks (ruff, mypy)

---

## 5. Dev Environment Setup

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev git docker.io docker-compose

# macOS
brew install python@3.11 git docker

# Windows (use WSL2 Ubuntu)
wsl --install
# Then follow Ubuntu instructions inside WSL2
```

### Repository Setup

```bash
# Clone and setup
git clone https://github.com/you/neuralclaw.git
cd neuralclaw

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate  # Windows WSL: same command

# Install all dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install Playwright browser
playwright install chromium

# Install pre-commit hooks
pre-commit install
```

### Secrets Configuration

```bash
# Copy example env file
cp .env.example .env

# Edit with your actual keys
nano .env
```

```bash
# .env file contents

# LLM API Keys (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Telegram (required for Telegram interface)
TELEGRAM_BOT_TOKEN=1234567890:ABC...
TELEGRAM_USER_ID=987654321  # Your numeric Telegram user ID

# Optional: Ollama (for local LLMs)
OLLAMA_BASE_URL=http://localhost:11434

# Optional: Custom search API
SERPAPI_KEY=  # Leave empty to use free DuckDuckGo
```

### Get Your Telegram User ID

```
1. Message @userinfobot on Telegram
2. It replies with your numeric user ID
3. Paste that number as TELEGRAM_USER_ID in .env
```

### Verify Setup

```bash
# Run verification script
python scripts/verify_setup.py

# Expected output:
# ✅ Python 3.11.9
# ✅ OpenAI API key valid
# ✅ Anthropic API key valid  
# ✅ Telegram bot token valid
# ✅ Playwright Chromium installed
# ✅ ChromaDB initialized
# ✅ SQLite database initialized
# ⚠️  Ollama not running (optional)
# ⚠️  Docker not available (optional)
```

### First Run

```bash
# Start with CLI interface (no Telegram setup needed)
python main.py --interface cli

# Start with Telegram
python main.py --interface telegram

# Start with all features enabled
python main.py --interface telegram --enable-mcp --enable-scheduler --log-level DEBUG
```

---

## 6. Dependency List

### requirements.txt (Production)

```txt
# LLM
openai>=1.40.0
anthropic>=0.30.0
ollama>=0.3.0
tiktoken>=0.7.0

# Memory
chromadb>=0.5.0
sentence-transformers>=3.0.0
aiosqlite>=0.20.0

# Browser
playwright>=1.45.0
beautifulsoup4>=4.12.0
lxml>=5.2.0

# HTTP
httpx>=0.27.0
httpx[http2]

# MCP
mcp>=1.0.0

# Telegram
aiogram>=3.10.0

# CLI
aioconsole>=0.7.0
rich>=13.7.0

# Config & Validation
pydantic>=2.7.0
pydantic-settings>=2.3.0
python-dotenv>=1.0.0
pyyaml>=6.0.0

# Scheduling
apscheduler>=4.0.0
sqlalchemy>=2.0.0  # For APScheduler job store

# Safety
jsonschema>=4.23.0
RestrictedPython>=7.0.0

# Observability
structlog>=24.2.0

# Utilities
tenacity>=8.3.0   # Retry logic
python-ulid>=2.2.0  # Better IDs than UUID
anyio>=4.4.0
```

### requirements-dev.txt (Development only)

```txt
# Testing
pytest>=8.2.0
pytest-asyncio>=0.23.0
pytest-mock>=3.14.0
pytest-cov>=5.0.0
respx>=0.21.0  # Mock httpx requests

# Code quality
ruff>=0.5.0
mypy>=1.10.0
pre-commit>=3.7.0

# Security
pip-audit>=2.7.0

# Debugging
ipython>=8.25.0
ipdb>=0.13.13
```

### pyproject.toml

```toml
[project]
name = "neuralclaw"
version = "1.0.0"
description = "Local-first autonomous AI agent platform"
requires-python = ">=3.11"
readme = "README.md"

[tool.ruff]
target-version = "py311"
line-length = 100
select = ["E", "F", "I", "B", "W", "UP"]

[tool.mypy]
python_version = "3.11"
strict = false
ignore_missing_imports = true

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "--cov=. --cov-report=html --cov-report=term-missing"

[tool.coverage.run]
omit = ["tests/*", ".venv/*", "scripts/*"]
```

---

## 7. Repo Structure

```
neuralclaw/
│
├── agent/
│   ├── __init__.py
│   ├── orchestrator.py          # Main agent loop (~200 lines)
│   ├── planner.py               # Task decomposition
│   ├── context_builder.py       # Builds LLM prompt
│   └── session.py               # Session state
│
├── brain/
│   ├── __init__.py
│   ├── llm_client.py            # Abstract base + factory
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── ollama_client.py
│   └── types.py
│
├── memory/
│   ├── __init__.py
│   ├── memory_manager.py        # Unified interface
│   ├── short_term.py            # ConversationBuffer
│   ├── long_term.py             # ChromaDB wrapper
│   ├── episodic.py              # SQLite episodes
│   └── embedder.py
│
├── tools/
│   ├── __init__.py
│   ├── tool_bus.py
│   ├── tool_registry.py
│   ├── types.py
│   ├── browser.py
│   ├── terminal.py
│   ├── filesystem.py
│   ├── search.py
│   ├── python_eval.py
│   └── plugins/
│       ├── __init__.py
│       ├── plugin_loader.py
│       └── example_plugin.py
│
├── mcp/
│   ├── __init__.py
│   ├── mcp_manager.py
│   ├── mcp_connection.py
│   ├── mcp_types.py
│   └── transports/
│       ├── __init__.py
│       ├── stdio_transport.py
│       └── http_transport.py
│
├── safety/
│   ├── __init__.py
│   ├── safety_kernel.py
│   ├── permissions.py
│   ├── whitelist.py
│   └── risk_scorer.py
│
├── interfaces/
│   ├── __init__.py
│   ├── telegram_bot.py
│   └── cli.py
│
├── scheduler/
│   ├── __init__.py
│   └── scheduler.py
│
├── observability/
│   ├── __init__.py
│   ├── logger.py
│   └── event_bus.py
│
├── config/
│   ├── __init__.py
│   ├── settings.py              # Pydantic BaseSettings
│   ├── config.yaml              # Default configuration
│   └── secrets.yaml.example
│
├── data/                        # gitignored - runtime data
│   ├── chroma/
│   ├── sqlite/
│   ├── logs/
│   └── agent_files/
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── unit/
│   │   ├── test_safety_kernel.py
│   │   ├── test_tool_registry.py
│   │   ├── test_memory_manager.py
│   │   ├── test_context_builder.py
│   │   └── test_llm_clients.py
│   ├── integration/
│   │   ├── test_tool_bus_integration.py
│   │   ├── test_memory_integration.py
│   │   └── test_agent_loop.py
│   └── e2e/
│       ├── test_cli_e2e.py
│       └── test_telegram_e2e.py
│
├── scripts/
│   ├── verify_setup.py          # Setup verification
│   ├── seed_memory.py           # Seed test memories
│   └── cleanup_data.py         # Clean runtime data
│
├── docs/
│   ├── Architecture.md
│   ├── PRD.md
│   ├── System_Design.md
│   └── Mvp_Tech.md
│
├── .env.example
├── .gitignore
├── .pre-commit-config.yaml
├── pyproject.toml
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── main.py                      # Entry point (~80 lines)
```

### `.gitignore`

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
.venv/
*.egg-info/
dist/
build/

# Secrets
.env
config/secrets.yaml
*.key

# Runtime data
data/
*.db
*.sqlite

# Playwright
playwright-report/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

---

## 8. Timeline to Build MVP

### Week 1: Foundation + Brain + Tools

| Day | Task | Hours |
|---|---|---|
| 1 | Project setup, config, logging | 3h |
| 2-3 | LLM brain (OpenAI + Anthropic) | 5h |
| 4-5 | Tool system: registry, bus, safety kernel | 5h |
| 5-6 | Core tools: filesystem, terminal, search | 4h |
| 7 | Tests for brain + tools | 3h |

**Week 1 Deliverable:** Agent works end-to-end via Python REPL (no UI yet)

---

### Week 2: Memory + CLI + Browser

| Day | Task | Hours |
|---|---|---|
| 8 | Memory: short-term + ChromaDB | 4h |
| 9 | Memory: episodic (SQLite) | 3h |
| 10-11 | CLI interface (REPL) | 3h |
| 11-12 | Browser tool (Playwright) | 4h |
| 12 | Memory + context builder integration | 3h |
| 13 | Integration tests | 3h |

**Week 2 Deliverable:** Full CLI conversation with memory, browser, tools

---

### Week 3: Telegram + MCP + Polish

| Day | Task | Hours |
|---|---|---|
| 14-16 | Telegram bot interface | 5h |
| 16-17 | MCP client (stdio + Blender) | 4h |
| 18 | Plugin system | 2h |
| 19 | Docker setup | 2h |
| 20 | Full test suite | 4h |
| 21 | README + documentation | 3h |

**Week 3 Deliverable:** Complete MVP — all requirements met

---

### Stretch Goals (Week 4, if ahead of schedule)

| Task | Estimated Hours |
|---|---|
| Autonomous mode + task planner | 8h |
| APScheduler integration | 4h |
| Web search tool enhancement | 2h |
| ollama_client.py implementation | 3h |
| Memory search Telegram command | 2h |

---

## 9. What NOT to Build Initially

### Skip These Entirely for MVP

**Web Dashboard**
Not needed. Telegram provides a rich enough interface. Adding a web UI requires FastAPI, React, a build pipeline, and security configuration — multiply effort by 3x for no immediate value.

**Multi-Agent System**
Complex coordination, shared memory schemas, agent-to-agent protocols. Solve for one agent first. Multi-agent can be added as a module without rearchitecting anything.

**Docker Sandbox for Terminal**
The whitelist-based safety is sufficient for MVP. Docker sandbox is more secure but adds complexity (docker socket access, container management, slower execution). Add in v1.1.

**Voice Interface**
Whisper transcription + TTS adds Telegram voice note handling, audio processing, and latency. Post-MVP.

**Advanced Re-planning**
When a plan step fails, MVP just reports the error and stops. Sophisticated re-planning (backtracking, alternative step generation) is a research problem — defer.

**Custom Embedding API**
Local sentence-transformers is sufficient. OpenAI embedding API would send memory data externally. Use local until forced otherwise.

**APScheduler Persistent Jobs**
For MVP, scheduled tasks can be in-memory (lost on restart). Persistence is a v1.1 quality-of-life improvement.

**Streaming LLM Responses**
Polling (send full response when complete) is simpler and works well enough for Telegram. Streaming requires real-time message editing and complicates the architecture.

**SSE MCP Transport**
stdio (local process) covers Blender MCP. HTTP covers local HTTP servers. SSE is for remote MCP servers — not needed for MVP.

---

## 10. Testing Plan

### Test Strategy

```
Unit Tests (70% of tests)
  Test each component in isolation
  Mock all external dependencies (LLM API, filesystem, network)
  Fast (< 100ms per test)
  Cover: edge cases, error handling, safety rules

Integration Tests (20% of tests)
  Test component interactions (orchestrator + memory + tool_bus)
  Use real ChromaDB (test collection), real SQLite (temp file)
  Mock: LLM API, external HTTP, browser
  Cover: end-to-end flows within the system

E2E Tests (10% of tests)
  Full conversation flow (CLI only, no Telegram in CI)
  Real tool execution against test fixtures
  Real LLM calls (optional, guarded by env flag)
  Cover: happy paths for each major use case
```

### Critical Test Cases

```python
# tests/unit/test_safety_kernel.py

class TestSafetyKernel:
    
    def test_rm_rf_is_blocked(self, safety_kernel, low_trust_session):
        tool_call = ToolCall(name="terminal_exec", arguments={"command": "rm -rf /"})
        decision = asyncio.run(safety_kernel.evaluate(tool_call, low_trust_session))
        assert decision.status == SafetyStatus.BLOCKED
        assert "blocked pattern" in decision.reason.lower()
    
    def test_ls_is_approved(self, safety_kernel, low_trust_session):
        tool_call = ToolCall(name="terminal_exec", arguments={"command": "ls -la"})
        decision = asyncio.run(safety_kernel.evaluate(tool_call, low_trust_session))
        assert decision.status == SafetyStatus.APPROVED
    
    def test_high_risk_requires_confirmation(self, safety_kernel, low_trust_session):
        tool_call = ToolCall(name="terminal_exec", arguments={"command": "git push origin main"})
        decision = asyncio.run(safety_kernel.evaluate(tool_call, low_trust_session))
        assert decision.status == SafetyStatus.CONFIRM_NEEDED
    
    def test_file_outside_allowed_path_blocked(self, safety_kernel, low_trust_session):
        tool_call = ToolCall(name="file_read", arguments={"path": "/etc/passwd"})
        decision = asyncio.run(safety_kernel.evaluate(tool_call, low_trust_session))
        assert decision.status == SafetyStatus.BLOCKED
    
    def test_high_trust_auto_approves_medium_risk(self, safety_kernel, high_trust_session):
        tool_call = ToolCall(name="file_write", arguments={
            "path": "~/agent_files/test.txt",
            "content": "hello"
        })
        decision = asyncio.run(safety_kernel.evaluate(tool_call, high_trust_session))
        assert decision.status == SafetyStatus.APPROVED


# tests/unit/test_memory_manager.py

class TestMemoryManager:
    
    async def test_store_and_retrieve(self, memory_manager):
        await memory_manager.store("Python is great for async programming")
        results = await memory_manager.search("async python")
        assert len(results) >= 1
        assert "python" in results[0].content.lower()
    
    async def test_short_term_persists_in_session(self, memory_manager, session):
        session.short_term.add(
            Message(role=USER, content="What is asyncio?"),
            Message(role=ASSISTANT, content="asyncio is Python's async library")
        )
        history = session.short_term.get_all()
        assert len(history) == 2
        assert "asyncio" in history[1].content


# tests/integration/test_agent_loop.py

class TestAgentLoop:
    
    async def test_simple_question_no_tools(self, orchestrator, session, mock_llm):
        mock_llm.returns("Hello! I'm NeuralClaw, your AI assistant.")
        
        response = await orchestrator.process_message("Hello", session)
        
        assert response.content == "Hello! I'm NeuralClaw, your AI assistant."
        assert len(session.short_term.get_all()) == 2  # user + assistant
    
    async def test_tool_call_flow(self, orchestrator, session, mock_llm, tmp_path):
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello from test file")
        
        # LLM first returns tool call, then returns text after result
        mock_llm.returns_sequence([
            LLMResponse(tool_calls=[ToolCall(name="file_read", arguments={"path": str(test_file)})]),
            LLMResponse(content="The file contains: Hello from test file")
        ])
        
        response = await orchestrator.process_message(f"Read {test_file}", session)
        
        assert "Hello from test file" in response.content
    
    async def test_safety_block_propagates_to_response(self, orchestrator, session, mock_llm):
        mock_llm.returns(LLMResponse(tool_calls=[
            ToolCall(name="terminal_exec", arguments={"command": "rm -rf /"})
        ]))
        
        response = await orchestrator.process_message("Delete everything", session)
        
        assert "blocked" in response.content.lower() or "security" in response.content.lower()
```

### Test Fixtures (conftest.py)

```python
# tests/conftest.py

import pytest
import asyncio
from unittest.mock import AsyncMock
from agent.session import Session
from safety.safety_kernel import SafetyKernel
from memory.memory_manager import MemoryManager
import tempfile
import os

@pytest.fixture
def low_trust_session():
    session = Session(session_id="test_session", user_id="test_user")
    session.state.trust_level = TrustLevel.LOW
    return session

@pytest.fixture
def high_trust_session():
    session = Session(session_id="test_session_high", user_id="test_user")
    session.state.trust_level = TrustLevel.HIGH
    return session

@pytest.fixture
def safety_kernel(tmp_path):
    config = SafetyConfig(
        allowed_paths=[str(tmp_path)],
        terminal_whitelist=["ls", "cat", "echo", "git"]
    )
    return SafetyKernel(config=config)

@pytest.fixture
async def memory_manager(tmp_path):
    manager = MemoryManager(
        chroma_dir=str(tmp_path / "chroma"),
        sqlite_path=str(tmp_path / "test.db")
    )
    await manager.init()
    return manager

@pytest.fixture
def mock_llm():
    """Creates a mockable LLM client."""
    mock = AsyncMock()
    mock.generate = AsyncMock()
    return mock
```

### Test Run Commands

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=. --cov-report=html
open htmlcov/index.html

# Run only unit tests (fast)
pytest tests/unit/ -v

# Run with real LLM (set REAL_LLM_TESTS=1)
REAL_LLM_TESTS=1 pytest tests/integration/ -v

# Run specific test
pytest tests/unit/test_safety_kernel.py::TestSafetyKernel::test_rm_rf_is_blocked -v
```

---

## 11. Configuration Reference

### Minimal config.yaml (to get started fast)

```yaml
# Minimum viable configuration
agent:
  name: "NeuralClaw"

llm:
  default_provider: "openai"
  default_model: "gpt-4o"

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/episodes.db"

tools:
  filesystem:
    allowed_paths:
      - "~/agent_files"
  terminal:
    working_dir: "~/agent_files"

telegram:
  # Bot token and user IDs come from .env

mcp:
  servers: {}  # Add Blender when ready
```

---

## 12. First Run Checklist

```
□ Python 3.11+ installed
□ Virtual environment created and activated
□ pip install -r requirements.txt completed without errors
□ playwright install chromium completed
□ .env file created with:
  □ OPENAI_API_KEY (or ANTHROPIC_API_KEY)
  □ TELEGRAM_BOT_TOKEN
  □ TELEGRAM_USER_ID
□ python scripts/verify_setup.py passes
□ python main.py --interface cli starts without errors
□ CLI: type "Hello, who are you?" → agent responds
□ CLI: type "list files in ~/agent_files" → terminal tool executes
□ CLI: type "browse https://example.com" → browser tool returns content
□ Telegram: /start → bot responds
□ Telegram: /ask What day is it? → agent responds
□ Telegram: /ask Read file ~/agent_files/test.txt → (if exists) reads it
□ Telegram: /help → shows command menu
```

---

*End of Mvp_Tech.md*
