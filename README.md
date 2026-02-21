# NeuralClaw 🦾

**Local-first autonomous AI agent platform for power developers.**

NeuralClaw is personal AI infrastructure — it runs on your machine, remembers
everything across sessions, executes tools on your behalf (terminal, filesystem,
web search), and can be directed via CLI (and soon Telegram). It is *not* a SaaS
product; all data stays local.

```
███╗   ██╗███████╗██╗   ██╗██████╗  █████╗ ██╗      ██████╗██╗      █████╗ ██╗    ██╗
████╗  ██║██╔════╝██║   ██║██╔══██╗██╔══██╗██║     ██╔════╝██║     ██╔══██╗██║    ██║
██╔██╗ ██║█████╗  ██║   ██║██████╔╝███████║██║     ██║     ██║     ███████║██║ █╗ ██║
██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██╔══██║██║     ██║     ██║     ██╔══██║██║███╗██║
██║ ╚████║███████╗╚██████╔╝██║  ██║██║  ██║███████╗╚██████╗███████╗██║  ██║╚███╔███╔╝
╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝ ╚═════╝╚══════╝╚═╝  ╚═╝ ╚══╝╚══╝
```

---

## Features

| Capability | Status |
|---|---|
| Multi-provider LLM brain (OpenAI, Anthropic, Ollama, OpenRouter, Gemini) | ✅ |
| 3-tier memory (short-term, ChromaDB long-term, SQLite episodic) | ✅ |
| Tool execution: filesystem, terminal, web search | ✅ |
| Safety kernel with risk scoring, whitelist, and trust levels | ✅ |
| Interactive CLI REPL with Rich UI | ✅ |
| Autonomous multi-step task planner (`/run` mode) | ✅ |
| MCP (Model Context Protocol) server integration | 🔧 Phase 6 |
| Telegram bot interface | 🔧 Phase 7 |
| Task scheduler | 🔧 Phase 8 |
| Docker sandbox for terminal | 🔧 Phase 9 |

---

## Quick Start

### 1. Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) (for local models) **or** an API key from OpenAI / Anthropic / etc.

### 2. Clone & install

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

### 3. Configure

```bash
cp .env.example .env
```

Edit `.env` and add your API key(s). At minimum, set one of:
- `OPENAI_API_KEY` — and set `default_llm_provider: openai` in `config/config.yaml`
- `ANTHROPIC_API_KEY` — and set `default_llm_provider: anthropic`
- Leave both empty to use **Ollama** (default) — just have Ollama running locally

### 4. Run

```bash
# Start the CLI (default)
python main.py

# Verbose logging
python main.py --log-level DEBUG

# Use a different config file
python main.py --config /path/to/config.yaml
```

---

## CLI Commands

| Command | Description |
|---|---|
| Just type | Send a message to the agent (same as `/ask`) |
| `/ask <message>` | Send a message and get a response |
| `/run <goal>` | Autonomous multi-step task execution |
| `/status` | Show session stats and current settings |
| `/memory <query>` | Search long-term memory |
| `/tools` | List all registered tools |
| `/trust <low\|medium\|high>` | Set the session trust level |
| `/clear` | Clear conversation history |
| `/cancel` | Cancel the running task |
| `/help` | Show help |
| `exit` / `quit` / Ctrl+D | Exit |

### Trust Levels

| Level | Behavior |
|---|---|
| `low` (default) | Confirm before HIGH and CRITICAL risk actions |
| `medium` | Only confirm CRITICAL risk actions |
| `high` | Auto-approve everything — use with care |

---

## Project Structure

```
neuralclaw/
├── main.py                  # Entry point
├── config/
│   ├── config.yaml          # All non-secret configuration
│   └── settings.py          # Pydantic settings (merges yaml + .env)
├── brain/                   # LLM client abstraction
│   ├── llm_client.py        # BaseLLMClient abstract class
│   ├── types.py             # Message, LLMResponse, ToolCall, etc.
│   ├── openai_client.py
│   ├── anthropic_client.py
│   ├── ollama_client.py
│   ├── openrouter_client.py
│   └── gemini_client.py
├── agent/                   # Core agent loop
│   ├── orchestrator.py      # observe → think → act → reflect loop
│   ├── planner.py           # Goal decomposition for /run mode
│   ├── reasoner.py          # Pre-flight reasoning for risky actions
│   ├── context_builder.py   # Assembles LLM prompt with memory context
│   ├── response_synthesizer.py
│   └── session.py           # Per-session state
├── memory/
│   ├── memory_manager.py    # Unified facade (use this, not the stores)
│   ├── short_term.py        # In-memory sliding window
│   ├── long_term.py         # ChromaDB vector store
│   ├── episodic.py          # SQLite episode + reflection store
│   └── embedder.py          # Sentence-transformer wrapper
├── tools/
│   ├── tool_registry.py     # Self-registration via @registry.register
│   ├── tool_bus.py          # Dispatcher: registry → safety → execution
│   ├── types.py             # ToolCall, ToolResult, ToolSchema, RiskLevel
│   ├── filesystem.py        # file_read, file_write, file_append, list_dir
│   ├── terminal.py          # terminal_exec
│   └── search.py            # web_search
├── safety/
│   ├── safety_kernel.py     # Gatekeeper: APPROVED | BLOCKED | CONFIRM_NEEDED
│   ├── whitelist.py         # Command & path allowlists
│   └── risk_scorer.py       # Heuristic risk level assignment
├── interfaces/
│   └── cli.py               # Rich-powered interactive REPL
├── observability/
│   └── logger.py            # Structured logging (structlog)
└── tests/
    └── unit/                # ~2200 lines of unit tests
```

---

## Configuration Reference

The main config file is `config/config.yaml`. Key sections:

```yaml
# Which LLM to use by default
default_llm_provider: "ollama"   # openai | anthropic | ollama | openrouter | gemini
default_llm_model: "llama3.1"

agent:
  max_iterations_per_turn: 10    # Max tool-call loops per user message
  default_trust_level: "low"

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/sqlite/episodes.db"
  embedding_model: "BAAI/bge-small-en-v1.5"   # Any sentence-transformers model

tools:
  terminal:
    working_dir: "./data/agent_files"
    whitelist_extra: []           # Add extra allowed commands here
  filesystem:
    allowed_paths:
      - "./data/agent_files"      # Agent can only read/write here

safety:
  require_confirmation_for:
    - "HIGH"
    - "CRITICAL"
```

Secrets go in `.env` (copy from `.env.example`). Environment variables always take
priority over `config.yaml`.

---

## Adding a Custom Tool

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

Then import the module in `interfaces/cli.py` so it self-registers:

```python
import tools.my_tool  # noqa: F401
```

---

## Running Tests

```bash
pytest tests/unit/ -v
pytest tests/unit/test_safety.py -v   # Safety kernel only
pytest tests/unit/ --cov=. --cov-report=html
```

---

## Roadmap

- **Phase 6** — MCP integration (connect Blender, Postgres, custom MCP servers)
- **Phase 7** — Full Telegram bot interface (aiogram)
- **Phase 8** — Task scheduler (cron-style recurring tasks)
- **Phase 9** — Docker sandbox for terminal isolation
- **Phase 10** — Prometheus metrics + Grafana dashboard

---

## License

MIT — personal project, use freely.
