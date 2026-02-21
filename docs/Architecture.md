# Architecture.md — NeuralClaw Autonomous Agent Platform

> **Version:** 1.0.0  
> **Status:** Design Specification  
> **Audience:** Solo developer / Senior engineer  
> **Last Updated:** 2025

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Design Philosophy](#2-design-philosophy)
3. [High-Level Architecture](#3-high-level-architecture)
4. [Module Breakdown](#4-module-breakdown)
5. [Agent Loop Design](#5-agent-loop-design)
6. [Tool Routing System](#6-tool-routing-system)
7. [Memory System Architecture](#7-memory-system-architecture)
8. [MCP Integration Architecture](#8-mcp-integration-architecture)
9. [Data Flow Diagrams](#9-data-flow-diagrams)
10. [Component Responsibilities](#10-component-responsibilities)
11. [Folder Structure](#11-folder-structure)
12. [Security Layers](#12-security-layers)
13. [Event Flow](#13-event-flow)
14. [Autonomy Loop Design](#14-autonomy-loop-design)
15. [Interface Adapters](#15-interface-adapters)
16. [Observability & Logging](#16-observability--logging)

---

## 1. System Overview

NeuralClaw is a **local-first, modular, autonomous AI agent platform** designed for a technical solo developer. It combines a powerful LLM brain with persistent memory, secure tool execution, MCP (Model Context Protocol) connectivity, and a Telegram-first control interface.

The platform is designed around the principle of **progressive autonomy**: the agent can operate interactively under human supervision, or fully autonomously on long-running tasks, with human-confirmation gates inserted at critical decision points.

### Core Identity

```
NeuralClaw = LLM Brain + Memory Graph + Tool Bus + Interface Layer + MCP Client + Safety Kernel
```

The system is LLM-agnostic at its core — OpenAI, Anthropic, Ollama, or any OpenAI-compatible endpoint can be plugged in via a unified adapter. The agent's decision-making is separated from its execution environment so that tool risk is always mediated through a safety kernel.

---

## 2. Design Philosophy

| Principle | Implementation |
|---|---|
| **Local-first** | All core state, memory, and tool execution run locally. Cloud LLMs are optional. |
| **Secure by default** | Whitelist-based tool execution. Sandboxed terminal. Confirmation gates for destructive ops. |
| **Modular** | Every component is a standalone Python module with a clear interface. Plug new tools in without touching core. |
| **Observable** | Every agent action, tool call, memory write, and LLM prompt is logged with structured JSON. |
| **MCP-native** | First-class support for MCP servers — agent discovers, connects, and calls external tools via MCP protocol. |
| **LLM-agnostic** | Brain module wraps any LLM behind a unified `generate(messages, tools) -> response` interface. |
| **Telegram-first** | Primary human control is through Telegram. CLI is secondary. Web dashboard is future. |
| **Extensible** | Tool plugin system. Drop a Python file into `tools/plugins/` to register new capabilities. |

---

## 3. High-Level Architecture

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                         NEURALCLAW AGENT PLATFORM                               ║
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                        INTERFACE LAYER                                  │    ║
║  │  ┌─────────────┐   ┌─────────────┐   ┌──────────────────────────────┐  │    ║
║  │  │  Telegram   │   │    CLI      │   │  Web Dashboard (future)      │  │    ║
║  │  │  Bot        │   │  Interface  │   │  REST API (future)           │  │    ║
║  │  └──────┬──────┘   └──────┬──────┘   └─────────────────────────────-┘  │    ║
║  └─────────┼────────────────-┼───────────────────────────────────────────-─┘    ║
║            │                 │                                                   ║
║            ▼                 ▼                                                   ║
║  ┌─────────────────────────────────────────────────────────────────────────┐    ║
║  │                      AGENT ORCHESTRATOR                                 │    ║
║  │                                                                         │    ║
║  │   Input → Context Build → LLM Call → Tool Dispatch → Response          │    ║
║  │                                                                         │    ║
║  │   ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────┐  │    ║
║  │   │  Planner     │  │  Reasoner    │  │  Response Synthesizer       │  │    ║
║  │   │  (task decomp)│  │  (chain-of- │  │  (formats output for UI)   │  │    ║
║  │   └──────────────┘  │   thought)  │  └─────────────────────────────┘  │    ║
║  │                     └──────────────┘                                   │    ║
║  └─────────────────────────┬───────────────────────────────────────────-──┘    ║
║                            │                                                   ║
║         ┌──────────────────┼──────────────────┐                                ║
║         ▼                  ▼                  ▼                                 ║
║  ┌─────────────┐   ┌──────────────┐   ┌─────────────────────────────────┐     ║
║  │  LLM BRAIN  │   │   MEMORY     │   │        TOOL BUS                 │     ║
║  │             │   │   ENGINE     │   │                                  │     ║
║  │  OpenAI     │   │  Short-term  │   │  ┌──────────┐  ┌─────────────┐  │     ║
║  │  Anthropic  │   │  Long-term   │   │  │ Browser  │  │  Terminal   │  │     ║
║  │  Ollama     │   │  Vector DB   │   │  │ (PW)     │  │  Sandbox    │  │     ║
║  │  Any API    │   │  SQLite log  │   │  └──────────┘  └─────────────┘  │     ║
║  └─────────────┘   └──────────────┘   │  ┌──────────┐  ┌─────────────┐  │     ║
║                                       │  │ File     │  │  MCP Client │  │     ║
║                                       │  │ System   │  │             │  │     ║
║                                       │  └──────────┘  └─────────────┘  │     ║
║                                       │  ┌──────────┐  ┌─────────────┐  │     ║
║                                       │  │ Plugin   │  │  Custom     │  │     ║
║                                       │  │ Tools    │  │  Tools      │  │     ║
║                                       │  └──────────┘  └─────────────┘  │     ║
║                                       └─────────────────────────────────┘     ║
║                                                      │                         ║
║  ┌─────────────────────────────────────────────────-─┘                         ║
║  │                    MCP LAYER                                                 ║
║  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                      ║
║  │  │  Blender MCP │  │  Custom MCP  │  │  Any MCP     │                      ║
║  │  │  Server      │  │  Servers     │  │  Server      │                      ║
║  │  └──────────────┘  └──────────────┘  └──────────────┘                      ║
║  └──────────────────────────────────────────────────────                        ║
║                                                                                 ║
║  ┌───────────────────────────────────────────────────────────────────────────┐  ║
║  │                         SAFETY KERNEL                                     │  ║
║  │   Permission Check → Risk Score → Whitelist → Confirm Gate → Audit Log   │  ║
║  └───────────────────────────────────────────────────────────────────────────┘  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. Module Breakdown

### 4.1 Core Modules

#### `agent/orchestrator.py` — Agent Orchestrator
The central coordinator. Receives user input from any interface, builds context, invokes the LLM brain, dispatches tool calls, collects results, and assembles the final response. Manages the agent's execution loop iteration.

**Responsibilities:**
- Maintain active session state
- Build the prompt context from memory + conversation history + system prompt
- Route tool call requests from LLM to the Tool Bus
- Handle multi-step reasoning chains
- Emit structured events for observability

#### `agent/planner.py` — Task Planner
Responsible for decomposing complex high-level tasks into sequences of tool calls and sub-tasks. Used for autonomous mode where the agent operates over multiple steps.

**Responsibilities:**
- Accept a high-level goal
- Produce an ordered plan as a DAG (directed acyclic graph) of steps
- Re-plan dynamically when steps fail
- Track plan execution state

#### `agent/reasoner.py` — Reasoning Engine
Wraps chain-of-thought and self-critique logic. Before executing a plan step, the reasoner can be asked to evaluate the correctness of the intended action.

---

### 4.2 Brain Module

#### `brain/llm_client.py` — LLM Client (Brain)
Abstraction layer over all LLM providers. Exposes a single `generate(messages, tools, config) -> LLMResponse` interface.

**Supported Backends:**
- OpenAI (GPT-4o, GPT-4-turbo)
- Anthropic (Claude 3.5 Sonnet, Claude 3 Opus)
- Ollama (local models: llama3, mixtral, etc.)
- Any OpenAI-compatible endpoint

**Key Classes:**
```python
class LLMClient(ABC):
    async def generate(messages: List[Message], tools: List[ToolSchema], config: LLMConfig) -> LLMResponse

class OpenAIClient(LLMClient): ...
class AnthropicClient(LLMClient): ...
class OllamaClient(LLMClient): ...
```

---

### 4.3 Memory Module

#### `memory/short_term.py` — Short-Term Memory
In-process conversation buffer. Stores the last N turns of dialogue. Automatically summarized when context window approaches limit.

#### `memory/long_term.py` — Long-Term Vector Memory
Persistent vector database for semantic search over past interactions, tool results, and agent knowledge. Uses ChromaDB locally.

#### `memory/episodic.py` — Episodic Memory
Stores structured records of completed tasks, decisions made, outcomes observed. SQLite-backed with rich metadata.

#### `memory/memory_manager.py` — Memory Manager
Unified interface over all memory types. Handles read (retrieve relevant context), write (store new information), and compression (summarize old turns).

---

### 4.4 Tool Bus

#### `tools/tool_bus.py` — Tool Bus
Central dispatcher for all tool calls. Receives tool invocation requests from the LLM brain, routes them to the appropriate handler, passes requests through the Safety Kernel, and returns results.

**Core Tools:**
- `tools/browser.py` — Playwright browser automation
- `tools/terminal.py` — Sandboxed terminal execution
- `tools/filesystem.py` — File read/write with path restrictions
- `tools/mcp_client.py` — MCP server connectivity
- `tools/search.py` — Web search integration
- `tools/plugins/` — User-defined tool plugins

---

### 4.5 MCP Module

#### `mcp/mcp_manager.py` — MCP Manager
Manages connections to multiple MCP servers. Handles server discovery, tool listing, and proxying tool calls. Integrates discovered MCP tools into the agent's tool registry automatically.

#### `mcp/mcp_connection.py` — MCP Connection
Handles a single MCP server connection (stdio or HTTP/SSE transport). Implements tool discovery, call, and result parsing per MCP spec.

---

### 4.6 Interface Adapters

#### `interfaces/telegram_bot.py` — Telegram Bot
Primary human-agent interface. Receives commands, sends responses, handles file uploads, manages conversation sessions, and presents confirmation dialogs via inline keyboards.

#### `interfaces/cli.py` — CLI Interface
Secondary interface for local development and scripting. Interactive REPL loop.

---

### 4.7 Safety Kernel

#### `safety/safety_kernel.py` — Safety Kernel
Intercepts all tool execution requests before they reach the tool handler. Evaluates risk, checks permissions, applies whitelists, and either approves, blocks, or escalates for human confirmation.

---

## 5. Agent Loop Design

The agent operates in an **observe → think → act → reflect** loop:

```
                    ┌──────────────────────────────────────┐
                    │           AGENT LOOP                  │
                    └──────────────────────────────────────┘
                                    │
              User Input / Scheduled Task / Event Trigger
                                    │
                                    ▼
                    ┌──────────────────────────────────────┐
                    │   1. OBSERVE (Input Processing)      │
                    │   - Parse user message               │
                    │   - Identify intent                  │
                    │   - Extract parameters               │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │   2. CONTEXT BUILD                   │
                    │   - Retrieve short-term history      │
                    │   - Semantic search long-term memory │
                    │   - Load active plan (if any)        │
                    │   - Build system prompt              │
                    │   - Attach available tool schemas    │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │   3. THINK (LLM Call)                │
                    │   - Send context + tools to LLM      │
                    │   - Receive: text response OR        │
                    │     tool_call(s) OR both             │
                    └──────────────────┬───────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
              ┌──────────────────┐    ┌──────────────────────┐
              │ Text Response    │    │ Tool Call(s)         │
              │ → Format output  │    │ → Safety Kernel check│
              │ → Send to UI     │    │ → Execute tool       │
              └──────────────────┘    │ → Collect result     │
                                      └──────────┬───────────┘
                                                 │
                                                 ▼
                    ┌──────────────────────────────────────┐
                    │   4. ACT (Tool Execution)            │
                    │   - Run tool in appropriate sandbox  │
                    │   - Capture output / errors          │
                    │   - Write result to memory           │
                    │   - Return result to LLM             │
                    └──────────────────┬───────────────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │   5. REFLECT (Result Integration)    │
                    │   - LLM processes tool result        │
                    │   - Decides: done OR more tool calls │
                    │   - Generates final response         │
                    │   - Stores episode to episodic mem   │
                    └──────────────────┬───────────────────┘
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
                 More Tool Calls            Final Response
                 (loop back to 3)           → Send to UI
```

### Loop Constraints
- **Max iterations per turn:** 10 (configurable)
- **Max execution time per turn:** 5 minutes (configurable)
- **Context window management:** Automatic summarization when >80% full
- **Error recovery:** 3 retry attempts with exponential backoff per tool

---

## 6. Tool Routing System

```
LLM Response (tool_call)
        │
        ▼
┌───────────────────┐
│   Tool Bus        │
│   .dispatch(call) │
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────────────┐
│   Tool Registry Lookup                    │
│   - Is tool_name registered?             │
│   - Get ToolHandler instance             │
│   - Get ToolSchema for validation        │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│   Parameter Validation                    │
│   - JSON schema validation               │
│   - Type coercion                        │
│   - Required field check                 │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│   SAFETY KERNEL                           │
│   - Permission level check               │
│   - Command whitelist (terminal tool)    │
│   - Path restriction (filesystem tool)   │
│   - Risk score calculation               │
│   - If HIGH risk → Confirmation request  │
└────────────────────┬──────────────────────┘
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  APPROVED                REQUIRES CONFIRMATION
  │                            │
  │                       Send confirm
  │                       request to UI
  │                            │
  │                       User approves/denies
  │                            │
  │                       If approved → continue
  │                       If denied → return error
  │                            │
  └──────────┬─────────────────┘
             │
             ▼
┌───────────────────────────────────────────┐
│   Tool Handler Execution                  │
│   - Async execution                      │
│   - Timeout enforcement                  │
│   - Error capture                        │
│   - Resource cleanup                     │
└────────────────────┬──────────────────────┘
                     │
                     ▼
┌───────────────────────────────────────────┐
│   Result Processing                       │
│   - Normalize result format              │
│   - Truncate oversized outputs           │
│   - Write to tool result memory          │
│   - Emit audit log event                 │
└────────────────────┬──────────────────────┘
                     │
                     ▼
            Return ToolResult to Orchestrator
```

### Tool Registration

Tools self-register using a decorator pattern:

```python
@tool_registry.register(
    name="browser_navigate",
    description="Navigate browser to a URL and return page content",
    risk_level=RiskLevel.LOW,
    requires_confirmation=False,
    schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to navigate to"},
            "wait_for": {"type": "string", "description": "CSS selector to wait for"}
        },
        "required": ["url"]
    }
)
async def browser_navigate(url: str, wait_for: str = None) -> ToolResult:
    ...
```

### Tool Categories and Risk Levels

| Category | Tools | Risk Level | Confirm Required |
|---|---|---|---|
| READ | browser_navigate, file_read, search | LOW | Never |
| COMPUTE | python_eval, data_analyze | MEDIUM | Never |
| WRITE | file_write, browser_fill | MEDIUM | Optional |
| SYSTEM | terminal_exec | HIGH | Always |
| DESTRUCTIVE | file_delete, terminal_rm | CRITICAL | Always |
| EXTERNAL | mcp_call, api_request | MEDIUM-HIGH | Configurable |

---

## 7. Memory System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MEMORY ENGINE                                 │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  SHORT-TERM MEMORY (In-Process)                           │  │
│  │  - ConversationBuffer: last 20 messages                  │  │
│  │  - SessionState: current task, active plan               │  │
│  │  - ToolResultCache: last 10 tool results                 │  │
│  │  - Auto-compresses at 80% context window capacity        │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  LONG-TERM VECTOR MEMORY (ChromaDB)                       │  │
│  │  Collections:                                             │  │
│  │   - "conversations"  : past dialogue summaries           │  │
│  │   - "knowledge"      : learned facts and context         │  │
│  │   - "tool_results"   : significant tool outputs          │  │
│  │   - "plans"          : completed/failed plan records     │  │
│  │  Search: semantic cosine similarity                       │  │
│  │  Embedding: local (sentence-transformers) or API         │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  EPISODIC MEMORY (SQLite)                                  │  │
│  │  Tables:                                                  │  │
│  │   - episodes: task_id, goal, steps, outcome, timestamp   │  │
│  │   - tool_calls: tool, params, result, duration, session  │  │
│  │   - reflections: lesson_learned, context, timestamp      │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  STRUCTURED KV STORE (JSON files / SQLite)                │  │
│  │  - Agent configuration                                   │  │
│  │  - User preferences                                      │  │
│  │  - MCP server configs                                    │  │
│  │  - Tool whitelist/permission settings                    │  │
│  └────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Memory Read Flow (Context Building)

```python
async def build_context(session_id, user_message) -> Context:
    # 1. Get recent conversation (short-term)
    recent = short_term.get_recent(n=10)
    
    # 2. Semantic search for relevant past context (long-term)
    relevant = await long_term.search(query=user_message, n=5)
    
    # 3. Get relevant past tool results
    tool_history = await long_term.search_collection("tool_results", query=user_message, n=3)
    
    # 4. Get current plan if active
    plan = session_state.get_active_plan()
    
    # 5. Compose context
    return Context(
        system_prompt=build_system_prompt(),
        conversation=recent,
        relevant_memories=relevant,
        tool_history=tool_history,
        active_plan=plan,
        available_tools=tool_registry.get_schemas()
    )
```

### Memory Write Flow

```python
async def commit_to_memory(episode: Episode):
    # 1. Update short-term conversation buffer
    short_term.add(episode.user_msg, episode.assistant_msg)
    
    # 2. Embed and store in long-term vector DB
    if episode.is_significant:
        await long_term.store(episode.summary, metadata=episode.metadata)
    
    # 3. Store tool results if noteworthy
    for tool_result in episode.tool_results:
        if tool_result.significance > THRESHOLD:
            await long_term.store_in_collection("tool_results", tool_result)
    
    # 4. Write full episode to SQLite
    await episodic.record(episode)
```

---

## 8. MCP Integration Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     MCP MANAGER                                  │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Server Registry                                        │   │
│   │  - blender_mcp:  {transport: stdio, cmd: blender_mcp}  │   │
│   │  - custom_mcp_1: {transport: http, url: localhost:8811} │   │
│   │  - remote_mcp:   {transport: sse,  url: api.example.com}│   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Tool Discovery (on connect)                           │   │
│   │  - Call server.list_tools()                            │   │
│   │  - Wrap each as a NeuralClaw ToolSchema               │   │
│   │  - Register in global ToolRegistry with prefix        │   │
│   │    e.g., blender::create_mesh, blender::render_scene  │   │
│   └─────────────────────────────────────────────────────────┘   │
│                                                                  │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Execution Flow                                        │   │
│   │  LLM calls blender::create_mesh(params)               │   │
│   │  → ToolBus routes to MCPClient("blender")             │   │
│   │  → MCPConnection.call_tool("create_mesh", params)     │   │
│   │  → Receives MCPResult                                 │   │
│   │  → Normalize → Return ToolResult                      │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                     │
         ▼                    ▼                     ▼
┌─────────────────┐  ┌────────────────┐  ┌─────────────────────┐
│  Blender MCP    │  │  Custom MCP    │  │  Remote MCP         │
│  (stdio)        │  │  (HTTP)        │  │  (SSE)              │
│  Local process  │  │  localhost     │  │  External service   │
└─────────────────┘  └────────────────┘  └─────────────────────┘
```

### MCP Transport Support

| Transport | Use Case | Implementation |
|---|---|---|
| `stdio` | Local MCP servers (Blender, local tools) | Subprocess with stdin/stdout |
| `http` | Local HTTP MCP servers | httpx async client |
| `sse` | Remote/streaming MCP servers | httpx SSE client |

### MCP Tool Namespacing

MCP tools are namespaced to avoid collision:
- `blender::create_object`
- `blender::set_material`
- `filesystem_mcp::read_file`
- `custom::my_tool`

---

## 9. Data Flow Diagrams

### 9.1 Complete Request Flow (Telegram → Tool → Response)

```
Telegram User
    │
    │  /ask "Search the web for latest LLM benchmarks and save to file"
    │
    ▼
Telegram Bot Interface
    │  parse_command()
    │  create_session() or resume_session()
    │
    ▼
Agent Orchestrator
    │  build_context(message, session)
    │    ├─ short_term.get_recent(10)
    │    ├─ long_term.search("LLM benchmarks")
    │    └─ tool_registry.get_schemas()
    │
    ▼
LLM Brain (e.g., GPT-4o)
    │  messages=[system, history, user_msg]
    │  tools=[web_search, file_write, ...]
    │
    │  Response: tool_call{web_search, query="LLM benchmarks 2025"}
    │
    ▼
Tool Bus → Safety Kernel
    │  risk_level=LOW → APPROVED
    │
    ▼
web_search tool
    │  → HTTP request to search API
    │  → Returns: [{title, url, snippet}, ...]
    │
    ▼
LLM Brain (tool result added to context)
    │  Response: tool_call{file_write, path="benchmarks.md", content="..."}
    │
    ▼
Tool Bus → Safety Kernel
    │  risk_level=MEDIUM → APPROVED (file write within allowed paths)
    │
    ▼
filesystem tool
    │  → Write file to allowed path
    │  → Return: {success: true, path: "~/agent_files/benchmarks.md"}
    │
    ▼
LLM Brain (final response)
    │  "I've searched for the latest LLM benchmarks and saved them to benchmarks.md"
    │
    ▼
Memory Engine
    │  short_term.add(user_msg, assistant_msg)
    │  long_term.store(episode_summary)
    │  episodic.record(episode)
    │
    ▼
Telegram Bot Interface
    │  format_response(text, attachments)
    │  send_message(user_id, formatted_response)
    │
    ▼
Telegram User receives response
```

### 9.2 Autonomous Task Flow

```
Telegram: /autonomous "Monitor HN front page every hour, summarize new posts"
    │
    ▼
Orchestrator → Planner
    │  decompose_goal(goal)
    │  → Plan:
    │     Step 1: navigate to news.ycombinator.com
    │     Step 2: extract front page posts
    │     Step 3: compare to last run (from memory)
    │     Step 4: identify new posts
    │     Step 5: summarize new posts with LLM
    │     Step 6: send summary to Telegram
    │     Step 7: schedule next run in 1 hour
    │
    ▼
Plan Executor
    │  execute step 1 → browser_navigate(url)
    │  execute step 2 → browser_extract(selector)
    │  execute step 3 → long_term.search("HN posts")
    │  ... continues through plan
    │
    ▼
Scheduler
    │  register_task(plan, interval=3600s)
    │  → runs in background
    │  → notifies user on each execution
```

---

## 10. Component Responsibilities

| Component | Primary Responsibility | Secondary Responsibility |
|---|---|---|
| **Orchestrator** | Coordinate agent loop iterations | Session management |
| **Planner** | Decompose goals into executable steps | Re-planning on failure |
| **LLM Brain** | Generate responses and tool calls | Reasoning, summarization |
| **Memory Manager** | Read/write agent memory | Context compression |
| **Tool Bus** | Route tool calls to handlers | Result normalization |
| **Safety Kernel** | Enforce permissions and risk gates | Audit trail |
| **MCP Manager** | Manage MCP server connections | Tool discovery |
| **Telegram Bot** | Handle user I/O | Session tracking |
| **Scheduler** | Run scheduled/recurring tasks | Task queue management |
| **Logger** | Structured event logging | Metrics collection |

---

## 11. Folder Structure

```
neuralclaw/
│
├── agent/                          # Core agent logic
│   ├── __init__.py
│   ├── orchestrator.py             # Main agent loop coordinator
│   ├── planner.py                  # Task planning and decomposition
│   ├── reasoner.py                 # Chain-of-thought reasoning
│   ├── context_builder.py          # Builds LLM prompt context
│   ├── response_synthesizer.py     # Formats agent responses
│   └── session.py                  # Session state management
│
├── brain/                          # LLM integration
│   ├── __init__.py
│   ├── llm_client.py               # Abstract base + factory
│   ├── openai_client.py            # OpenAI adapter
│   ├── anthropic_client.py         # Anthropic adapter
│   ├── ollama_client.py            # Ollama adapter
│   └── types.py                    # Message, Response, Config types
│
├── memory/                         # Memory system
│   ├── __init__.py
│   ├── memory_manager.py           # Unified memory interface
│   ├── short_term.py               # In-process conversation buffer
│   ├── long_term.py                # ChromaDB vector store
│   ├── episodic.py                 # SQLite episode store
│   └── embedder.py                 # Text embedding utility
│
├── tools/                          # Tool system
│   ├── __init__.py
│   ├── tool_bus.py                 # Tool dispatcher
│   ├── tool_registry.py            # Tool registration system
│   ├── types.py                    # ToolCall, ToolResult, ToolSchema
│   ├── browser.py                  # Playwright browser tool
│   ├── terminal.py                 # Sandboxed terminal tool
│   ├── filesystem.py               # File system tool
│   ├── search.py                   # Web search tool
│   ├── python_eval.py              # Safe Python execution
│   └── plugins/                    # User-defined tools
│       ├── __init__.py
│       ├── plugin_loader.py        # Auto-discovers plugin files
│       └── example_plugin.py       # Template for new tools
│
├── mcp/                            # MCP protocol support
│   ├── __init__.py
│   ├── mcp_manager.py              # Multi-server manager
│   ├── mcp_connection.py           # Single server connection
│   ├── mcp_types.py                # MCP protocol types
│   └── transports/
│       ├── stdio_transport.py
│       ├── http_transport.py
│       └── sse_transport.py
│
├── safety/                         # Safety and permissions
│   ├── __init__.py
│   ├── safety_kernel.py            # Central safety enforcement
│   ├── permissions.py              # Permission level definitions
│   ├── whitelist.py                # Command/path whitelists
│   ├── risk_scorer.py              # Tool call risk scoring
│   └── sandbox.py                  # Docker/subprocess sandbox
│
├── interfaces/                     # User interfaces
│   ├── __init__.py
│   ├── telegram_bot.py             # Telegram bot interface
│   ├── cli.py                      # CLI REPL interface
│   └── webhooks.py                 # Incoming webhook handler
│
├── scheduler/                      # Task scheduling
│   ├── __init__.py
│   ├── scheduler.py                # APScheduler wrapper
│   └── task_queue.py               # Async task queue
│
├── observability/                  # Logging and monitoring
│   ├── __init__.py
│   ├── logger.py                   # Structured JSON logger
│   ├── event_bus.py                # Internal event emitter
│   └── metrics.py                  # Performance metrics
│
├── config/                         # Configuration management
│   ├── __init__.py
│   ├── settings.py                 # Pydantic settings model
│   ├── config.yaml                 # Default configuration
│   └── secrets.yaml.example        # Secrets template (never commit)
│
├── data/                           # Persistent data (gitignored)
│   ├── chroma/                     # ChromaDB vector storage
│   ├── sqlite/                     # SQLite databases
│   ├── logs/                       # Structured log files
│   └── agent_files/                # Files created by agent
│
├── docker/                         # Docker configuration
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── sandbox/
│       └── Dockerfile.sandbox      # Isolated execution environment
│
├── tests/                          # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── scripts/                        # Utility scripts
│   ├── setup.sh
│   └── install_mcp_servers.sh
│
├── docs/                           # Documentation
│   ├── Architecture.md             # This document
│   ├── PRD.md
│   ├── System_Design.md
│   └── Mvp_Tech.md
│
├── .env.example                    # Environment variable template
├── pyproject.toml                  # Python project config
├── requirements.txt
├── requirements-dev.txt
└── main.py                         # Entry point
```

---

## 12. Security Layers

```
Layer 1: API Key Security
├── All API keys stored in environment variables or secrets.yaml (gitignored)
├── Never logged, never passed through tool results
└── Secrets loaded at startup, never serialized

Layer 2: Input Validation
├── All user inputs sanitized before LLM context
├── All tool parameters validated against JSON schema
├── No direct string interpolation into shell commands
└── Parameter injection prevention

Layer 3: Safety Kernel (Runtime Enforcement)
├── Permission levels: NONE, READ, WRITE, EXECUTE, ADMIN
├── Default level: READ (can be elevated per-command)
├── Command whitelist for terminal tool (configurable list)
├── Path allowlist for filesystem tool (~/agent_files/ by default)
├── Risk scoring: LOW/MEDIUM/HIGH/CRITICAL
└── Human confirmation gate for HIGH+ risk actions

Layer 4: Execution Sandbox
├── Terminal commands run in Docker container (optional)
├── Browser runs in isolated Playwright context
├── Python eval uses restricted builtins (RestrictedPython)
├── Process resource limits (CPU, memory, time)
└── Network access restrictions for sandbox

Layer 5: Audit Logging
├── Every tool call logged: tool, params, result, risk_level, user, timestamp
├── Every LLM call logged: model, input_tokens, output_tokens, duration
├── Every safety decision logged: action, reason, decision
├── Logs are append-only (no modification after write)
└── Log integrity via hash chaining (optional)

Layer 6: MCP Security
├── MCP servers listed explicitly in config (no auto-discovery from network)
├── Each MCP server has an assigned permission level
├── MCP tool calls pass through the same Safety Kernel as local tools
└── MCP server output sanitized before passing to LLM
```

---

## 13. Event Flow

The platform uses an internal event bus for decoupled component communication:

```python
# Event Types
class Events:
    USER_MESSAGE        = "user.message"
    AGENT_RESPONSE      = "agent.response"
    TOOL_CALL_START     = "tool.call.start"
    TOOL_CALL_SUCCESS   = "tool.call.success"
    TOOL_CALL_FAILURE   = "tool.call.failure"
    SAFETY_APPROVED     = "safety.approved"
    SAFETY_BLOCKED      = "safety.blocked"
    SAFETY_CONFIRM_REQ  = "safety.confirm.required"
    LLM_CALL_START      = "llm.call.start"
    LLM_CALL_COMPLETE   = "llm.call.complete"
    MEMORY_WRITE        = "memory.write"
    PLAN_CREATED        = "plan.created"
    PLAN_STEP_COMPLETE  = "plan.step.complete"
    TASK_SCHEDULED      = "task.scheduled"
    MCP_CONNECTED       = "mcp.connected"
    MCP_DISCONNECTED    = "mcp.disconnected"
    ERROR               = "system.error"
```

Every event is:
1. Emitted by the originating component
2. Picked up by the Logger (writes to log file)
3. Picked up by relevant subscribers (e.g., Telegram bot reacts to `SAFETY_CONFIRM_REQ`)

---

## 14. Autonomy Loop Design

The agent supports three modes of operation:

### Interactive Mode (Default)
```
User sends message → Agent responds → User sends next message
One turn at a time. Human in the loop at every step.
```

### Semi-Autonomous Mode
```
User sets a goal → Agent creates plan → Executes each step → 
Pauses for confirmation at HIGH-risk steps → Notifies user of progress
```

### Fully Autonomous Mode
```
User sets goal + trust level → Agent executes complete plan independently →
Notifies user only at completion or on failure
Requires explicit trust elevation: /trust HIGH
```

### Autonomy Loop Pseudocode

```python
async def autonomous_run(goal: str, trust_level: TrustLevel):
    plan = await planner.create_plan(goal)
    await notify_user(f"Plan created: {plan.summary}")
    
    for step in plan.steps:
        # Check if step requires confirmation based on trust level
        if step.risk_level > trust_level.max_auto_risk:
            confirmed = await request_user_confirmation(step)
            if not confirmed:
                await notify_user("Step skipped by user")
                continue
        
        result = await execute_step(step)
        
        if result.failed:
            # Attempt recovery
            recovery = await planner.create_recovery(step, result.error)
            if recovery:
                plan.insert_recovery(recovery)
            else:
                await notify_user(f"Step failed: {result.error}. Stopping.")
                break
        
        await notify_user(f"Step {step.index}/{len(plan.steps)} complete")
    
    await notify_user(f"Task complete: {plan.goal}")
    await memory.commit_episode(plan, results)
```

---

## 15. Interface Adapters

### Telegram Bot Flow

```
Incoming update → Handler router
├── /ask <prompt>          → interactive_handler()
├── /run <goal>            → autonomous_handler()
├── /status                → status_handler()
├── /memory search <query> → memory_handler()
├── /tools list            → tools_handler()
├── /trust <level>         → trust_handler()
├── /confirm <yes|no>      → confirm_handler()
├── /cancel                → cancel_handler()
└── Plain text message     → interactive_handler()
```

### CLI Interface

```
$ neuralclaw
> Welcome to NeuralClaw CLI
> 
> neuralclaw> ask What is the current time?
> neuralclaw> run Monitor my GitHub PRs
> neuralclaw> memory search "python async patterns"
> neuralclaw> tools list
> neuralclaw> exit
```

---

## 16. Observability & Logging

### Log Structure (JSON Lines format)

```json
{
  "timestamp": "2025-01-01T12:00:00.000Z",
  "level": "INFO",
  "event": "tool.call.success",
  "session_id": "sess_abc123",
  "user_id": "telegram_user_456",
  "data": {
    "tool": "browser_navigate",
    "params": {"url": "https://example.com"},
    "result_size_bytes": 45230,
    "duration_ms": 2341,
    "risk_level": "LOW"
  }
}
```

### Log Levels

| Level | Use |
|---|---|
| `DEBUG` | Full LLM prompts, tool params, memory reads |
| `INFO` | Tool calls, responses, memory writes |
| `WARNING` | Safety decisions, retries, degraded performance |
| `ERROR` | Tool failures, LLM errors, connection issues |
| `CRITICAL` | Safety kernel blocks, sandbox violations |

### Metrics Collected

- LLM tokens used per session/day
- Tool call success/failure rates
- Average response time
- Memory hit rate (% of context from long-term memory)
- Safety block rate

---

*End of Architecture.md*
