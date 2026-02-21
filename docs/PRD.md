# PRD.md — NeuralClaw Autonomous Agent Platform
## Product Requirements Document

> **Version:** 1.0.0  
> **Status:** Active  
> **Owner:** Solo Developer  
> **Classification:** Internal / Personal Project  
> **Last Updated:** 2025

---

## Table of Contents

1. [Product Vision](#1-product-vision)
2. [Target User](#2-target-user)
3. [Use Cases](#3-use-cases)
4. [Functional Requirements](#4-functional-requirements)
5. [Non-Functional Requirements](#5-non-functional-requirements)
6. [Security Requirements](#6-security-requirements)
7. [Performance Expectations](#7-performance-expectations)
8. [UX Flow — Telegram-First Control](#8-ux-flow--telegram-first-control)
9. [Example Commands](#9-example-commands)
10. [Milestones](#10-milestones)
11. [Future Roadmap](#11-future-roadmap)
12. [Risks](#12-risks)
13. [Constraints](#13-constraints)
14. [Acceptance Criteria](#14-acceptance-criteria)

---

## 1. Product Vision

### The Problem

A technically sophisticated solo developer has access to powerful AI models, automation tools, browser control, and shell access. But these capabilities exist as isolated, uncoordinated tools. The developer must context-switch constantly — running commands in terminal, copy-pasting results into ChatGPT, manually browsing for information, re-entering context at every session. There is no memory, no autonomy, no single coherent agent that ties everything together.

### The Solution

**NeuralClaw** is a personal, local-first autonomous AI agent platform. It gives a single developer the capability of a small AI-powered team: it remembers everything, executes tools on your behalf, controls the browser, runs terminal commands, connects to specialized tools via MCP, and can be directed entirely via Telegram from anywhere in the world — even while the agent runs autonomously on a local machine or VPS.

### Vision Statement

> *"A tireless, intelligent digital assistant that lives on your machine, remembers your context, executes your intentions, and can be controlled from your phone — with the safety and transparency of a system you designed yourself."*

### Product Positioning

NeuralClaw is not a SaaS product. It is a **personal infrastructure layer** — like a home server for AI autonomy. It runs locally, stores all data locally, connects to LLMs via API, and is entirely under the developer's control. It is the developer's "AI operating system."

---

## 2. Target User

### Primary Persona: The Power Developer

**Name:** Alex (fictional)  
**Role:** Full-stack developer / AI researcher / indie hacker  
**Technical Level:** Advanced — comfortable with Python, Docker, APIs, shell scripting  
**Working Style:** Builds in long sessions; context-switches between projects; heavy browser user; uses AI assistants daily but frustrated by lack of memory and integration  

**Core Pain Points:**
- Every AI chat session starts from zero — no memory of past work
- Can't automate tasks that require browser + terminal + LLM judgment
- No single interface to delegate complex multi-step tasks
- Doesn't trust cloud tools to handle sensitive local files and commands
- Wants an agent that runs unsupervised while sleeping/working

**What Alex Needs:**
- An agent that remembers everything across sessions
- Telegram control so it can be directed from mobile
- Ability to say "monitor X and notify me when Y happens"
- Ability to say "research topic Z, write a report, save it to disk"
- Blender control for 3D automation
- Full transparency into what the agent is doing and why
- The ability to trust the agent with HIGH-risk actions via explicit permission

---

## 3. Use Cases

### UC-001: Research and Synthesis
**Actor:** Developer  
**Goal:** Research a technical topic, synthesize information from multiple web sources, and produce a structured document  

**Flow:**
1. User sends: `/ask Research the current state of WebGPU adoption and create a summary report`
2. Agent searches the web (multiple queries)
3. Agent visits multiple URLs, extracts content
4. Agent synthesizes into a structured markdown document
5. Agent saves to `~/agent_files/reports/webgpu_report.md`
6. Agent sends summary + file path to Telegram

**Value:** Saves 2-3 hours of manual research and writing

---

### UC-002: Codebase Analysis
**Actor:** Developer  
**Goal:** Analyze a codebase for specific patterns or issues  

**Flow:**
1. User sends: `/ask Scan my ~/projects/my-app directory for any API keys in source files`
2. Agent uses filesystem tool to traverse directory
3. Agent uses terminal to run `grep` with appropriate patterns
4. Agent collects matches, formats results
5. Agent sends report to Telegram

**Value:** Instant codebase auditing without manual scanning

---

### UC-003: Automated Monitoring
**Actor:** Developer (sets task, walks away)  
**Goal:** Continuously monitor a data source and notify on specific conditions  

**Flow:**
1. User sends: `/schedule "Check my GitHub PRs every 30 minutes and notify me of any new review comments"`
2. Agent creates scheduled task
3. Every 30 minutes: agent navigates to GitHub, checks PR status
4. If new comments detected: sends Telegram notification with details
5. Agent persists state between runs via memory

**Value:** Passive monitoring without writing a custom script

---

### UC-004: Blender 3D Automation
**Actor:** Developer / 3D Artist  
**Goal:** Control Blender via natural language  

**Flow:**
1. User sends: `/ask Create a donut mesh in Blender, add a pink torus around it, and render to ~/renders/donut.png`
2. Agent connects to Blender MCP server
3. Agent calls `blender::create_mesh({type: "TORUS"})` multiple times
4. Agent calls `blender::set_material({name: "pink_glaze", color: [1.0, 0.4, 0.6, 1.0]})`
5. Agent calls `blender::render({output_path: "~/renders/donut.png"})`
6. Agent sends rendered image to Telegram

**Value:** No-code Blender automation via natural language

---

### UC-005: Terminal Automation with Oversight
**Actor:** Developer  
**Goal:** Execute a series of system administration tasks safely  

**Flow:**
1. User sends: `/ask Update all pip packages in my virtual environments`
2. Agent plans: find all venvs → for each, run pip list outdated → ask confirmation → run pip upgrade
3. Agent finds venvs via filesystem tool
4. Agent sends confirmation to Telegram: "Found 3 virtual environments. Upgrade all? [Yes] [No]"
5. User confirms
6. Agent executes upgrades, streams output to Telegram

**Value:** Complex system tasks with human oversight at risk decision points

---

### UC-006: Long-Term Project Memory
**Actor:** Developer returning after a break  
**Goal:** Resume a project with full context recall  

**Flow:**
1. User sends: `/ask What was I working on yesterday in the NeuralClaw project?`
2. Agent searches episodic memory for yesterday's sessions
3. Agent retrieves: conversation summaries, files written, tasks completed
4. Agent composes a briefing: "Yesterday you worked on the MCP integration module. You wrote mcp_connection.py and were troubleshooting the stdio transport. You left a TODO about error handling."

**Value:** Perfect project memory without maintaining manual notes

---

### UC-007: Web Automation (Form Filling / Data Extraction)
**Actor:** Developer  
**Goal:** Automate a repetitive web task  

**Flow:**
1. User sends: `/ask Go to my email newsletter, extract all subscriber counts from the last 5 campaigns, and put them in a CSV file`
2. Agent navigates to the site, logs in (credentials from secure config)
3. Agent extracts data from multiple pages
4. Agent writes CSV to filesystem
5. Agent sends summary + file to Telegram

**Value:** Replaces repetitive manual web data collection

---

### UC-008: Multi-Step Code Generation and Testing
**Actor:** Developer  
**Goal:** Generate, execute, and validate code  

**Flow:**
1. User sends: `/ask Write a Python script to convert all PNG files in ~/images to WebP format, then test it on 3 sample images`
2. Agent writes script using LLM
3. Agent saves script to `~/agent_files/scripts/convert_images.py`
4. Agent executes in terminal sandbox on sample images
5. Agent verifies output, reports test results
6. Agent sends script + test results to Telegram

**Value:** Full code → execute → validate loop without manual intervention

---

## 4. Functional Requirements

### FR-001: LLM Brain

| ID | Requirement | Priority |
|---|---|---|
| FR-001.1 | Support OpenAI API (GPT-4o, GPT-4-turbo) as LLM backend | P0 |
| FR-001.2 | Support Anthropic API (Claude 3.5 Sonnet, Claude 3 Opus) | P0 |
| FR-001.3 | Support Ollama (local models: llama3, mixtral, etc.) | P1 |
| FR-001.4 | Support any OpenAI-compatible endpoint | P1 |
| FR-001.5 | Support tool/function calling for all backends | P0 |
| FR-001.6 | Support streaming responses for long outputs | P1 |
| FR-001.7 | Handle context window limits with automatic summarization | P0 |
| FR-001.8 | Support system prompt customization | P0 |
| FR-001.9 | Allow per-session model switching | P2 |

### FR-002: Memory System

| ID | Requirement | Priority |
|---|---|---|
| FR-002.1 | Maintain conversation history within a session | P0 |
| FR-002.2 | Persist memory across sessions (vector DB) | P0 |
| FR-002.3 | Semantic search over past conversations | P0 |
| FR-002.4 | Store and retrieve tool execution results | P1 |
| FR-002.5 | Support episodic memory (task → steps → outcome) | P1 |
| FR-002.6 | Automatic memory summarization when context fills | P0 |
| FR-002.7 | Memory retrieval with relevance scoring | P1 |
| FR-002.8 | Manual memory search via `/memory search <query>` | P1 |
| FR-002.9 | Memory deletion/management via commands | P2 |

### FR-003: Telegram Interface

| ID | Requirement | Priority |
|---|---|---|
| FR-003.1 | Receive text commands via Telegram | P0 |
| FR-003.2 | Send text responses (with markdown formatting) | P0 |
| FR-003.3 | Send files (documents, images) to Telegram | P0 |
| FR-003.4 | Receive files from user (for agent to process) | P1 |
| FR-003.5 | Inline keyboard confirmation dialogs | P0 |
| FR-003.6 | Progress updates for long-running tasks | P0 |
| FR-003.7 | Auth: restrict to specific Telegram user ID(s) | P0 |
| FR-003.8 | Command routing (`/ask`, `/run`, `/status`, etc.) | P0 |
| FR-003.9 | Session persistence per Telegram chat | P0 |
| FR-003.10 | Error notifications with context | P0 |

### FR-004: Browser Automation

| ID | Requirement | Priority |
|---|---|---|
| FR-004.1 | Navigate to URLs | P0 |
| FR-004.2 | Extract page text content | P0 |
| FR-004.3 | Extract structured data (tables, lists) | P0 |
| FR-004.4 | Click elements by selector or text | P1 |
| FR-004.5 | Fill forms and submit | P1 |
| FR-004.6 | Take screenshots | P1 |
| FR-004.7 | Handle JavaScript-heavy SPAs | P1 |
| FR-004.8 | Manage cookies/sessions | P2 |
| FR-004.9 | Download files from browser | P2 |
| FR-004.10 | Run in headless mode by default | P0 |

### FR-005: Terminal Execution

| ID | Requirement | Priority |
|---|---|---|
| FR-005.1 | Execute shell commands (bash) | P0 |
| FR-005.2 | Capture stdout and stderr | P0 |
| FR-005.3 | Enforce command whitelist | P0 |
| FR-005.4 | Set execution timeout | P0 |
| FR-005.5 | Support working directory specification | P0 |
| FR-005.6 | Stream output for long-running commands | P1 |
| FR-005.7 | Optional Docker sandboxing | P1 |
| FR-005.8 | Environment variable injection | P1 |

### FR-006: File System Access

| ID | Requirement | Priority |
|---|---|---|
| FR-006.1 | Read text files | P0 |
| FR-006.2 | Write text files | P0 |
| FR-006.3 | List directory contents | P0 |
| FR-006.4 | Create directories | P0 |
| FR-006.5 | Delete files (HIGH risk, requires confirmation) | P0 |
| FR-006.6 | Move/copy files | P1 |
| FR-006.7 | Search files by name or content | P1 |
| FR-006.8 | Restrict access to configured allowed paths | P0 |

### FR-007: MCP Integration

| ID | Requirement | Priority |
|---|---|---|
| FR-007.1 | Connect to MCP servers via stdio transport | P0 |
| FR-007.2 | Connect to MCP servers via HTTP transport | P0 |
| FR-007.3 | Connect to MCP servers via SSE transport | P1 |
| FR-007.4 | Auto-discover tools from connected MCP servers | P0 |
| FR-007.5 | Register MCP tools in global tool registry | P0 |
| FR-007.6 | Call MCP tools from LLM brain | P0 |
| FR-007.7 | Support Blender MCP server | P0 |
| FR-007.8 | Support multiple simultaneous MCP connections | P1 |
| FR-007.9 | Handle MCP server disconnection gracefully | P1 |

### FR-008: Safety System

| ID | Requirement | Priority |
|---|---|---|
| FR-008.1 | Risk-level scoring for all tool calls | P0 |
| FR-008.2 | Human confirmation for HIGH/CRITICAL risk actions | P0 |
| FR-008.3 | Command whitelist for terminal tool | P0 |
| FR-008.4 | Path allowlist for filesystem tool | P0 |
| FR-008.5 | Permission levels per action type | P0 |
| FR-008.6 | Audit log for all tool calls and safety decisions | P0 |
| FR-008.7 | Trust level setting per session | P1 |
| FR-008.8 | Ability to cancel any in-progress task | P0 |

### FR-009: Plugin System

| ID | Requirement | Priority |
|---|---|---|
| FR-009.1 | Tool auto-discovery from plugins/ directory | P0 |
| FR-009.2 | Plugin registration via decorator | P0 |
| FR-009.3 | Plugins receive same tool infrastructure (safety, logging) | P0 |
| FR-009.4 | Plugin dependency declaration | P1 |
| FR-009.5 | Hot-reload of plugins without restart | P2 |

### FR-010: Task Scheduling

| ID | Requirement | Priority |
|---|---|---|
| FR-010.1 | Schedule tasks with cron-like intervals | P1 |
| FR-010.2 | Run one-time delayed tasks | P1 |
| FR-010.3 | List and cancel scheduled tasks | P1 |
| FR-010.4 | Persist scheduled tasks across restarts | P2 |

---

## 5. Non-Functional Requirements

### NFR-001: Performance
- Response time for simple queries: < 5 seconds (excluding LLM latency)
- Tool call overhead (excluding external operation): < 200ms
- Memory search: < 500ms for collections up to 100,000 vectors
- System startup time: < 10 seconds
- Max concurrent tasks: 5 (configurable)

### NFR-002: Reliability
- Graceful degradation if LLM API is unavailable (queue and retry)
- Tool failure isolation (one tool failure does not crash the agent)
- Automatic session recovery on process restart
- Scheduled tasks survive process restarts
- All state stored persistently — no in-memory-only critical state

### NFR-003: Maintainability
- Every module has clear interfaces and is independently testable
- Code coverage target: 70% for core modules
- All public functions documented with docstrings
- Configuration-driven (no hardcoded behavior)
- Plugin system allows new tools without core modification

### NFR-004: Portability
- Runs on Linux (Ubuntu 22.04+, Debian 12+)
- Runs on Windows 10/11 (via WSL2 recommended)
- Runs on macOS 12+
- Docker deployment supported
- No root/admin required for core functionality

### NFR-005: Observability
- Structured JSON logging for all major events
- Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- Performance metrics collection (optional, local only)
- No telemetry to external services
- All logs stored locally

---

## 6. Security Requirements

### SR-001: Credential Management
- API keys stored ONLY in environment variables or encrypted secrets file
- API keys never logged, never included in tool results, never sent to LLM context
- Telegram bot token stored as secret
- No credentials hardcoded in source

### SR-002: Access Control
- Telegram bot responds ONLY to authorized user IDs (configured whitelist)
- Each interface has a minimum permission level requirement
- Trust elevation requires explicit command (`/trust HIGH`) and is session-scoped

### SR-003: Execution Safety
- Terminal tool enforces command whitelist by default
- Filesystem tool enforces path allowlist by default
- All tool calls pass through Safety Kernel before execution
- Docker sandbox available for high-risk terminal execution

### SR-004: Data Privacy
- All agent data stored locally (no cloud sync)
- Vector embeddings computed locally by default (sentence-transformers)
- Option to use API-based embeddings only with user explicit opt-in
- Memory content never sent anywhere without user intent

### SR-005: Audit Trail
- Immutable append-only audit log
- Every tool call, safety decision, LLM call logged with full context
- Log files rotated by size/date but not deleted automatically
- Safety blocks logged with reason

---

## 7. Performance Expectations

| Metric | Target | Maximum Acceptable |
|---|---|---|
| Simple Q&A response (no tools) | < 3s end-to-end | < 8s |
| Single tool call round trip | < 5s | < 15s |
| Multi-step task (3-5 tool calls) | < 30s | < 90s |
| Memory search (100k vectors) | < 500ms | < 2s |
| Telegram message delivery | < 1s | < 3s |
| Browser page load + extraction | < 5s | < 15s |
| Terminal command execution | < 2s | per-tool timeout |
| System cold start | < 10s | < 30s |
| MCP tool discovery on connect | < 2s per server | < 10s |

---

## 8. UX Flow — Telegram-First Control

### Initial Setup

```
User → Telegram Bot (@neuralclaw_bot)
    /start
    
Bot → "Welcome to NeuralClaw! I'm your personal AI agent.
       Authorized user confirmed.
       Type /help to see available commands."
```

### Help Menu

```
/help

NeuralClaw Commands:
🧠 AGENT
  /ask <message>     - Chat with agent (interactive mode)
  /run <goal>        - Run an autonomous task
  /plan <goal>       - Show plan before executing (preview)
  /cancel            - Cancel current task

🔧 TOOLS
  /tools             - List available tools
  /status            - Show agent status and active tasks
  /confirm <yes|no>  - Respond to confirmation request

🧠 MEMORY
  /memory search <q> - Search agent memory
  /memory clear      - Clear short-term memory
  /recall            - What was I working on last?

⏰ SCHEDULING
  /schedule <task> every <interval>  - Schedule recurring task
  /tasks             - List scheduled tasks
  /unschedule <id>   - Cancel scheduled task

⚙️ SETTINGS
  /trust <low|medium|high>  - Set trust level (controls confirmations)
  /model <name>             - Switch LLM model
  /ping                     - Check agent is alive
```

### Interactive Conversation Flow

```
User: /ask What Python libraries should I use for async HTTP in 2025?

Bot: [typing indicator shown]

Bot: For async HTTP in Python in 2025, here are the top choices:

**httpx** — The modern standard. Supports both async and sync, 
HTTP/2 out of the box, clean API, excellent for most use cases.

**aiohttp** — Battle-tested, slightly lower level, great for 
high-performance scenarios where you need fine control.

**requests** (with httpx) — If you need sync code, httpx is 
a drop-in requests replacement.

My recommendation: start with **httpx** unless you have specific 
performance requirements.

Anything specific you're building?
```

### Tool Confirmation Flow

```
User: /ask Delete all .pyc files in my ~/projects directory

Bot: ⚠️ Confirmation Required

I'm about to execute:
  Tool: terminal_exec
  Command: find ~/projects -name "*.pyc" -delete
  Risk: HIGH (destructive file operation)
  Files affected: ~847 .pyc files found

[ ✅ Confirm ] [ ❌ Cancel ]

User: [taps Confirm]

Bot: ✅ Executing...

🗑️ Deleted 847 .pyc files from ~/projects
Total space freed: 23.4 MB
Duration: 1.2s
```

### Autonomous Task Flow

```
User: /run Research the top 5 AI coding assistants, compare features, 
      and create a comparison table saved to ~/reports/ai_coding_tools.md

Bot: 📋 Creating plan...

Plan: Research AI Coding Assistants
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: Web search for AI coding assistants
Step 2: Visit each product's website (5 sites)
Step 3: Extract feature information
Step 4: Synthesize comparison with LLM
Step 5: Write markdown report to file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimated: 2-4 minutes | Risk: LOW

[ ▶️ Start ] [ 👁️ Modify ] [ ❌ Cancel ]

User: [taps Start]

Bot: ▶️ Starting task...

[2 min later]

Bot: ✅ Task Complete!

Research: AI Coding Assistants Comparison
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Compared: GitHub Copilot, Cursor, Cody, Tabnine, Continue
📄 Report saved: ~/reports/ai_coding_tools.md (4.2 KB)
⏱️ Duration: 2m 34s | Tool calls: 12

Here's a summary:
[Brief summary of findings]

[📎 Open report] [💬 Ask follow-up]
```

---

## 9. Example Commands

### Research Commands
```
/ask What are the best practices for Python packaging in 2025?
/ask Summarize the Attention Is All You Need paper
/ask Research FastAPI vs Django for a new REST API project and give me a comparison
/run Research llm context window limits across all major models and create a CSV comparison
```

### Code Commands
```
/ask Write a Python script to batch rename files with timestamps
/run Write tests for my ~/projects/api/auth.py module
/ask Review this code for security issues [paste code]
/run Scan ~/projects for TODO comments and create a task list
```

### Browser Commands
```
/ask Go to https://news.ycombinator.com and summarize the top 10 posts
/run Monitor https://example.com/status every 5 minutes and alert me if it goes down
/ask Extract all email addresses from https://example.com/team
```

### System Commands
```
/ask Check disk usage on my machine
/ask List all running Python processes
/run Update all packages in ~/projects/myapp/.venv
/ask What is my machine's IP address?
```

### Memory Commands
```
/recall What was I working on last Tuesday?
/memory search "fastapi authentication"
/ask Summarize everything you know about my NeuralClaw project
```

### Blender Commands
```
/ask Create a simple scene in Blender with a cube and sphere, add a metallic material to both, and render it
/run Generate 10 random abstract 3D objects in Blender and render each as a PNG
```

### Scheduling Commands
```
/schedule "Check my GitHub notifications and summarize new activity" every 1 hour
/schedule "Search for new papers on arxiv about LLM agents" every day at 9am
/tasks
/unschedule task_abc123
```

---

## 10. Milestones

### Milestone 1: Core Foundation (Weeks 1-2)
**Goal:** Agent runs, talks to LLM, has basic memory  
- [x] Project structure created
- [x] LLM client (OpenAI + Anthropic)
- [x] Basic conversation memory (short-term)
- [x] CLI interface working
- [x] Basic logging

**Exit Criteria:** Developer can have a conversation with the agent via CLI, and context persists within a session.

---

### Milestone 2: Tools (Weeks 3-4)
**Goal:** Agent can use tools  
- [x] Tool Bus and Registry
- [x] Safety Kernel (basic)
- [x] Browser tool (Playwright)
- [x] Terminal tool (sandboxed)
- [x] Filesystem tool
- [x] Web search tool

**Exit Criteria:** Developer can ask agent to browse the web, execute commands, and read/write files — with safety confirmation for risky operations.

---

### Milestone 3: Telegram Interface (Week 5)
**Goal:** Control agent from phone  
- [x] Telegram bot setup
- [x] Command routing
- [x] Confirmation dialogs (inline keyboards)
- [x] File send/receive
- [x] Progress updates

**Exit Criteria:** Developer can control agent fully from Telegram, including receiving files and confirming dangerous actions.

---

### Milestone 4: Persistent Memory (Week 6)
**Goal:** Agent remembers across sessions  
- [x] ChromaDB integration (long-term vector memory)
- [x] SQLite episodic memory
- [x] Memory search command
- [x] Context building from memory
- [x] Memory compression

**Exit Criteria:** Agent recalls relevant past context when asked about previous sessions and uses it to improve responses.

---

### Milestone 5: MCP Integration (Week 7)
**Goal:** Connect to external MCP servers  
- [x] MCP Manager
- [x] stdio transport
- [x] HTTP transport
- [x] Blender MCP connection
- [x] Tool auto-discovery and registration

**Exit Criteria:** Agent can control Blender via natural language commands routed through MCP.

---

### Milestone 6: Autonomous Mode (Week 8)
**Goal:** Agent runs multi-step tasks independently  
- [x] Task Planner
- [x] Plan executor
- [x] Task Scheduler (APScheduler)
- [x] Progress notifications
- [x] Error recovery and re-planning

**Exit Criteria:** Developer can assign a multi-step research task and walk away — agent completes it and notifies via Telegram.

---

### Milestone 7: Plugin System + Polish (Week 9-10)
**Goal:** Easy extension + production-ready  
- [x] Plugin loader
- [x] Plugin template
- [x] Configuration system (Pydantic settings)
- [x] Full test suite
- [x] Docker deployment
- [x] Documentation

**Exit Criteria:** Developer can add a new tool by dropping a Python file into plugins/ and restarting. System runs in Docker.

---

## 11. Future Roadmap

### v1.1 — Web Dashboard
- Local web UI (FastAPI + React) for conversation history, memory browser, tool logs
- System metrics dashboard
- Memory visualization (graph view)

### v1.2 — Multi-Agent
- Multiple specialized agents (researcher, coder, monitor)
- Agent-to-agent communication
- Orchestrator agent delegates to specialized sub-agents
- Shared memory pool between agents

### v1.3 — Voice Interface
- Voice input via Telegram voice messages (Whisper transcription)
- Text-to-speech responses (optional)

### v1.4 — Enhanced Autonomy
- Agent-initiated notifications (proactive, not just reactive)
- Goal-oriented background tasks
- Self-improvement: agent writes new plugins based on needs
- Calendar/event awareness

### v1.5 — Cloud Deployment
- Optional VPS deployment mode
- Remote access without Telegram dependency
- Encrypted cloud memory backup
- Multi-device sync

### v2.0 — Enterprise-Grade
- Multi-user support
- Role-based access control
- Team memory namespacing
- Audit dashboard
- SLA monitoring

---

## 12. Risks

### Risk 1: LLM API Costs
**Risk:** Autonomous tasks with many tool calls generate high API costs  
**Severity:** Medium  
**Mitigation:** 
- Cost tracking per session
- Daily/monthly budget limits
- Default to cheaper models for routine tasks
- Option to use local Ollama models

### Risk 2: Terminal Execution Safety
**Risk:** Agent executes destructive terminal commands  
**Severity:** High  
**Mitigation:**
- Strict command whitelist (deny-first policy)
- Docker sandbox for ALL terminal execution (optional, enabled by default in prod)
- CRITICAL risk level for any rm/delete operations
- Mandatory confirmation for all system-level commands

### Risk 3: LLM Hallucination
**Risk:** Agent takes wrong action based on hallucinated context  
**Severity:** Medium  
**Mitigation:**
- Chain-of-thought reasoning before action
- Tool results feed back to LLM for verification
- Human confirmation gates for irreversible actions
- Max retries with different reasoning approach

### Risk 4: Memory Pollution
**Risk:** Bad tool results or incorrect information stored permanently in vector DB  
**Severity:** Medium  
**Mitigation:**
- Memory relevance scoring before storage
- Manual memory deletion command
- Memory age-based decay (optional)
- Separate collections for different memory types

### Risk 5: Telegram API Rate Limits
**Risk:** Excessive messages trigger Telegram rate limiting  
**Severity:** Low  
**Mitigation:**
- Message batching and throttling
- Graceful handling of 429 responses
- Aggregated progress updates (not per-step)

### Risk 6: Dependency Rot
**Risk:** Python/Node dependencies become outdated, breaking system  
**Severity:** Low  
**Mitigation:**
- Pin all dependency versions in requirements.txt
- Monthly dependency update review process
- Docker image with pinned versions

---

## 13. Constraints

### Technical Constraints
- Must run on a single personal machine (not distributed by default)
- Python 3.11+ required
- Must support Linux and Windows (WSL2 for Windows)
- Docker must be optional (not required for basic operation)
- Total disk usage for the platform: < 10GB (excluding LLM model data)
- Must operate within 4GB RAM minimum (8GB recommended)

### Operational Constraints
- Solo developer — no team, no CI/CD pipeline initially
- No external database required — SQLite and ChromaDB (file-based)
- All configuration via YAML/ENV — no database-backed config
- No mandatory cloud services — all optional

### LLM Constraints
- Cannot guarantee LLM output correctness
- LLM context window limits apply
- API keys required for cloud LLMs (not bundled)
- Local models (Ollama) may have reduced capability

### Regulatory Constraints
- No personal data of third parties stored without consent
- Agent should not scrape sites that prohibit it (robots.txt respected)
- No automated account creation or impersonation

---

## 14. Acceptance Criteria

### MVP Acceptance Criteria (Definition of Done)

The system is considered MVP-complete when ALL of the following are true:

1. **Agent runs** and can hold a multi-turn conversation via CLI
2. **LLM Brain** connects to at least 2 providers (OpenAI + one other)
3. **Memory** persists conversations across restarts and can be retrieved semantically
4. **Browser tool** can navigate to a URL and return page content to LLM
5. **Terminal tool** can execute whitelisted commands with output capture
6. **Filesystem tool** can read/write files within allowed paths
7. **Telegram interface** handles all core commands (`/ask`, `/run`, `/status`, `/confirm`)
8. **Safety Kernel** blocks all non-whitelisted terminal commands and requires confirmation for HIGH risk
9. **MCP** connects to Blender MCP server and can call at least one Blender tool
10. **Plugins** directory is scanned on startup and discovered tools are available to LLM
11. **Logging** records all tool calls, LLM calls, and safety decisions in structured JSON
12. **Configuration** is fully externalized in config.yaml / environment variables
13. **Tests** cover at least the Safety Kernel, Tool Bus, and Memory Manager
14. **README** documents setup from scratch to first Telegram interaction in < 30 minutes

---

*End of PRD.md*
