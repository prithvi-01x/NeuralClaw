# Gateway & Web UI Architecture

> **Version:** 1.0.0 · **Last Updated:** 2026-02-27

---

## Overview

NeuralClaw's Gateway is a **WebSocket control plane** that decouples all interfaces (CLI, Web UI, apps, channels) from the core agent. Instead of interfaces talking directly to the orchestrator, they send typed JSON messages over WebSocket to the Gateway, which routes them to the agent kernel and streams responses back.

```
┌──────────┐
│  CLI     │──┐
├──────────┤  │     ┌───────────────┐      ┌──────────────────┐
│  Web UI  │──┼────▶│   Gateway     │─────▶│  Orchestrator    │
├──────────┤  │     │  (ws://9090)  │      │  + Agent Kernel  │
│  Mobile  │──┘     └───────────────┘      └──────────────────┘
│  (future)│              ▲
└──────────┘              │
                   Static HTTP
                   (http://8080)
```

---

## Components

### Gateway Server (`gateway/gateway_server.py`)

The core WebSocket server built on `websockets`. Handles:

| Responsibility | Details |
|---|---|
| **Connection management** | Accept/reject WebSocket connections, optional auth token |
| **Session lifecycle** | Create, update, destroy sessions via `session_store.py` |
| **Message routing** | Route typed JSON messages to the correct handler |
| **Response streaming** | Stream `AgentResponse` chunks back to the client |
| **Confirmation flow** | Bidirectional — server requests user approval, client responds |
| **Admin handlers** | Config, env vars, skills, system info CRUD operations |
| **Static file serving** | Serves the Web UI (`webui/`) on port 8080 |

### Gateway Protocol (`gateway/protocol.py`)

All messages follow a typed JSON schema:

```json
{
  "type": "ask",
  "id": "abc123",
  "session_id": "sess_f76809e",
  "data": {"message": "hello"}
}
```

**Message Types:**

| Type | Direction | Purpose |
|------|-----------|---------|
| `session.create` | Client → Server | Create a new session |
| `session.created` | Server → Client | Session ID assigned |
| `ask` | Client → Server | Send a chat message |
| `run` | Client → Server | Start autonomous task |
| `response` | Server → Client | Response (text/stream) |
| `confirm_request` | Server → Client | Ask user for approval |
| `confirm` | Client → Server | User's approval/denial |
| `config.get` | Client → Server | Read config.yaml |
| `config.set` | Client → Server | Update config.yaml |
| `env.list` | Client → Server | List .env variables |
| `env.set` | Client → Server | Update an env variable |
| `skills.reload` | Client → Server | Hot-reload all skills |
| `system.info` | Client → Server | Get system diagnostics |

### Session Store (`gateway/session_store.py`)

Thread-safe async session store using `asyncio.Lock`. Maps `session_id` → `Session` objects. Manages creation, retrieval, and cleanup.

### Gateway Client (`gateway/gateway_client.py`)

Async Python client for connecting to the gateway. Used by `gateway-cli` and programmatic clients.

---

## Web UI

The Web UI is a single-page application served by the gateway's static file server.

### Files

| File | Purpose |
|------|---------|
| `webui/index.html` | HTML structure — 5-tab layout |
| `webui/style.css` | Premium dark theme with glassmorphism |
| `webui/app.js` | Full application logic |

### Tabs

| Tab | Features |
|-----|----------|
| 💬 **Chat** | Send messages, view responses with markdown, confirmation dialogs, quick actions, trust level selector |
| ⚙️ **Settings** | Edit `config.yaml` — fields organized by section (agent, llm, providers, memory, tools). Type-aware inputs. Save button. |
| 🔧 **Skills** | View all loaded skills with name, description, category badge, risk level badge. Reload button for hot-reload. |
| 🔑 **Env Vars** | View `.env` variables with masked values. Edit/add variables via modal dialog. Changes update both `.env` file and running process. |
| 📊 **System** | Dashboard cards showing version, Python version, platform, skills count, sessions, connections. |

### WebSocket Communication

The Web UI connects directly to the gateway on `ws://hostname:9090`. It auto-reconnects with exponential backoff. The HTTP static server runs on port 8080.

---

## Starting the Web UI

```bash
python main.py --interface webui
```

This starts both:
- **HTTP server** on `http://127.0.0.1:8080` (serves `webui/`)
- **WebSocket gateway** on `ws://127.0.0.1:9090` (handles all communication)

---

## Gateway CLI

A thin CLI client that connects via `GatewayClient`:

```bash
# Start gateway server first
python main.py --interface gateway

# Connect with thin CLI
python main.py --interface gateway-cli
```
