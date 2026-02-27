## Getting Started

Two ways to install NeuralClaw — pip (quick) or clone (recommended for development).

---

### Method 1 — pip install *(experimental)*

> ⚠️ NeuralClaw is currently published on **TestPyPI** for early testing. Expect rough edges.

```bash
# Create and activate a virtual environment
python3 -m venv neuralclaw-env
source neuralclaw-env/bin/activate        # Windows: neuralclaw-env\Scripts\activate

# Install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ neural-claw
```

Once installed, set up your config:

```bash
# Create a working directory
mkdir my-neuralclaw && cd my-neuralclaw

# Create your .env file
cat > .env << 'EOF'
# Add whichever API key(s) you want to use — at least one required unless using Ollama
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
BYTEZ_API_KEY=
OPENROUTER_API_KEY=

# Optional: SerpAPI key for web_search skill
SERPAPI_API_KEY=
EOF

# Create your config.yaml
cat > config.yaml << 'EOF'
agent:
  name: "NeuralClaw"
  version: "1.0.0"
  max_iterations_per_turn: 10
  max_turn_timeout_seconds: 300
  default_trust_level: "low"       # low | medium | high

llm:
  default_provider: "ollama"       # ollama | openai | anthropic | gemini | bytez | openrouter
  default_model: "qwen3:8b"        # model name for chosen provider
  temperature: 0.4
  max_tokens: 4096
  retry:
    max_attempts: 3
    base_delay: 1.0
    max_delay: 30.0
  fallback_providers: []           # e.g. [openai, anthropic]
  providers:
    ollama:
      base_url: "http://localhost:11434"
    openai:
      base_url:
    anthropic: {}
    bytez:
      api_key_env: "BYTEZ_API_KEY"

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/sqlite/episodes.db"
  embedding_model: "BAAI/bge-small-en-v1.5"
  max_short_term_turns: 20
  relevance_threshold: 0.55
  compact_after_turns: 15
  compact_keep_recent: 4

tools:
  terminal:
    working_dir: "./data/agent_files"
    default_timeout_seconds: 30
    docker_sandbox: false
    whitelist_extra: []
  filesystem:
    allowed_paths:
      - "./data/agent_files"

safety:
  default_permission_level: "read"
  require_confirmation_for:
    - "HIGH"
    - "CRITICAL"

scheduler:
  timezone: "UTC"
  max_concurrent_tasks: 3
  heartbeat_enabled: true
  heartbeat_interval_minutes: 30

voice:
  enabled: false
  whisper_model: "base.en"
  whisper_device: "cpu"
  piper_model_path: ""

logging:
  level: "INFO"
  log_dir: "./data/logs"
  console_output: false

clawhub:
  enabled: true
  skills_dir: "./data/clawhub/skills"
  registry_url: "https://clawhub.ai"
  execution:
    allow_binary_skills: true
    auto_install_deps: false
    sandbox_binary_skills: true
  risk_defaults:
    prompt_only: "LOW"
    api_http: "LOW"
    binary_execution: "HIGH"
EOF
```

Then run the onboard wizard:

```bash
neuralclaw onboard
```

---

### Method 2 — Clone *(recommended)*

```bash
# Clone the repo
git clone https://github.com/prithvi-01x/neuralclaw.git
cd neuralclaw

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate              # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment file
cp .env.example .env
# Open .env and fill in your API key(s)
```

Then create your `config.yaml` from the template:

```bash
cp config/config.yaml.example config/config.yaml
# Or use the full reference below
```

<details>
<summary>📄 Full config.yaml reference</summary>

```yaml
agent:
  name: "NeuralClaw"
  version: "1.0.0"
  max_iterations_per_turn: 10
  max_turn_timeout_seconds: 300
  default_trust_level: "low"       # low | medium | high

llm:
  default_provider: "ollama"       # ollama | openai | anthropic | gemini | bytez | openrouter
  default_model: "qwen3:8b"
  temperature: 0.4
  max_tokens: 4096
  retry:
    max_attempts: 3
    base_delay: 1.0
    max_delay: 30.0
  fallback_providers: []
  providers:
    ollama:
      base_url: "http://localhost:11434"
    openai:
      base_url:
    anthropic: {}
    bytez:
      api_key_env: "BYTEZ_API_KEY"

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/sqlite/episodes.db"
  embedding_model: "BAAI/bge-small-en-v1.5"
  max_short_term_turns: 20
  relevance_threshold: 0.55
  compact_after_turns: 15
  compact_keep_recent: 4

tools:
  browser:
    headless: true
    user_agent: "Mozilla/5.0 (compatible; NeuralClaw/1.0)"
    timeout_ms: 15000
  terminal:
    working_dir: "./data/agent_files"
    default_timeout_seconds: 30
    docker_sandbox: false
    docker_image: "python:3.11-slim"
    whitelist_extra: []
  filesystem:
    allowed_paths:
      - "./data/agent_files"

safety:
  default_permission_level: "read"
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

telegram:
  authorized_user_ids: []

scheduler:
  timezone: "UTC"
  max_concurrent_tasks: 3
  heartbeat_enabled: true
  heartbeat_interval_minutes: 30

voice:
  enabled: true
  whisper_model: "base.en"        # tiny | base | small | medium
  whisper_device: "cpu"           # cpu | cuda
  piper_model_path: ""            # path to .onnx model file
  sample_rate: 16000
  channels: 1
  vad_aggressiveness: 0           # 0–3
  silence_duration_ms: 800
  max_utterance_s: 30
  min_utterance_ms: 300
  wake_word_enabled: false
  wake_word_model: "hey_mycroft"
  wake_sensitivity: 0.5
  mic_device_index: 0

logging:
  level: "INFO"                   # DEBUG | INFO | WARNING | ERROR
  log_dir: "./data/logs"
  max_file_size_mb: 100
  backup_count: 5
  console_output: false

skills:
  retry:
    retryable_errors:
      - SkillTimeoutError
      - LLMRateLimitError
    max_attempts: 3
    base_delay: 1.0
    max_delay: 30.0
    jitter: true
    overrides:
      terminal_exec:
        max_attempts: 1           # never retry shell commands
      web_fetch:
        max_attempts: 3

clawhub:
  enabled: true
  skills_dir: "./data/clawhub/skills"
  registry_url: "https://clawhub.ai"
  execution:
    allow_binary_skills: true
    auto_install_deps: false
    sandbox_binary_skills: true
  env:
    block_on_missing_env: true
    show_env_requirements_on_install: true
  risk_defaults:
    prompt_only: "LOW"
    api_http: "LOW"
    binary_execution: "HIGH"
    install_directive: "HIGH"
```

</details>

---

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.11+** | Tested on 3.11, 3.12, 3.13 |
| **Ollama** *(optional)* | For local models — [ollama.com](https://ollama.com/) |
| **API Key** *(optional)* | Any one of: OpenAI, Anthropic, Gemini, Bytez, OpenRouter |

> You need at least one LLM source — either Ollama running locally or one API key in `.env`.

---

### First Run — Onboard Wizard

```bash
python main.py onboard
```

The wizard walks you through LLM provider selection, API key validation, persona creation (`SOUL.md`), and a health check. Run this before anything else.

---

### Start the Agent

```bash
# CLI (default)
python main.py

# Web UI — browser at http://127.0.0.1:8080
python main.py --interface webui

# Gateway WebSocket control plane
python main.py --interface gateway

# Gateway CLI (thin client, connects to running gateway)
python main.py --interface gateway-cli

# Telegram bot
python main.py --interface telegram

# Voice interface (Whisper STT + Piper TTS)
python main.py --interface voice

# Qt tray app
python main.py --interface voice-app
```

### Quick Options

```bash
python main.py --log-level DEBUG           # verbose logging
python main.py --config path/to/conf.yaml  # custom config path
python main.py --enable-mcp               # enable MCP servers
```
