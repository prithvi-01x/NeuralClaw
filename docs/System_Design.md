# System_Design.md — NeuralClaw Autonomous Agent Platform
## Detailed Technical System Design

> **Version:** 1.0.0  
> **Audience:** Senior engineer / Architect  
> **Purpose:** Implementation reference for all system components

---

## Table of Contents

1. [LLM Orchestration](#1-llm-orchestration)
2. [Tool Calling Pipeline](#2-tool-calling-pipeline)
3. [MCP Client Architecture](#3-mcp-client-architecture)
4. [Browser Automation Layer](#4-browser-automation-layer)
5. [Terminal Execution Sandbox](#5-terminal-execution-sandbox)
6. [Memory Architecture](#6-memory-architecture)
7. [Event Loop Design](#7-event-loop-design)
8. [State Management](#8-state-management)
9. [Multi-Agent Extension Design](#9-multi-agent-extension-design)
10. [Scaling Considerations](#10-scaling-considerations)
11. [Deployment Models](#11-deployment-models)
12. [Data Storage Choices](#12-data-storage-choices)
13. [API Structure](#13-api-structure)
14. [Error Handling Strategy](#14-error-handling-strategy)

---

## 1. LLM Orchestration

### 1.1 Abstraction Design

All LLM providers are wrapped behind a single abstract interface. This prevents vendor lock-in and allows provider switching via config only.

```python
# brain/types.py

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum

class MessageRole(Enum):
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    TOOL      = "tool"

@dataclass
class Message:
    role: MessageRole
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    name: Optional[str] = None

@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]

@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCall]
    finish_reason: str        # "stop", "tool_calls", "length", "error"
    input_tokens: int
    output_tokens: int
    model: str
    raw_response: Any

@dataclass
class LLMConfig:
    model: str
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    stream: bool = False
    timeout: int = 120
```

### 1.2 Abstract Base Client

```python
# brain/llm_client.py

from abc import ABC, abstractmethod
from typing import List, Optional
from brain.types import Message, LLMResponse, LLMConfig
from tools.types import ToolSchema

class LLMClient(ABC):
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Message],
        tools: Optional[List[ToolSchema]] = None,
        config: Optional[LLMConfig] = None
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        raise NotImplementedError
    
    @abstractmethod
    async def count_tokens(self, messages: List[Message]) -> int:
        """Count tokens in a message list for context management."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def max_context_tokens(self) -> int:
        """Maximum context window size."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Whether this provider supports native tool calling."""
        raise NotImplementedError

class LLMClientFactory:
    
    @staticmethod
    def create(provider: str, config: Dict) -> LLMClient:
        providers = {
            "openai":    OpenAIClient,
            "anthropic": AnthropicClient,
            "ollama":    OllamaClient,
        }
        if provider not in providers:
            raise ValueError(f"Unknown LLM provider: {provider}")
        return providers[provider](**config)
```

### 1.3 OpenAI Client Implementation

```python
# brain/openai_client.py

import openai
from brain.llm_client import LLMClient
from brain.types import Message, LLMResponse, LLMConfig, ToolCall, MessageRole

class OpenAIClient(LLMClient):
    
    def __init__(self, api_key: str, base_url: Optional[str] = None):
        self.client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model_context_limits = {
            "gpt-4o":       128000,
            "gpt-4-turbo":  128000,
            "gpt-3.5-turbo": 16385,
        }
    
    async def generate(self, messages, tools=None, config=None) -> LLMResponse:
        config = config or LLMConfig(model="gpt-4o")
        
        # Convert internal messages to OpenAI format
        oai_messages = self._convert_messages(messages)
        
        # Convert tool schemas to OpenAI format
        oai_tools = [self._convert_tool_schema(t) for t in tools] if tools else None
        
        params = {
            "model": config.model,
            "messages": oai_messages,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
        }
        if oai_tools:
            params["tools"] = oai_tools
            params["tool_choice"] = "auto"
        
        try:
            response = await self.client.chat.completions.create(**params)
        except openai.APIError as e:
            raise LLMError(f"OpenAI API error: {e}") from e
        
        msg = response.choices[0].message
        
        # Parse tool calls
        tool_calls = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments)
                ))
        
        return LLMResponse(
            content=msg.content,
            tool_calls=tool_calls,
            finish_reason=response.choices[0].finish_reason,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            model=response.model,
            raw_response=response
        )
    
    def _convert_messages(self, messages: List[Message]) -> List[dict]:
        result = []
        for m in messages:
            converted = {"role": m.role.value, "content": m.content}
            if m.tool_call_id:
                converted["tool_call_id"] = m.tool_call_id
            if m.tool_calls:
                converted["tool_calls"] = m.tool_calls
            result.append(converted)
        return result
```

### 1.4 Context Window Management

A critical orchestration concern is keeping the LLM context within token limits:

```
Context Budget Allocation:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total context: 128,000 tokens (GPT-4o)
│
├── System Prompt:        ~2,000 tokens (reserved, static)
├── Tool Schemas:         ~3,000 tokens (reserved, based on tool count)
├── Long-term Memory:     ~5,000 tokens (top-5 relevant past context)
├── Conversation History: ~10,000 tokens (last ~20 turns)
├── Tool Results:         ~5,000 tokens (last 3 tool outputs)
├── Current Message:      ~1,000 tokens
└── Response Buffer:      ~4,096 tokens (max_tokens reserved)
    ──────────────────────────────
    Total used:           ~30,096 tokens
    Free:                 ~97,904 tokens (for long docs, etc.)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

```python
# agent/context_builder.py

class ContextBuilder:
    
    TOKEN_BUDGET = {
        "system_prompt": 2000,
        "tool_schemas":  3000,
        "long_term_mem": 5000,
        "conversation":  10000,
        "tool_results":  5000,
        "current_msg":   2000,
        "response":      4096,
    }
    
    async def build(self, session: Session, user_message: str, tools: List[ToolSchema]) -> List[Message]:
        messages = []
        
        # 1. System prompt (always included)
        messages.append(Message(
            role=MessageRole.SYSTEM,
            content=self._build_system_prompt(session)
        ))
        
        # 2. Retrieve long-term memory context
        if relevant_memories := await self.memory.search(user_message, n=5):
            memory_context = self._format_memories(relevant_memories)
            messages.append(Message(
                role=MessageRole.SYSTEM,
                content=f"<relevant_memory>\n{memory_context}\n</relevant_memory>"
            ))
        
        # 3. Recent conversation history (with token-based truncation)
        history = session.short_term.get_all()
        truncated_history = self._truncate_to_budget(history, self.TOKEN_BUDGET["conversation"])
        messages.extend(truncated_history)
        
        # 4. Current user message
        messages.append(Message(role=MessageRole.USER, content=user_message))
        
        return messages
    
    def _truncate_to_budget(self, messages: List[Message], budget: int) -> List[Message]:
        """Keep most recent messages that fit within token budget."""
        result = []
        tokens_used = 0
        for msg in reversed(messages):
            msg_tokens = self.llm.count_tokens_approx(msg.content)
            if tokens_used + msg_tokens > budget:
                break
            result.insert(0, msg)
            tokens_used += msg_tokens
        return result
```

---

## 2. Tool Calling Pipeline

### 2.1 Pipeline Overview

```
LLM Response with tool_calls
           │
           ▼
┌──────────────────────────────────────────────────┐
│  1. PARSE TOOL CALLS                             │
│  for each tool_call in response.tool_calls:      │
│    → ToolCall(id, name, arguments)              │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  2. VALIDATE ARGUMENTS                           │
│  → JSON Schema validation against tool schema   │
│  → Type coercion (str→int where needed)         │
│  → Required fields check                        │
│  → Raises ToolValidationError on failure        │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  3. SAFETY CHECK                                 │
│  safety_kernel.evaluate(tool_call, session)      │
│  → Returns: APPROVED, BLOCKED, or CONFIRM_NEEDED│
│                                                  │
│  If BLOCKED → Return error result immediately   │
│  If CONFIRM_NEEDED → Request user confirmation  │
│    wait (async) for user response               │
│    If confirmed → continue                      │
│    If denied → Return denied result             │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  4. ROUTE TO HANDLER                             │
│  handler = tool_registry.get(tool_name)          │
│  → Local handler OR MCPClient                   │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  5. EXECUTE WITH TIMEOUT                         │
│  async with asyncio.timeout(tool.timeout):       │
│    result = await handler.execute(**arguments)  │
│  → Catches: TimeoutError, ExecutionError        │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────────┐
│  6. NORMALIZE AND STORE RESULT                  │
│  tool_result = ToolResult(                       │
│    tool_call_id=call.id,                        │
│    content=str(result),                         │
│    error=None,                                  │
│    duration_ms=elapsed                          │
│  )                                               │
│  → Store to memory if significant               │
│  → Emit tool.call.success event                 │
└────────────────────┬─────────────────────────────┘
                     │
                     ▼
        Append ToolResult to conversation as
        Message(role=TOOL, tool_call_id=call.id)
                     │
                     ▼
           Feed back into LLM for next turn
```

### 2.2 Tool Schema Definition

```python
# tools/types.py

@dataclass
class ToolSchema:
    name: str
    description: str
    parameters: Dict          # JSON Schema object
    risk_level: RiskLevel     # LOW, MEDIUM, HIGH, CRITICAL
    timeout_seconds: int = 30
    requires_confirmation: bool = False  # Override safety kernel
    category: str = "general"

@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    error: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    duration_ms: int = 0
    
    @property
    def success(self) -> bool:
        return self.error is None
    
    def to_message(self) -> Message:
        content = self.content if self.success else f"ERROR: {self.error}"
        return Message(
            role=MessageRole.TOOL,
            content=content,
            tool_call_id=self.tool_call_id
        )
```

### 2.3 Parallel Tool Execution

When LLM returns multiple tool calls in one response, execute them in parallel where safe:

```python
# agent/orchestrator.py

async def execute_tool_calls(
    self, 
    tool_calls: List[ToolCall], 
    session: Session
) -> List[ToolResult]:
    
    # Separate into sequential (CRITICAL risk) and parallel (LOW/MEDIUM)
    sequential = [tc for tc in tool_calls if self._requires_sequential(tc)]
    parallel   = [tc for tc in tool_calls if not self._requires_sequential(tc)]
    
    results = []
    
    # Run low-risk tools in parallel
    if parallel:
        tasks = [self.tool_bus.execute(tc, session) for tc in parallel]
        parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
        results.extend(parallel_results)
    
    # Run high-risk tools sequentially with confirmation gates
    for tc in sequential:
        result = await self.tool_bus.execute(tc, session)
        results.append(result)
    
    return results

def _requires_sequential(self, tool_call: ToolCall) -> bool:
    schema = self.tool_registry.get_schema(tool_call.name)
    return schema.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL)
```

---

## 3. MCP Client Architecture

### 3.1 MCP Protocol Overview

The Model Context Protocol defines how AI models communicate with external tool servers. NeuralClaw implements an MCP client that supports three transport types.

```
NeuralClaw Agent
    │
    │ calls tool: blender::create_mesh({"type": "TORUS", "radius": 2.0})
    │
    ▼
MCPManager.call_tool("blender", "create_mesh", {"type": "TORUS", ...})
    │
    ▼
MCPConnection("blender")
    │
    ▼
[stdio transport]
    │  stdin:  {"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"create_mesh","arguments":{...}}}
    │
    ▼
Blender MCP Server Process
    │
    │  stdout: {"jsonrpc":"2.0","id":1,"result":{"content":[{"type":"text","text":"Mesh created: TORUS"}]}}
    │
    ▼
MCPConnection parses result → ToolResult
    │
    ▼
Returns to orchestrator
```

### 3.2 MCP Connection State Machine

```
           ┌─────────────┐
           │  INIT       │
           └──────┬──────┘
                  │ connect()
                  ▼
           ┌─────────────┐
           │  CONNECTING │ ◄─── retry on failure
           └──────┬──────┘
                  │ handshake complete
                  ▼
           ┌─────────────┐
           │  DISCOVERING│ ◄─── list_tools()
           └──────┬──────┘
                  │ tools registered
                  ▼
           ┌─────────────┐
   ┌──────►│  READY      │◄──────────────────┐
   │       └──────┬──────┘                   │
   │              │ call_tool()              │
   │              ▼                          │
   │       ┌─────────────┐                  │
   │       │  EXECUTING  │ ─ result ─────────┘
   │       └──────┬──────┘
   │              │ error / timeout
   │              ▼
   │       ┌─────────────┐
   │       │  ERROR      │ ─ retry ─────────► CONNECTING
   └───────│             │
           └──────┬──────┘
                  │ disconnect()
                  ▼
           ┌─────────────┐
           │  CLOSED     │
           └─────────────┘
```

### 3.3 MCP Manager Implementation

```python
# mcp/mcp_manager.py

class MCPManager:
    """Manages multiple MCP server connections."""
    
    def __init__(self, tool_registry: ToolRegistry, config: MCPConfig):
        self.connections: Dict[str, MCPConnection] = {}
        self.tool_registry = tool_registry
        self.config = config
    
    async def start(self):
        """Connect to all configured MCP servers."""
        for server_name, server_config in self.config.servers.items():
            await self._connect_server(server_name, server_config)
    
    async def _connect_server(self, name: str, config: MCPServerConfig):
        transport = self._create_transport(config)
        conn = MCPConnection(name=name, transport=transport)
        
        try:
            await conn.connect()
            tools = await conn.list_tools()
            
            # Register discovered tools in global registry
            for tool in tools:
                namespaced_name = f"{name}::{tool.name}"
                self.tool_registry.register_mcp_tool(
                    name=namespaced_name,
                    schema=tool,
                    handler=lambda tc, c=conn, t=tool.name: c.call_tool(t, tc.arguments)
                )
            
            self.connections[name] = conn
            logger.info(f"MCP server '{name}' connected. {len(tools)} tools discovered.")
            
        except MCPConnectionError as e:
            logger.error(f"Failed to connect MCP server '{name}': {e}")
    
    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> MCPResult:
        if server_name not in self.connections:
            raise MCPError(f"No connection to server: {server_name}")
        return await self.connections[server_name].call_tool(tool_name, arguments)
    
    def _create_transport(self, config: MCPServerConfig) -> MCPTransport:
        if config.transport == "stdio":
            return StdioTransport(command=config.command, args=config.args, env=config.env)
        elif config.transport == "http":
            return HTTPTransport(url=config.url, headers=config.headers)
        elif config.transport == "sse":
            return SSETransport(url=config.url, headers=config.headers)
        raise ValueError(f"Unknown transport: {config.transport}")
```

### 3.4 Stdio Transport Implementation

```python
# mcp/transports/stdio_transport.py

class StdioTransport(MCPTransport):
    """MCP transport over subprocess stdin/stdout."""
    
    def __init__(self, command: str, args: List[str], env: Dict = None):
        self.command = command
        self.args = args
        self.env = env
        self.process: Optional[asyncio.subprocess.Process] = None
        self._pending: Dict[int, asyncio.Future] = {}
        self._msg_id = 0
        self._reader_task: Optional[asyncio.Task] = None
    
    async def connect(self):
        self.process = await asyncio.create_subprocess_exec(
            self.command, *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, **(self.env or {})}
        )
        self._reader_task = asyncio.create_task(self._read_loop())
        
        # MCP handshake: initialize
        await self._send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "clientInfo": {"name": "neuralclaw", "version": "1.0.0"}
        })
    
    async def send_request(self, method: str, params: Dict) -> Dict:
        msg_id = self._next_id()
        future = asyncio.get_event_loop().create_future()
        self._pending[msg_id] = future
        
        message = json.dumps({
            "jsonrpc": "2.0",
            "id": msg_id,
            "method": method,
            "params": params
        }) + "\n"
        
        self.process.stdin.write(message.encode())
        await self.process.stdin.drain()
        
        return await asyncio.wait_for(future, timeout=30.0)
    
    async def _read_loop(self):
        """Continuously read JSON-RPC responses from server stdout."""
        while True:
            try:
                line = await self.process.stdout.readline()
                if not line:
                    break
                message = json.loads(line.decode())
                if "id" in message and message["id"] in self._pending:
                    future = self._pending.pop(message["id"])
                    if "error" in message:
                        future.set_exception(MCPError(message["error"]))
                    else:
                        future.set_result(message.get("result", {}))
            except Exception as e:
                logger.error(f"MCP read error: {e}")
```

---

## 4. Browser Automation Layer

### 4.1 Design

The browser tool wraps Playwright's async API. A persistent browser context is maintained across calls within a session to preserve cookies and state.

```python
# tools/browser.py

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from tools.types import ToolResult

class BrowserTool:
    """Playwright-based browser automation tool."""
    
    def __init__(self, config: BrowserConfig):
        self.config = config
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._contexts: Dict[str, BrowserContext] = {}  # session_id → context
        self._pages: Dict[str, Page] = {}
    
    async def start(self):
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(
            headless=self.config.headless,
            args=["--no-sandbox", "--disable-setuid-sandbox"]
        )
        logger.info("Browser started")
    
    async def _get_context(self, session_id: str) -> BrowserContext:
        """Get or create a browser context for a session."""
        if session_id not in self._contexts:
            self._contexts[session_id] = await self._browser.new_context(
                user_agent=self.config.user_agent,
                viewport={"width": 1280, "height": 720}
            )
        return self._contexts[session_id]
    
    async def navigate(self, url: str, session_id: str, wait_for: str = None) -> ToolResult:
        """Navigate to URL and return page content."""
        context = await self._get_context(session_id)
        page = await context.new_page()
        
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            
            if wait_for:
                await page.wait_for_selector(wait_for, timeout=5000)
            
            # Extract meaningful content
            content = await self._extract_content(page)
            
            return ToolResult(
                tool_call_id="",  # set by caller
                content=content,
                metadata={"url": page.url, "title": await page.title()}
            )
        finally:
            await page.close()
    
    async def _extract_content(self, page: Page) -> str:
        """Extract readable content from page."""
        # Remove noise (nav, ads, scripts)
        await page.evaluate("""
            ['nav', 'footer', 'script', 'style', 'iframe', '.ad', '#ad'].forEach(s => {
                document.querySelectorAll(s).forEach(el => el.remove());
            });
        """)
        
        # Get text with structure preserved
        content = await page.evaluate("""
            () => {
                const headings = [];
                document.querySelectorAll('h1,h2,h3,p,li,td,th').forEach(el => {
                    const text = el.innerText.trim();
                    if (text) {
                        const tag = el.tagName.toLowerCase();
                        if (tag.startsWith('h')) headings.push('#'.repeat(parseInt(tag[1])) + ' ' + text);
                        else headings.push(text);
                    }
                });
                return headings.join('\\n');
            }
        """)
        
        # Truncate if too large
        if len(content) > 50000:
            content = content[:50000] + "\n[Content truncated at 50,000 chars]"
        
        return content
    
    async def screenshot(self, session_id: str, selector: str = None) -> bytes:
        """Take a screenshot, optionally of a specific element."""
        context = await self._get_context(session_id)
        pages = context.pages
        if not pages:
            raise ToolError("No active page to screenshot")
        page = pages[-1]
        
        if selector:
            element = await page.query_selector(selector)
            return await element.screenshot()
        return await page.screenshot(full_page=True)
    
    async def click(self, selector: str, session_id: str) -> ToolResult:
        context = await self._get_context(session_id)
        pages = context.pages
        page = pages[-1]
        await page.click(selector)
        await page.wait_for_load_state("networkidle")
        content = await self._extract_content(page)
        return ToolResult(tool_call_id="", content=content)
    
    async def fill_form(self, fields: Dict[str, str], submit_selector: str, session_id: str) -> ToolResult:
        context = await self._get_context(session_id)
        page = context.pages[-1]
        
        for selector, value in fields.items():
            await page.fill(selector, value)
        
        if submit_selector:
            await page.click(submit_selector)
            await page.wait_for_load_state("networkidle")
        
        return ToolResult(tool_call_id="", content=await self._extract_content(page))
```

### 4.2 Browser Tool Registration

```python
@tool_registry.register(
    name="browser_navigate",
    description="Navigate to a URL and return the page content as text",
    risk_level=RiskLevel.LOW,
    schema={
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "Full URL to navigate to"},
            "wait_for": {"type": "string", "description": "Optional CSS selector to wait for"}
        },
        "required": ["url"]
    }
)
async def browser_navigate_handler(url: str, wait_for: str = None, **ctx) -> ToolResult:
    return await browser_tool.navigate(url, session_id=ctx["session_id"], wait_for=wait_for)
```

---

## 5. Terminal Execution Sandbox

### 5.1 Sandbox Architecture

```
User request: run terminal command
        │
        ▼
┌─────────────────────────────────┐
│  WHITELIST CHECK                │
│  Is command prefix whitelisted? │
│  Default allowed:               │
│   - ls, cat, head, tail, grep   │
│   - find, wc, echo, pwd, df     │
│   - git status/log/diff         │
│   - pip, python, node (read)    │
│  Default blocked:               │
│   - rm, rmdir, dd, mkfs         │
│   - sudo, su, chmod 777         │
│   - curl | sh, wget | bash      │
└──────────────┬──────────────────┘
               │ passes whitelist
               ▼
┌─────────────────────────────────┐
│  RISK SCORE                     │
│  HIGH if: any pipe+exec pattern │
│  HIGH if: writes to system dirs │
│  MEDIUM if: network commands    │
│  LOW if: read-only commands     │
└──────────────┬──────────────────┘
               │
               ▼
     [safety kernel gate]
               │
               ▼
┌─────────────────────────────────┐
│  EXECUTION METHOD               │
│  IF docker sandbox enabled:     │
│    run in isolated container    │
│  ELSE:                          │
│    run in subprocess with:      │
│    - timeout enforcement        │
│    - resource limits (ulimit)   │
│    - restricted working dir     │
└──────────────┬──────────────────┘
               │
               ▼
        Return (stdout, stderr, exit_code)
```

### 5.2 Terminal Tool Implementation

```python
# tools/terminal.py

import asyncio
import shlex
from safety.whitelist import CommandWhitelist

class TerminalTool:
    
    WHITELISTED_PREFIXES = {
        # Read operations
        "ls", "cat", "head", "tail", "grep", "find", "wc",
        "echo", "pwd", "df", "du", "free", "ps", "top",
        "file", "stat", "md5sum", "sha256sum",
        # Development
        "git", "python", "python3", "pip", "pip3", "node",
        "npm", "cargo", "go", "make",
        # Text processing  
        "awk", "sed", "sort", "uniq", "cut", "tr", "jq",
        # Network read
        "curl", "wget", "ping", "nslookup", "dig",
    }
    
    BLOCKED_PATTERNS = [
        r"rm\s+-rf",           # recursive delete
        r"dd\s+",              # disk destroyer
        r"mkfs",               # format filesystem
        r">\s*/dev/",          # write to device
        r"curl.*\|\s*bash",    # curl pipe to bash
        r"wget.*\|\s*bash",    # wget pipe to bash
        r"sudo",               # privilege escalation
        r"chmod\s+777",        # overly permissive chmod
        r":(){ :|:& };:",      # fork bomb
    ]
    
    def __init__(self, config: TerminalConfig, docker_sandbox: bool = False):
        self.config = config
        self.docker_sandbox = docker_sandbox
        self.allowed_dir = os.path.expanduser(config.working_dir)
    
    async def execute(
        self, 
        command: str, 
        working_dir: str = None,
        timeout: int = 30,
        env_vars: Dict = None
    ) -> ToolResult:
        
        # Validate command
        self._validate_command(command)
        
        # Resolve working directory
        work_dir = os.path.expanduser(working_dir or self.allowed_dir)
        if not self._is_allowed_path(work_dir):
            raise ToolError(f"Working directory not in allowed paths: {work_dir}")
        
        if self.docker_sandbox:
            return await self._execute_in_docker(command, work_dir, timeout, env_vars)
        else:
            return await self._execute_subprocess(command, work_dir, timeout, env_vars)
    
    def _validate_command(self, command: str):
        """Validate command against whitelist and blocked patterns."""
        import re
        
        # Check blocked patterns first
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                raise SecurityError(f"Command matches blocked pattern: {pattern}")
        
        # Check whitelist (first word of command)
        first_word = shlex.split(command)[0].split("/")[-1]  # handle full paths
        if first_word not in self.WHITELISTED_PREFIXES:
            raise SecurityError(
                f"Command '{first_word}' not in whitelist. "
                f"Add to whitelist if intentional."
            )
    
    async def _execute_subprocess(
        self, command: str, work_dir: str, timeout: int, env_vars: Dict
    ) -> ToolResult:
        
        env = {**os.environ}
        if env_vars:
            env.update(env_vars)
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), 
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                raise ToolError(f"Command timed out after {timeout}s")
            
            output = stdout.decode("utf-8", errors="replace")
            error  = stderr.decode("utf-8", errors="replace")
            
            # Truncate large outputs
            if len(output) > 20000:
                output = output[:20000] + f"\n[Output truncated. {len(output)} total chars]"
            
            content = output
            if error and proc.returncode != 0:
                content += f"\n\nSTDERR:\n{error}"
            
            return ToolResult(
                tool_call_id="",
                content=content if content else "(no output)",
                error=None if proc.returncode == 0 else f"Exit code: {proc.returncode}",
                metadata={"exit_code": proc.returncode, "command": command}
            )
            
        except SecurityError:
            raise
        except Exception as e:
            raise ToolError(f"Execution failed: {e}") from e
    
    async def _execute_in_docker(
        self, command: str, work_dir: str, timeout: int, env_vars: Dict
    ) -> ToolResult:
        """Execute command inside an isolated Docker container."""
        
        env_flags = " ".join(f"-e {k}={v}" for k, v in (env_vars or {}).items())
        
        docker_cmd = (
            f"docker run --rm "
            f"--memory=512m --cpus=0.5 "
            f"--network=none "  # No network in sandbox
            f"--read-only --tmpfs /tmp "
            f"-v {work_dir}:/workspace:ro "  # Mount as read-only
            f"{env_flags} "
            f"-w /workspace "
            f"python:3.11-slim "
            f"bash -c {shlex.quote(command)}"
        )
        
        return await self._execute_subprocess(docker_cmd, "/", timeout, {})
```

---

## 6. Memory Architecture

### 6.1 Vector Memory (ChromaDB)

```python
# memory/long_term.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

class LongTermMemory:
    
    COLLECTIONS = {
        "conversations": "Past conversation summaries",
        "knowledge":     "Learned facts and context",
        "tool_results":  "Significant tool execution results",
        "plans":         "Completed task plans and outcomes",
    }
    
    def __init__(self, persist_dir: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedder = SentenceTransformer(embedding_model)
        self._collections = {}
        self._init_collections()
    
    def _init_collections(self):
        for name, description in self.COLLECTIONS.items():
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"description": description, "hnsw:space": "cosine"}
            )
    
    async def store(
        self, 
        text: str, 
        collection: str = "conversations",
        metadata: Dict = None,
        doc_id: str = None
    ) -> str:
        """Embed and store text in specified collection."""
        
        collection_obj = self._collections[collection]
        
        # Compute embedding (run in thread pool to not block event loop)
        embedding = await asyncio.get_event_loop().run_in_executor(
            None, self.embedder.encode, [text]
        )
        
        doc_id = doc_id or f"{collection}_{uuid4().hex}"
        
        collection_obj.add(
            documents=[text],
            embeddings=embedding.tolist(),
            metadatas=[{
                "timestamp": datetime.utcnow().isoformat(),
                **(metadata or {})
            }],
            ids=[doc_id]
        )
        
        return doc_id
    
    async def search(
        self, 
        query: str, 
        collection: str = "conversations",
        n: int = 5,
        filter: Dict = None
    ) -> List[MemoryResult]:
        """Semantic search over a collection."""
        
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            None, self.embedder.encode, [query]
        )
        
        results = self._collections[collection].query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(n, self._collections[collection].count()),
            where=filter
        )
        
        memories = []
        for i, doc in enumerate(results["documents"][0]):
            memories.append(MemoryResult(
                content=doc,
                distance=results["distances"][0][i],
                metadata=results["metadatas"][0][i],
                id=results["ids"][0][i]
            ))
        
        # Filter by relevance threshold
        return [m for m in memories if m.distance < 0.85]
    
    async def search_all(self, query: str, n_per_collection: int = 2) -> List[MemoryResult]:
        """Search across all collections."""
        all_results = []
        for collection in self.COLLECTIONS:
            results = await self.search(query, collection=collection, n=n_per_collection)
            all_results.extend(results)
        
        # Sort by relevance across collections
        return sorted(all_results, key=lambda r: r.distance)[:n_per_collection * len(self.COLLECTIONS)]
```

### 6.2 Episodic Memory (SQLite)

```python
# memory/episodic.py

import aiosqlite
from datetime import datetime

class EpisodicMemory:
    """Records complete task episodes in SQLite for structured querying."""
    
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS episodes (
        id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        goal TEXT,
        plan_json TEXT,
        steps_json TEXT,
        outcome TEXT,
        tool_calls_count INTEGER,
        duration_seconds REAL,
        success INTEGER,
        created_at TEXT,
        completed_at TEXT
    );
    
    CREATE TABLE IF NOT EXISTS tool_calls (
        id TEXT PRIMARY KEY,
        episode_id TEXT,
        session_id TEXT NOT NULL,
        tool_name TEXT NOT NULL,
        params_json TEXT,
        result_summary TEXT,
        success INTEGER,
        duration_ms INTEGER,
        risk_level TEXT,
        timestamp TEXT,
        FOREIGN KEY (episode_id) REFERENCES episodes(id)
    );
    
    CREATE TABLE IF NOT EXISTS reflections (
        id TEXT PRIMARY KEY,
        episode_id TEXT,
        lesson_learned TEXT,
        context TEXT,
        timestamp TEXT
    );
    
    CREATE INDEX IF NOT EXISTS idx_episodes_session ON episodes(session_id);
    CREATE INDEX IF NOT EXISTS idx_episodes_created ON episodes(created_at);
    CREATE INDEX IF NOT EXISTS idx_tool_calls_session ON tool_calls(session_id);
    CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    async def init(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(self.SCHEMA)
            await db.commit()
    
    async def record_episode(self, episode: Episode):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO episodes 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                episode.id, episode.session_id, episode.goal,
                json.dumps(episode.plan), json.dumps(episode.steps),
                episode.outcome, episode.tool_calls_count,
                episode.duration_seconds, int(episode.success),
                episode.created_at.isoformat(), episode.completed_at.isoformat()
            ))
            await db.commit()
    
    async def query_episodes(
        self, 
        session_id: str = None,
        since: datetime = None,
        limit: int = 20
    ) -> List[Episode]:
        conditions, params = [], []
        if session_id:
            conditions.append("session_id = ?")
            params.append(session_id)
        if since:
            conditions.append("created_at >= ?")
            params.append(since.isoformat())
        
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.append(limit)
        
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT * FROM episodes {where} ORDER BY created_at DESC LIMIT ?",
                params
            ) as cursor:
                rows = await cursor.fetchall()
                return [Episode.from_row(row) for row in rows]
```

---

## 7. Event Loop Design

### 7.1 Asyncio Architecture

The entire system runs on a single asyncio event loop. All I/O is non-blocking. CPU-bound tasks (embedding computation) are offloaded to a thread pool executor.

```
asyncio Event Loop
│
├── Telegram Bot (aiogram polling / webhook)
│   └── on_message() → session.dispatch(message)
│
├── CLI Interface (aioconsole readline)
│   └── on_input() → session.dispatch(input)
│
├── Scheduler (APScheduler async)
│   └── on_trigger() → agent.run_task(task)
│
├── MCP connections (per-server reader tasks)
│   └── stdio_reader_loop() [background task]
│   └── http_poller_loop() [background task]
│
└── Agent Orchestrator (per-session coroutines)
    └── process_message() → context_build → llm_call → tool_dispatch
```

### 7.2 Session Concurrency Model

Multiple sessions (users) can run concurrently. Each session has its own task queue and runs independently. Tool calls within a session can be parallelized (for low-risk tools). Cross-session resource sharing (browser, memory) uses async locks.

```python
# agent/session.py

class Session:
    def __init__(self, session_id: str, user_id: str):
        self.id = session_id
        self.user_id = user_id
        self.short_term = ConversationBuffer(max_turns=20)
        self.state = SessionState()
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._current_task: Optional[asyncio.Task] = None
        self._cancelled = asyncio.Event()
    
    async def dispatch(self, message: UserMessage):
        """Queue a message for processing."""
        await self._task_queue.put(message)
    
    async def run(self, orchestrator: Orchestrator):
        """Process messages from queue."""
        while True:
            message = await self._task_queue.get()
            self._current_task = asyncio.create_task(
                orchestrator.process(message, self)
            )
            try:
                await self._current_task
            except asyncio.CancelledError:
                logger.info(f"Session {self.id}: task cancelled")
            finally:
                self._current_task = None
    
    async def cancel_current(self):
        """Cancel currently running task."""
        if self._current_task and not self._current_task.done():
            self._current_task.cancel()
            self._cancelled.set()
```

---

## 8. State Management

### 8.1 State Hierarchy

```
Global State (process lifetime)
│   LLM clients (initialized once)
│   Browser instance (singleton)
│   MCP connections (maintained)
│   Tool registry (static after startup)
│   Scheduler (always running)
│
Per-Session State (per Telegram chat / CLI session)
│   ConversationBuffer (short-term memory)
│   SessionState (active plan, trust level, model choice)
│   Pending confirmations
│   Task cancellation flags
│
Per-Request State (single agent loop iteration)
    Context (built fresh per request)
    Tool call results (temporary accumulator)
    LLM response (discarded after turn)
```

### 8.2 State Persistence

| State Type | Storage | Persistence |
|---|---|---|
| Conversation history | In-memory + SQLite on session end | Session |
| Long-term memories | ChromaDB (disk) | Permanent |
| Episodes | SQLite (disk) | Permanent |
| Scheduled tasks | SQLite (disk) | Across restarts |
| MCP server config | YAML config file | Manual update |
| Trust level | Session state (in-memory) | Session only |
| Browser cookies | Playwright context (in-memory) | Session only |

---

## 9. Multi-Agent Extension Design

The platform is designed so that multiple agent instances can cooperate:

```
Orchestrator Agent (coordinator)
│
├── Research Agent (specialized)
│   └── Tools: browser, search, filesystem
│
├── Coder Agent (specialized)
│   └── Tools: terminal, filesystem, python_eval, git
│
├── Monitor Agent (specialized)
│   └── Tools: browser, http_check, telegram_notify
│
└── Blender Agent (specialized)
    └── Tools: blender MCP (all)
```

### 9.1 Agent-to-Agent Communication

```python
# agent/multi_agent.py

class AgentBus:
    """Pub/sub message bus for inter-agent communication."""
    
    def __init__(self):
        self._agents: Dict[str, Agent] = {}
        self._subscriptions: Dict[str, List[Callable]] = defaultdict(list)
    
    def register(self, agent_id: str, agent: Agent):
        self._agents[agent_id] = agent
    
    async def delegate(self, from_agent: str, to_agent: str, task: Task) -> TaskResult:
        """Delegate a task from one agent to another."""
        if to_agent not in self._agents:
            raise AgentError(f"Unknown agent: {to_agent}")
        
        result = await self._agents[to_agent].execute_task(task)
        return result
    
    async def broadcast(self, event: AgentEvent):
        """Broadcast an event to all subscribed agents."""
        for handler in self._subscriptions[event.type]:
            asyncio.create_task(handler(event))

# Example: Orchestrator delegates research to Research Agent
async def run_complex_task(goal: str):
    # Research sub-task → delegate to Research Agent
    research_result = await agent_bus.delegate(
        from_agent="orchestrator",
        to_agent="researcher",
        task=Task(goal=f"Research: {goal}", tools=["browser", "search"])
    )
    
    # Coding sub-task → delegate to Coder Agent
    code_result = await agent_bus.delegate(
        from_agent="orchestrator",
        to_agent="coder",
        task=Task(
            goal=f"Implement based on research: {research_result.summary}",
            context=research_result.content,
            tools=["terminal", "filesystem"]
        )
    )
```

---

## 10. Scaling Considerations

### 10.1 Current Bottlenecks (Single Instance)

| Bottleneck | Impact | Mitigation |
|---|---|---|
| LLM API latency | 2-10s per call | Streaming, caching common queries |
| ChromaDB search | Grows with collection size | Index tuning, collection size limits |
| Browser instance | Single browser shared | Per-session contexts, timeout cleanup |
| Tool parallelism | Sequential by default | Parallel execution for LOW-risk tools |

### 10.2 Vertical Scaling (More Power on One Machine)

- Increase concurrent sessions via higher `MAX_SESSIONS` config
- Run Ollama local model on GPU for sub-1s LLM responses
- Use SSD for ChromaDB storage (critical for search speed)

### 10.3 Horizontal Scaling (Future - Multiple Nodes)

- Replace ChromaDB with Qdrant (supports distributed deployment)
- Replace SQLite with PostgreSQL for shared state
- Add Redis for session state sharing between instances
- Agent instances communicate via message queue (Redis/RabbitMQ)

---

## 11. Deployment Models

### 11.1 Local Development (Minimal)

```bash
# Install dependencies
pip install -r requirements.txt
playwright install chromium

# Configure
cp .env.example .env
# Edit .env with API keys

# Start
python main.py --interface cli
```

### 11.2 Full Local Deployment

```bash
# Start Telegram bot + all features
python main.py --interface telegram --enable-mcp --enable-scheduler

# With Ollama local LLM
ollama serve &
python main.py --interface telegram --llm ollama --model llama3
```

### 11.3 Docker Deployment

```yaml
# docker-compose.yml

version: "3.9"

services:
  neuralclaw:
    build: .
    container_name: neuralclaw
    restart: unless-stopped
    volumes:
      - ./data:/app/data              # Persistent storage
      - ./config:/app/config          # Configuration
      - /tmp/agent_files:/agent_files # Agent file output
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN}
    ports:
      - "8080:8080"  # Optional web dashboard
    
  sandbox:
    image: python:3.11-slim
    container_name: neuralclaw_sandbox
    network_mode: none              # No network for sandbox
    read_only: true
    tmpfs:
      - /tmp
    volumes:
      - /tmp/agent_files:/workspace:ro
```

### 11.4 VPS Deployment

```bash
# On VPS (Ubuntu 22.04)
git clone https://github.com/you/neuralclaw
cd neuralclaw
cp .env.example .env && nano .env

# Install and start with systemd
sudo cp deploy/neuralclaw.service /etc/systemd/system/
sudo systemctl enable neuralclaw
sudo systemctl start neuralclaw

# Or Docker
docker-compose up -d
```

---

## 12. Data Storage Choices

| Data Type | Storage | Rationale |
|---|---|---|
| Long-term memories | **ChromaDB** (local) | Embedded vector DB, no server needed, file-based, Python native |
| Episodes, tool calls | **SQLite** (aiosqlite) | ACID-compliant, zero-config, excellent for structured queries on single machine |
| Config + secrets | **YAML + .env** | Human-readable, standard for personal projects |
| Agent files output | **Local filesystem** | Direct, inspectable by developer |
| Browser sessions | **Playwright in-memory** | Not persisted by design (re-login when needed) |
| Logs | **JSONL files** | Structured, appendable, inspectable with jq |
| Scheduled tasks | **SQLite** | Persists through restart |

### Why not PostgreSQL/Redis?

For a single-developer personal agent, SQLite + ChromaDB is sufficient and avoids running additional services. The migration path to PostgreSQL/Redis is straightforward when multi-user or distributed needs arise.

---

## 13. API Structure

### 13.1 Internal Module APIs

```python
# Public interface for each major module

# Orchestrator
orchestrator.process_message(message: str, session: Session) -> AgentResponse
orchestrator.cancel_current_task(session_id: str) -> bool

# Memory Manager
memory.store(text: str, collection: str, metadata: Dict) -> str
memory.search(query: str, n: int, collection: str) -> List[MemoryResult]
memory.get_recent_history(session_id: str, n: int) -> List[Message]
memory.commit_episode(episode: Episode) -> None

# Tool Bus
tool_bus.execute(tool_call: ToolCall, session: Session) -> ToolResult
tool_bus.get_schemas() -> List[ToolSchema]
tool_bus.list_tools() -> List[str]

# Safety Kernel
safety.evaluate(tool_call: ToolCall, session: Session) -> SafetyDecision
safety.request_confirmation(tool_call: ToolCall, session: Session) -> bool

# MCP Manager
mcp.connect(server_name: str) -> None
mcp.disconnect(server_name: str) -> None
mcp.call_tool(server: str, tool: str, args: Dict) -> MCPResult
mcp.get_connected_servers() -> List[str]

# Scheduler
scheduler.add_task(task: ScheduledTask) -> str (task_id)
scheduler.remove_task(task_id: str) -> bool
scheduler.list_tasks() -> List[ScheduledTask]
```

### 13.2 Configuration Schema

```yaml
# config/config.yaml

agent:
  name: "NeuralClaw"
  max_iterations_per_turn: 10
  max_turn_timeout_seconds: 300
  default_trust_level: "low"   # low | medium | high

llm:
  default_provider: "openai"   # openai | anthropic | ollama
  default_model: "gpt-4o"
  temperature: 0.7
  max_tokens: 4096
  providers:
    openai:
      api_key: ${OPENAI_API_KEY}
      base_url: null           # null = default OpenAI
    anthropic:
      api_key: ${ANTHROPIC_API_KEY}
    ollama:
      base_url: "http://localhost:11434"

memory:
  chroma_persist_dir: "./data/chroma"
  sqlite_path: "./data/episodes.db"
  embedding_model: "all-MiniLM-L6-v2"  # local embedding model
  max_short_term_turns: 20
  relevance_threshold: 0.85

tools:
  browser:
    headless: true
    user_agent: "Mozilla/5.0 (compatible; NeuralClaw/1.0)"
    timeout_ms: 15000
  terminal:
    working_dir: "~/agent_files"
    default_timeout_seconds: 30
    docker_sandbox: false
    docker_image: "python:3.11-slim"
  filesystem:
    allowed_paths:
      - "~/agent_files"
      - "~/projects"
      - "~/Documents"

safety:
  default_permission_level: "read"
  require_confirmation_for: ["HIGH", "CRITICAL"]
  terminal_whitelist_extra: []   # Additional allowed commands

mcp:
  servers:
    blender:
      transport: stdio
      command: "uvx"
      args: ["blender-mcp"]
      enabled: false           # Set to true when Blender running
    
telegram:
  bot_token: ${TELEGRAM_BOT_TOKEN}
  authorized_user_ids:
    - ${TELEGRAM_USER_ID}     # Your Telegram numeric user ID

scheduler:
  timezone: "UTC"
  max_concurrent_tasks: 3

logging:
  level: "INFO"
  log_dir: "./data/logs"
  max_file_size_mb: 100
  backup_count: 5
```

---

## 14. Error Handling Strategy

### 14.1 Error Taxonomy

```python
class NeuralClawError(Exception): pass

# LLM errors
class LLMError(NeuralClawError): pass
class LLMRateLimitError(LLMError): pass
class LLMContextLengthError(LLMError): pass
class LLMTimeoutError(LLMError): pass

# Tool errors
class ToolError(NeuralClawError): pass
class ToolValidationError(ToolError): pass
class ToolTimeoutError(ToolError): pass
class ToolNotFoundError(ToolError): pass

# Safety errors
class SecurityError(NeuralClawError): pass
class PermissionDeniedError(SecurityError): pass
class ConfirmationDeniedError(SecurityError): pass

# Memory errors
class MemoryError(NeuralClawError): pass

# MCP errors
class MCPError(NeuralClawError): pass
class MCPConnectionError(MCPError): pass
```

### 14.2 Retry Strategy

```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

@retry(
    retry=retry_if_exception_type((LLMRateLimitError, LLMTimeoutError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3)
)
async def llm_call_with_retry(client, messages, tools, config):
    return await client.generate(messages, tools, config)
```

### 14.3 Error Propagation to User

All errors ultimately surface to the user interface with appropriate context:

```python
async def handle_error(error: Exception, session: Session, interface: Interface):
    
    if isinstance(error, SecurityError):
        msg = f"🚫 Security: {error}"
    elif isinstance(error, ToolTimeoutError):
        msg = f"⏰ Tool timed out: {error}"
    elif isinstance(error, LLMError):
        msg = f"🤖 LLM error: {error}\nRetrying..."
    elif isinstance(error, MCPError):
        msg = f"🔌 MCP connection error: {error}"
    else:
        msg = f"❌ Unexpected error: {type(error).__name__}: {error}"
    
    await interface.send_message(session.user_id, msg)
    logger.error("Error in session", error=error, session_id=session.id, exc_info=True)
```

---

*End of System_Design.md*
