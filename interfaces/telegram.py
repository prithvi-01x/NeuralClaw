"""
interfaces/telegram.py — NeuralClaw Telegram Bot Interface

Full Telegram control interface using python-telegram-bot (aiogram-style async).
Lets the developer control the agent from a phone, anywhere in the world.

Features:
  - Authorized user whitelist (TELEGRAM_USER_ID in .env)
  - /ask  — interactive Q&A with the agent
  - /run  — autonomous multi-step task execution
  - /status  — session stats
  - /memory  — search long-term memory
  - /tools   — list registered tools
  - /trust   — change trust level
  - /cancel  — cancel running task
  - /help    — show this list
  - Inline confirmation dialogs for HIGH/CRITICAL tool calls
  - Streaming progress updates as messages
  - Graceful shutdown on SIGINT/SIGTERM

Usage:
    python main.py --interface telegram
"""

from __future__ import annotations

import asyncio
import html
from typing import Optional

from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.orchestrator import Orchestrator
from agent.response_synthesizer import AgentResponse, ResponseKind
from agent.session import Session
from brain import LLMClientFactory
from config.settings import Settings
from memory.memory_manager import MemoryManager
from observability.logger import get_logger
from safety.safety_kernel import SafetyKernel
from skills.types import TrustLevel

# Skill system
from pathlib import Path as _SkillPath
from skills.loader import SkillLoader as _SkillLoader
from skills.bus import SkillBus as _SkillBus

from interfaces.model_selector import (
    build_telegram_model_keyboard,
    build_telegram_model_keyboard_async,
    format_telegram_model_list,
    fetch_ollama_options,
    build_llm_client_for_model,
    save_default_model,
    load_default_model,
    current_model_key,
    MODEL_OPTIONS,
)

log = get_logger(__name__)


_MAX_MESSAGE_LEN = 4000  # Telegram limit is 4096 chars


class TelegramBot:
    """
    NeuralClaw Telegram Bot.

    One Session is created per authorized chat_id on first message.
    """

    def __init__(self, settings: Settings):
        self._settings = settings
        self._authorized_ids: set[int] = set(settings.authorized_telegram_ids)
        self._sessions: dict[int, Session] = {}
        self._orchestrators: dict[int, Orchestrator] = {}
        self._memory: Optional[MemoryManager] = None
        self._app: Optional[Application] = None
        # Per-chat active model key (provider:model_id)
        self._active_models: dict[int, str] = {}
        # Pending default selection (chat_id → last selected model_key)
        self._pending_default: dict[int, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialize and start the bot polling loop."""
        token = self._settings.telegram_bot_token
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set.")

        # Init memory
        self._memory = MemoryManager(
            chroma_persist_dir=self._settings.memory.chroma_persist_dir,
            sqlite_path=self._settings.memory.sqlite_path,
            embedding_model=self._settings.memory.embedding_model,
            max_short_term_turns=self._settings.memory.max_short_term_turns,
            relevance_threshold=self._settings.memory.relevance_threshold,
        )
        await self._memory.init()

        # Build the application
        self._app = Application.builder().token(token).build()
        self._register_handlers()

        log.info("telegram.starting", authorized_ids=list(self._authorized_ids))
        print(f"🤖 NeuralClaw Telegram bot starting…")
        print(f"   Authorized users: {list(self._authorized_ids)}")

        await self._app.initialize()
        await self._app.start()
        await self._app.bot.initialize()

        # start polling manually
        await self._app.updater.start_polling()

        # keep alive forever
        await asyncio.Event().wait()

    async def stop(self) -> None:
        if self._memory:
            await self._memory.close()
        if self._app:
            await self._app.stop()
        log.info("telegram.stopped")

    # ── Handler registration ──────────────────────────────────────────────────

    def _register_handlers(self) -> None:
        app = self._app
        app.add_handler(CommandHandler("start", self._cmd_start))
        app.add_handler(CommandHandler("help", self._cmd_help))
        app.add_handler(CommandHandler("ask", self._cmd_ask))
        app.add_handler(CommandHandler("run", self._cmd_run))
        app.add_handler(CommandHandler("status", self._cmd_status))
        app.add_handler(CommandHandler("memory", self._cmd_memory))
        app.add_handler(CommandHandler("tools", self._cmd_tools))
        app.add_handler(CommandHandler("trust", self._cmd_trust))
        app.add_handler(CommandHandler("cancel", self._cmd_cancel))
        app.add_handler(CommandHandler("clear", self._cmd_clear))
        app.add_handler(CommandHandler("model", self._cmd_model))

        # Callback for inline confirmation buttons
        app.add_handler(CallbackQueryHandler(self._on_confirm_callback))

        # Plain text messages → /ask
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text)
        )

    # ── Auth helper ───────────────────────────────────────────────────────────

    def _is_authorized(self, user_id: int) -> bool:
        if not self._authorized_ids:
            return True  # no restriction configured
        return user_id in self._authorized_ids

    async def _auth_check(self, update: Update) -> bool:
        """Return True if authorized; send error and return False otherwise."""
        user = update.effective_user
        if not user or not self._is_authorized(user.id):
            await update.message.reply_text("⛔ Unauthorized.")
            log.warning("telegram.unauthorized", user_id=user.id if user else None)
            return False
        return True

    # ── Session / Orchestrator factory ────────────────────────────────────────

    def _get_session(self, chat_id: int) -> Session:
        if chat_id not in self._sessions:
            self._sessions[chat_id] = Session.create(
                user_id=str(chat_id),
                trust_level=TrustLevel(self._settings.agent.default_trust_level),
                max_turns=self._settings.memory.max_short_term_turns,
            )
            # Register in MemoryManager._sessions so both share the same
            # ShortTermMemory object — prevents ghost-session desync (finding #3).
            if self._memory:
                self._memory._sessions[self._sessions[chat_id].id] = (
                    self._sessions[chat_id].memory
                )
            log.info("telegram.session_created", chat_id=chat_id)
        return self._sessions[chat_id]

    def _get_orchestrator(self, chat_id: int, update: Update) -> Orchestrator:
        if chat_id not in self._orchestrators:
            session = self._get_session(chat_id)

            llm_client = LLMClientFactory.from_settings(self._settings)

            allowed_paths = (
                self._settings.tools.filesystem.allowed_paths
                or ["./data/agent_files"]
            )
            extra_cmds = self._settings.tools.terminal.whitelist_extra or []
            safety = SafetyKernel(
                allowed_paths=allowed_paths,
                whitelist_extra=extra_cmds,
            )

            # Bug 11 fix: the original closure captured `update` at orchestrator
            # creation time.  If the user sent a second message while the first
            # was still running, _stream_callback would reply to the *old*
            # update object, sending responses to the wrong message context.
            #
            # Fix: store the latest update on the TelegramInterface instance
            # (keyed by chat_id) and read it inside the callback so it always
            # refers to the most-recently-active update for that chat.
            self._latest_update: dict[int, Update] = getattr(
                self, "_latest_update", {}
            )
            self._latest_update[chat_id] = update

            def _stream_callback(resp: AgentResponse) -> None:
                # Read the current update at callback-invocation time, not at
                # orchestrator-creation time, so concurrent messages are safe.
                current_update = self._latest_update.get(chat_id)
                if current_update is not None:
                    asyncio.create_task(
                        self._send_response(chat_id, resp, current_update)
                    )

            bus = ToolBus(
                registry=global_registry,
                safety_kernel=safety,
                timeout_seconds=self._settings.tools.terminal.default_timeout_seconds,
            )

            # Skill registry + SkillBus for this chat's orchestrator
            _base = _SkillPath(__file__).parent.parent
            _skill_registry = _SkillLoader().load_all(
                [
                    _base / "skills" / "builtin",
                    _base / "skills" / "plugins",
                ],
                strict=True,   # production: raise on broken skill files
            )
            _skill_bus = _SkillBus(
                registry=_skill_registry,
                safety_kernel=safety,
                default_timeout_seconds=self._settings.tools.terminal.default_timeout_seconds,
            )

            self._orchestrators[chat_id] = Orchestrator.from_settings(
                settings=self._settings,
                llm_client=llm_client,
                tool_bus=_skill_bus,
                tool_registry=_skill_registry,
                memory_manager=self._memory,
                on_response=_stream_callback,
            )

        # Always refresh the latest update for this chat so the stream callback
        # uses the right reply context for subsequent messages.
        if hasattr(self, "_latest_update"):
            self._latest_update[chat_id] = update

        return self._orchestrators[chat_id]

    # ── Commands ──────────────────────────────────────────────────────────────

    async def _cmd_start(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        await update.message.reply_text(
            "🤖 *NeuralClaw is ready.*\n\n"
            "Send any message to chat with the agent, or use:\n"
            "/ask — chat\n"
            "/run — autonomous multi-step task\n"
            "/help — full command list",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_help(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        await update.message.reply_text(
            "🤖 *NeuralClaw Commands*\n\n"
            "/ask `<message>` — Chat with the agent\n"
            "/run `<goal>` — Autonomous task execution\n"
            "/status — Session stats\n"
            "/memory `<query>` — Search long-term memory\n"
            "/tools — List registered tools\n"
            "/trust `low|medium|high` — Set trust level\n"
            "/cancel — Cancel running task\n"
            "/clear — Clear conversation history\n"
            "/help — This message\n\n"
            "_Just type normally to chat — no /ask needed._",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_ask(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        message = " ".join(ctx.args) if ctx.args else ""
        if not message:
            await update.message.reply_text("Usage: /ask <your message>")
            return
        await self._run_ask(update, message)

    async def _on_text(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle plain text messages as /ask."""
        if not await self._auth_check(update):
            return
        text = update.message.text or ""
        if text:
            await self._run_ask(update, text)

    async def _run_ask(self, update: Update, message: str) -> None:
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        orc = self._get_orchestrator(chat_id, update)

        thinking_msg = await update.message.reply_text("⏳ Thinking…")

        turn_result = await orc.run_turn(session, message)

        try:
            await thinking_msg.delete()
        except Exception as _del_err:
            log.debug("telegram.delete_thinking_msg_failed", error=str(_del_err))

        await self._send_response(chat_id, turn_result.response, update)

    async def _cmd_run(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        goal = " ".join(ctx.args) if ctx.args else ""
        if not goal:
            await update.message.reply_text("Usage: /run <goal description>")
            return

        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        orc = self._get_orchestrator(chat_id, update)

        await update.message.reply_text(
            f"🚀 *Starting autonomous task:*\n_{goal}_",
            parse_mode=ParseMode.MARKDOWN,
        )

        try:
            async for resp in orc.run_autonomous(session, goal):
                await self._send_response(chat_id, resp, update)
        except asyncio.CancelledError:
            await update.message.reply_text("🛑 Task cancelled.")

    async def _cmd_status(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        s = session.status_summary()
        plan_line = ""
        if session.active_plan:
            plan_line = f"\n📋 Plan: {session.active_plan.progress_summary}"
        text = (
            f"🤖 *NeuralClaw Status*\n\n"
            f"Session: `{s['session_id']}`\n"
            f"Trust: *{s['trust_level'].upper()}*\n"
            f"Turns: {s['turns']}  ·  Tools: {s['tool_calls']}\n"
            f"Tokens: {s['tokens_in']:,} in / {s['tokens_out']:,} out\n"
            f"Uptime: {s['uptime_seconds']}s"
            f"{plan_line}"
        )
        await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)

    async def _cmd_memory(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        query = " ".join(ctx.args) if ctx.args else ""
        if not query:
            await update.message.reply_text("Usage: /memory <search query>")
            return

        thinking = await update.message.reply_text("🔍 Searching memory…")
        results = await self._memory.search_all(query, n_per_collection=3)
        try:
            await thinking.delete()
        except Exception as _del_err:
            log.debug("telegram.delete_thinking_msg_failed", error=str(_del_err))

        if not results:
            await update.message.reply_text("No relevant memories found.")
            return

        lines = [f"🧠 *Memory search: \"{query}\"*\n"]
        for collection, entries in results.items():
            lines.append(f"*{collection}*")
            for entry in entries:
                score = f"{entry.relevance_score:.2f}"
                lines.append(f"  [{score}] {entry.text[:150]}")
        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_tools(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        schemas = global_registry.list_schemas(enabled_only=True)
        if not schemas:
            await update.message.reply_text("No tools registered.")
            return

        lines = ["🔧 *Registered Tools*\n"]
        for s in sorted(schemas, key=lambda x: x.name):
            risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🟠", "CRITICAL": "🔴"}.get(
                s.risk_level.value, "⚪"
            )
            lines.append(f"{risk_icon} `{s.name}` — {s.description[:80]}")

        await update.message.reply_text(
            "\n".join(lines), parse_mode=ParseMode.MARKDOWN
        )

    async def _cmd_trust(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        level = (ctx.args[0] if ctx.args else "").lower()
        valid = {t.value: t for t in TrustLevel}
        if level not in valid:
            await update.message.reply_text(
                f"Invalid trust level. Use: {', '.join(valid.keys())}"
            )
            return

        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)

        if valid[level] == TrustLevel.HIGH:
            # Require explicit re-confirmation via inline keyboard
            keyboard = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("✅ Yes, set HIGH trust", callback_data="trust_confirm_high"),
                    InlineKeyboardButton("❌ Cancel", callback_data="trust_cancel"),
                ]
            ])
            await update.message.reply_text(
                "⚠️ *HIGH trust auto-approves ALL actions.*\n"
                "The agent will execute HIGH and CRITICAL risk commands without asking.\n\n"
                "Are you sure?",
                reply_markup=keyboard,
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        session.set_trust_level(valid[level])
        await update.message.reply_text(
            f"✅ Trust level set to *{level.upper()}*",
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _cmd_cancel(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        session.cancel()
        await update.message.reply_text("🛑 Cancel signal sent.")

    async def _cmd_clear(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._auth_check(update):
            return
        chat_id = update.effective_chat.id
        session = self._get_session(chat_id)
        session.clear_conversation()
        await update.message.reply_text("✓ Conversation history cleared.")

    async def _cmd_model(self, update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        """Show the inline model picker with live Ollama metadata."""
        if not await self._auth_check(update):
            return
        chat_id     = update.effective_chat.id
        current_key = self._active_models.get(chat_id, current_model_key(self._settings))

        # Show "Loading…" message immediately, then update with live data
        loading_msg = await update.message.reply_text("🔄 _Fetching model list…_", parse_mode="Markdown")

        try:
            # Fetch live Ollama data in parallel with building the keyboard
            keyboard = await build_telegram_model_keyboard_async(self._settings, current_key)
            text     = format_telegram_model_list(self._settings, current_key)
            await loading_msg.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)
        except Exception:
            # Fallback to sync (no live Ollama data)
            text     = format_telegram_model_list(self._settings, current_key)
            keyboard = build_telegram_model_keyboard(self._settings, current_key)
            await loading_msg.edit_text(text, parse_mode="Markdown", reply_markup=keyboard)



    async def _on_confirm_callback(
        self, update: Update, ctx: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle inline button presses for tool confirmation, trust change, or model selection."""
        query = update.callback_query
        await query.answer()

        data = query.data or ""
        chat_id = update.effective_chat.id

        # ── Model selection ───────────────────────────────────────────────────

        if data == "model_cancel":
            await query.edit_message_text("Model unchanged.")
            return

        if data.startswith("model_unavailable:"):
            model_key = data.split(":", 1)[1]
            await query.answer(
                f"⚠ {model_key} is unavailable (missing API key).",
                show_alert=True,
            )
            return

        if data == "model_set_default":
            last_key = self._pending_default.get(chat_id)
            if last_key:
                save_default_model(last_key)
                await query.edit_message_text(
                    f"★ Default model saved: `{last_key}`",
                    parse_mode=ParseMode.MARKDOWN,
                )
            else:
                await query.edit_message_text(
                    "Select a model first, then tap *Set as Default*.",
                    parse_mode=ParseMode.MARKDOWN,
                )
            return

        if data.startswith("model_select:"):
            model_key = data.split(":", 1)[1]
            opt = next((o for o in MODEL_OPTIONS if o.key == model_key), None)

            if opt is None:
                # allow dynamic models like ollama:qwen2.5:3b
                if ":" in model_key:
                    provider, model_id = model_key.split(":", 1)
                    from interfaces.model_selector import ModelOption
                    opt = ModelOption(
                        key=model_key,
                        name=model_id,
                        description="Dynamic model",
                        provider=provider,
                        model_id=model_id,
                        requires_key="",
                    )
                else:
                    await query.edit_message_text(f"⚠ Unknown model: `{model_key}`", parse_mode="Markdown")
                    return

            new_client, err = build_llm_client_for_model(opt, self._settings)
            if err or new_client is None:
                await query.edit_message_text(
                    f"❌ Could not switch to `{opt.name}`:\n_{err or 'unknown error'}_",
                    parse_mode=ParseMode.MARKDOWN,
                )
                return

            orch = self._orchestrators.get(chat_id)
            if orch:
                try:
                    orch.swap_llm_client(new_client, new_model_id=opt.model_id)
                except AttributeError:
                    orch._llm = new_client
                    if hasattr(orch, "_config"):
                        orch._config.model = opt.model_id
                    if hasattr(orch, "_planner") and hasattr(orch._planner, "_config"):
                        orch._planner._config.model = opt.model_id
                    if hasattr(orch, "_reasoner") and hasattr(orch._reasoner, "_config"):
                        orch._reasoner._config.model = opt.model_id

            self._active_models[chat_id] = model_key
            self._pending_default[chat_id] = model_key

            # Build capability notice for the confirmation message
            cap_note = ""
            try:
                from brain.capabilities import get_capabilities
                caps = get_capabilities(opt.provider, opt.model_id)
                if not caps.supports_tools:
                    cap_note = "\n⚠️ _Chat-only mode — tools not supported by this model._"
                else:
                    cap_note = "\n✅ _Tool calling: enabled_"
            except Exception as _cap_err:
                log.debug("telegram.capability_check_failed", error=str(_cap_err))

            await query.edit_message_text(
                f"✅ Switched to *{opt.name}*\n"
                f"_Provider: {opt.provider} · session only_{cap_note}\n\n"
                f"Use /model → *Set as Default* to persist across restarts.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        # ── Trust HIGH confirmation ───────────────────────────────────────────

        if data == "trust_confirm_high":
            session = self._get_session(chat_id)
            session.set_trust_level(TrustLevel.HIGH)
            await query.edit_message_text("🔴 Trust level set to *HIGH*.", parse_mode=ParseMode.MARKDOWN)
            return

        if data == "trust_cancel":
            await query.edit_message_text("Trust level unchanged.")
            return

        # ── Tool confirmation: "confirm_<tool_call_id>_yes/no" ───────────────

        if data.startswith("confirm_"):
            parts = data.split("_", 2)
            if len(parts) == 3:
                _, tool_call_id, answer = parts
                session = self._get_session(chat_id)
                approved = answer == "yes"
                session.resolve_confirmation(tool_call_id, approved)
                status = "✅ Approved" if approved else "❌ Denied"
                await query.edit_message_text(f"{status}.")



    # ── Response rendering ────────────────────────────────────────────────────

    async def _send_response(
        self,
        chat_id: int,
        response: AgentResponse,
        update: Update,
    ) -> None:
        """Convert AgentResponse → Telegram message."""
        kind = response.kind

        if kind == ResponseKind.CONFIRMATION:
            await self._send_confirmation(chat_id, response, update)
            return

        if kind == ResponseKind.PROGRESS:
            # Send short progress updates as plain messages
            await self._safe_send(chat_id, f"⏳ {response.text}", update)
            return

        if kind == ResponseKind.ERROR:
            await self._safe_send(
                chat_id,
                f"❌ *Error*\n{html.escape(response.text)}",
                update,
                parse_mode=ParseMode.MARKDOWN,
            )
            return

        if kind == ResponseKind.PLAN:
            await self._safe_send(
                chat_id, f"📋 {response.text}", update, parse_mode=ParseMode.MARKDOWN
            )
            return

        # TEXT, TOOL_RESULT, STATUS, etc.
        if response.text:
            await self._safe_send(
                chat_id, response.text, update, parse_mode=ParseMode.MARKDOWN
            )

    async def _send_confirmation(
        self, chat_id: int, response: AgentResponse, update: Update
    ) -> None:
        """Send an inline keyboard for tool confirmation."""
        tool_call_id = response.tool_call_id or "unknown"
        keyboard = InlineKeyboardMarkup([
            [
                InlineKeyboardButton(
                    "✅ Allow", callback_data=f"confirm_{tool_call_id}_yes"
                ),
                InlineKeyboardButton(
                    "❌ Deny", callback_data=f"confirm_{tool_call_id}_no"
                ),
            ]
        ])
        await self._safe_send(
            chat_id,
            response.text,
            update,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )

    async def _safe_send(
        self,
        chat_id: int,
        text: str,
        update: Update,
        parse_mode: Optional[str] = None,
        reply_markup=None,
    ) -> None:
        """Send a message, splitting if over Telegram's limit."""
        chunks = _split_message(text)
        bot = update.get_bot()
        for i, chunk in enumerate(chunks):
            try:
                await bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    parse_mode=parse_mode,
                    reply_markup=reply_markup if i == len(chunks) - 1 else None,
                )
            except Exception as e:
                log.warning("telegram.send_failed", error=str(e), chat_id=chat_id)
                # Fallback: send as plain text
                try:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=_strip_markdown(chunk)[:_MAX_MESSAGE_LEN],
                    )
                except Exception as _send_err:
                    log.warning("telegram.send_plaintext_fallback_failed", error=str(_send_err), chat_id=chat_id)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _split_message(text: str, max_len: int = _MAX_MESSAGE_LEN) -> list[str]:
    """Split long messages into chunks at newline boundaries."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    while len(text) > max_len:
        split_at = text.rfind("\n", 0, max_len)
        if split_at == -1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].lstrip("\n")
    if text:
        chunks.append(text)
    return chunks


def _strip_markdown(text: str) -> str:
    """Very light markdown stripping for fallback sends."""
    for ch in ("*", "_", "`", "[", "]"):
        text = text.replace(ch, "")
    return text


# ── Entry point ───────────────────────────────────────────────────────────────


async def run_telegram(settings: Settings, log) -> None:
    """Entry point called from main.py."""
    bot = TelegramBot(settings=settings)
    log.info("telegram_bot.starting")
    try:
        await bot.start()
    except KeyboardInterrupt:
        log.info("telegram_bot.interrupted")
    finally:
        await bot.stop()
        log.info("telegram_bot.stopped")