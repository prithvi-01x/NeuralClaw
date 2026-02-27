"""
skills/clawhub/bridge_executor.py — Three-Tier Execution for ClawHub Skills

Tier 1 (PromptOnlyExecutor):  Inject skill body into LLM system prompt.
Tier 2 (HttpApiExecutor):     Instruct LLM to produce HTTP spec, execute via httpx.
Tier 3 (BinaryExecutor):      Route through terminal_exec with safety checks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TYPE_CHECKING

from neuralclaw.skills.types import SkillResult

if TYPE_CHECKING:
    from neuralclaw.skills.clawhub.bridge_parser import ClawhubSkillManifest

from neuralclaw.observability.compat import get_safe_logger, safe_log

_log_raw = get_safe_logger(__name__)


def _log(level: str, event: str, **kwargs) -> None:
    safe_log(_log_raw, level, event, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────────────

class BridgeExecutor(ABC):
    """Base class for tier-specific ClawHub skill executors."""

    @abstractmethod
    async def run(
        self,
        manifest: "ClawhubSkillManifest",
        kwargs: dict[str, Any],
    ) -> SkillResult:
        """
        Execute the ClawHub skill.

        Args:
            manifest: Parsed ClawHub skill manifest (contains body, requires, etc.)
            kwargs: Skill call arguments — always includes 'request' (str).

        Returns:
            SkillResult
        """
        ...


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 — Prompt Only
# ─────────────────────────────────────────────────────────────────────────────

class PromptOnlyExecutor(BridgeExecutor):
    """
    Tier 1 executor for prompt-only ClawHub skills.

    Injects the skill's SKILL.md body as a system prompt, sends the user's
    request, and returns the LLM's response.
    """

    def __init__(self, llm_client=None, llm_config=None):
        self._llm_client = llm_client
        self._llm_config = llm_config

    async def run(
        self,
        manifest: "ClawhubSkillManifest",
        kwargs: dict[str, Any],
    ) -> SkillResult:
        request = kwargs.get("request", "")
        call_id = kwargs.get("_skill_call_id", "")
        skill_name = manifest.name

        if not request:
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error="No request provided",
                error_type="ValueError",
            )

        # If no LLM client, return the instructions as fallback
        if self._llm_client is None:
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output={
                    "type": "clawhub_prompt_skill",
                    "instructions": manifest.body,
                    "request": request,
                    "message": (
                        f"Skill '{skill_name}' is a ClawHub prompt skill. "
                        "Use the instructions above to handle the request."
                    ),
                },
            )

        # Call LLM with skill instructions as system prompt
        try:
            messages = [
                {"role": "system", "content": manifest.body},
                {"role": "user", "content": request},
            ]
            if self._llm_config is None:
                return SkillResult.fail(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    error=(
                        f"ClawHub skill '{skill_name}' requires an LLM client "
                        "but none was configured. This is a startup configuration error."
                    ),
                    error_type="ClawhubConfigError",
                )
            config = self._llm_config
            response = await self._llm_client.generate(
                messages=messages,
                config=config,
                tools=None,
            )
            text = response.text if hasattr(response, "text") else str(response)
            _log("info", "clawhub_exec.tier1.complete",
                 skill=skill_name, chars=len(text))
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output=text,
            )
        except Exception as e:
            _log("warning", "clawhub_exec.tier1.error",
                 skill=skill_name, error=str(e))
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error=f"LLM call failed: {e}",
                error_type=type(e).__name__,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 — HTTP API
# ─────────────────────────────────────────────────────────────────────────────

class HttpApiExecutor(BridgeExecutor):
    """
    Tier 2 executor for HTTP/API ClawHub skills.

    Asks the LLM to produce a JSON HTTP call spec instead of curl commands,
    then executes it via httpx.
    """

    def __init__(self, llm_client=None, llm_config=None):
        self._llm_client = llm_client
        self._llm_config = llm_config

    async def run(
        self,
        manifest: "ClawhubSkillManifest",
        kwargs: dict[str, Any],
    ) -> SkillResult:
        request = kwargs.get("request", "")
        call_id = kwargs.get("_skill_call_id", "")
        skill_name = manifest.name

        # Check env vars
        from neuralclaw.skills.clawhub.env_injector import validate_env, format_missing_env_message
        env_ok, missing = validate_env(manifest.requires)
        if not env_ok:
            msg = format_missing_env_message(skill_name, missing, manifest.primary_env)
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error=msg,
                error_type="MissingEnvVarError",
            )

        if not request:
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error="No request provided",
                error_type="ValueError",
            )

        # If no LLM client, return instructions with env info
        if self._llm_client is None:
            from neuralclaw.skills.clawhub.env_injector import get_env_values
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output={
                    "type": "clawhub_http_skill",
                    "instructions": manifest.body,
                    "request": request,
                    "env_available": list(get_env_values(manifest.requires).keys()),
                    "message": (
                        f"Skill '{skill_name}' is a ClawHub HTTP/API skill. "
                        "Use the instructions and available env vars to make "
                        "HTTP calls via the web_fetch tool."
                    ),
                },
            )

        # Call LLM with instructions to produce JSON HTTP spec
        try:
            augmented_body = (
                manifest.body
                + "\n\nIMPORTANT: Do not use curl or shell commands. "
                "Instead, output a JSON object with: "
                "{\"method\": \"GET|POST|...\", \"url\": \"...\", "
                "\"headers\": {...}, \"body\": ...} for the HTTP call to make. "
                "Replace environment variable references with their actual values."
            )

            # Inject env var values into the prompt
            from neuralclaw.skills.clawhub.env_injector import get_env_values
            env_vals = get_env_values(manifest.requires)
            if env_vals:
                env_block = "\n".join(f"{k}={v}" for k, v in env_vals.items())
                augmented_body += f"\n\nAvailable environment variables:\n{env_block}"

            messages = [
                {"role": "system", "content": augmented_body},
                {"role": "user", "content": request},
            ]
            if self._llm_config is None:
                return SkillResult.fail(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    error=(
                        f"ClawHub skill '{skill_name}' requires an LLM client "
                        "but none was configured. This is a startup configuration error."
                    ),
                    error_type="ClawhubConfigError",
                )
            config = self._llm_config
            response = await self._llm_client.generate(
                messages=messages,
                config=config,
                tools=None,
            )
            text = response.text if hasattr(response, "text") else str(response)

            # Try to parse and execute the HTTP call
            http_result = await self._execute_http(text, skill_name)
            if http_result is not None:
                # Feed HTTP result back to LLM for final answer
                messages.append({"role": "assistant", "content": text})
                messages.append({"role": "user", "content": (
                    f"HTTP Response:\n{http_result}\n\n"
                    "Based on this response, provide the final answer."
                )})
                final = await self._llm_client.generate(
                    messages=messages, config=config, tools=None,
                )
                final_text = final.text if hasattr(final, "text") else str(final)
                return SkillResult.ok(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    output=final_text,
                )

            # If we couldn't parse HTTP spec, return the raw response
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output=text,
            )
        except Exception as e:
            _log("warning", "clawhub_exec.tier2.error",
                 skill=skill_name, error=str(e))
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error=f"HTTP execution failed: {e}",
                error_type=type(e).__name__,
            )

    async def _execute_http(self, llm_text: str, skill_name: str) -> str | None:
        """Try to parse a JSON HTTP spec from LLM output and execute it."""
        import json
        try:
            # Try to extract JSON from the response
            text = llm_text.strip()
            # Handle markdown code blocks
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            spec = json.loads(text)
            if not isinstance(spec, dict) or "url" not in spec:
                return None

            method = spec.get("method", "GET").upper()
            url = spec["url"]
            headers = spec.get("headers", {})
            body = spec.get("body")

            import httpx
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    json=body if isinstance(body, (dict, list)) else None,
                    content=body if isinstance(body, str) else None,
                )
                result = f"Status: {resp.status_code}\n{resp.text[:4000]}"
                _log("info", "clawhub_exec.tier2.http_ok",
                     skill=skill_name, status=resp.status_code)
                return result

        except (json.JSONDecodeError, KeyError, TypeError):
            return None
        except Exception as e:
            _log("warning", "clawhub_exec.tier2.http_error",
                 skill=skill_name, error=str(e))
            return f"HTTP Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tier 3 — Binary Execution
# ─────────────────────────────────────────────────────────────────────────────

class BinaryExecutor(BridgeExecutor):
    """
    Tier 3 executor for ClawHub skills that need CLI binaries.

    Routes through NeuralClaw's terminal_exec built-in skill, going through
    the full SafetyKernel pipeline.
    """

    def __init__(self, llm_client=None, llm_config=None, skill_bus=None,
                 auto_install: bool = False):
        self._llm_client = llm_client
        self._llm_config = llm_config
        self._skill_bus = skill_bus
        self._auto_install = auto_install

    async def run(
        self,
        manifest: "ClawhubSkillManifest",
        kwargs: dict[str, Any],
    ) -> SkillResult:
        request = kwargs.get("request", "")
        call_id = kwargs.get("_skill_call_id", "")
        skill_name = manifest.name

        # Check env vars
        from neuralclaw.skills.clawhub.env_injector import validate_env, format_missing_env_message
        env_ok, missing_env = validate_env(manifest.requires)
        if not env_ok:
            msg = format_missing_env_message(skill_name, missing_env, manifest.primary_env)
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error=msg,
                error_type="MissingEnvVarError",
            )

        # Check required binaries
        from neuralclaw.skills.clawhub.dependency_checker import check_bins
        bins_ok, missing_bins = check_bins(manifest.requires)
        if not bins_ok:
            if self._auto_install and manifest.install_directives:
                # Try to auto-install missing deps
                from neuralclaw.skills.clawhub.dependency_checker import run_install_directive
                for directive in manifest.install_directives:
                    await run_install_directive(directive, self._skill_bus)
                # Re-check
                bins_ok, missing_bins = check_bins(manifest.requires)

            if not bins_ok:
                return SkillResult.fail(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    error=(
                        f"Skill '{skill_name}' requires binaries that are not installed: "
                        f"{', '.join(missing_bins)}. "
                        "Install them manually or set clawhub.execution.auto_install_deps: true"
                    ),
                    error_type="MissingBinaryError",
                )

        if not request:
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error="No request provided",
                error_type="ValueError",
            )

        # If no LLM client, return instructions with dependency info
        if self._llm_client is None:
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output={
                    "type": "clawhub_binary_skill",
                    "instructions": manifest.body,
                    "request": request,
                    "required_bins": manifest.requires.bins,
                    "message": (
                        f"Skill '{skill_name}' is a ClawHub binary skill. "
                        "Use terminal_exec to run the commands described in "
                        "the instructions above."
                    ),
                },
            )

        # Call LLM with instructions and terminal_exec tool
        try:
            augmented_body = (
                manifest.body
                + "\n\nIMPORTANT: Use shell commands to accomplish this. "
                "Output commands that should be run in the terminal. "
                "Available tools: terminal_exec."
            )
            messages = [
                {"role": "system", "content": augmented_body},
                {"role": "user", "content": request},
            ]
            if self._llm_config is None:
                return SkillResult.fail(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    error=(
                        f"ClawHub skill '{skill_name}' requires an LLM client "
                        "but none was configured. This is a startup configuration error."
                    ),
                    error_type="ClawhubConfigError",
                )
            config = self._llm_config

            # Define terminal_exec tool schema for the LLM
            tools = [
                {
                    "name": "terminal_exec",
                    "description": "Execute a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Shell command to execute",
                            },
                        },
                        "required": ["command"],
                    },
                },
            ]

            response = await self._llm_client.generate(
                messages=messages,
                config=config,
                tools=tools,
            )

            # If the LLM produced tool calls, route them through the SkillBus
            if hasattr(response, "tool_calls") and response.tool_calls and self._skill_bus:
                from neuralclaw.skills.types import SkillCall
                results = []
                for tc in response.tool_calls:
                    if tc.get("name") == "terminal_exec" or tc.get("function", {}).get("name") == "terminal_exec":
                        args = tc.get("arguments", tc.get("function", {}).get("arguments", {}))
                        if isinstance(args, str):
                            import json
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {"command": args}

                        term_call = SkillCall(
                            id=tc.get("id", call_id),
                            skill_name="terminal_exec",
                            arguments=args,
                        )
                        term_result = await self._skill_bus.dispatch(term_call)
                        results.append(term_result.to_llm_content())

                output = "\n---\n".join(results) if results else "No commands executed"
                return SkillResult.ok(
                    skill_name=skill_name,
                    skill_call_id=call_id,
                    output=output,
                )

            # If no tool calls, return the text response
            text = response.text if hasattr(response, "text") else str(response)
            return SkillResult.ok(
                skill_name=skill_name,
                skill_call_id=call_id,
                output=text,
            )

        except Exception as e:
            _log("warning", "clawhub_exec.tier3.error",
                 skill=skill_name, error=str(e))
            return SkillResult.fail(
                skill_name=skill_name,
                skill_call_id=call_id,
                error=f"Binary execution failed: {e}",
                error_type=type(e).__name__,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_executor(
    tier: int,
    llm_client=None,
    llm_config=None,
    skill_bus=None,
    auto_install: bool = False,
) -> BridgeExecutor:
    """Create the appropriate executor for the given execution tier."""
    if tier == 1:
        return PromptOnlyExecutor(llm_client=llm_client, llm_config=llm_config)
    elif tier == 2:
        return HttpApiExecutor(llm_client=llm_client, llm_config=llm_config)
    elif tier == 3:
        return BinaryExecutor(
            llm_client=llm_client,
            llm_config=llm_config,
            skill_bus=skill_bus,
            auto_install=auto_install,
        )
    else:
        # Default to Tier 1 for unknown tiers
        _log("warning", "clawhub_exec.unknown_tier", tier=tier)
        return PromptOnlyExecutor(llm_client=llm_client, llm_config=llm_config)
