"""
skills/plugins/meta_autonomous_research.py — Meta-Skill: Autonomous Research

Fixed multi-step research workflow. Orchestrates:
    web_search (builtin) → web_fetch (builtin) → data_summarize_doc →
    memory_write → automation_report_render

The agent searches for a topic, fetches the top result pages, summarizes each,
writes the consolidated findings to memory (ChromaDB), and produces a Markdown
research report. The orchestrator sees this as one plan step.

Risk: MEDIUM — net:fetch + fs:read + fs:write + data:write capabilities.
No confirmation required (no system-altering actions).

Phase C meta-skill. No kernel changes required.

Note on web_search / web_fetch:
    These are builtin skills (skills/builtin/) and are registered in the same
    registry as plugin skills. They are dispatched through the SkillBus like
    any other skill.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, ClassVar

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _truncate(text: Any, max_len: int = 2000) -> str:
    s = str(text) if not isinstance(text, str) else text
    return s[:max_len] + "…" if len(s) > max_len else s


def _result_summary(result: SkillResult) -> str:
    if not result.success:
        return f"❌ FAILED — {result.error}"
    output = result.output
    if isinstance(output, str):
        return f"✅ {_truncate(output, 120)}"
    if isinstance(output, dict):
        return f"✅ OK — {list(output.keys())}"
    if isinstance(output, list):
        return f"✅ {len(output)} items"
    return "✅ OK"


def _skill_call(name: str, args: dict) -> Any:
    from skills.types import SkillCall
    return SkillCall(
        id=f"meta-research-{uuid.uuid4().hex[:8]}",
        skill_name=name,
        arguments=args,
    )


async def _dispatch(bus: Any, call: Any, session: Any) -> SkillResult:
    return await bus.dispatch(
        call,
        trust_level=session.trust_level if session else None,
        granted_capabilities=session.granted_capabilities if session else frozenset(),
    )


def _output_to_md(output: Any, max_len: int = 4000) -> str:
    if isinstance(output, str):
        return _truncate(output, max_len)
    import json
    try:
        return f"```json\n{json.dumps(output, indent=2, default=str)[:max_len]}\n```"
    except Exception:
        return str(output)[:max_len]


def _extract_urls(search_output: Any) -> list[str]:
    """Extract URLs from web_search output (list of results or dict)."""
    urls: list[str] = []
    if isinstance(search_output, list):
        for item in search_output:
            if isinstance(item, dict):
                url = item.get("url") or item.get("link") or item.get("href")
                if url:
                    urls.append(url)
    elif isinstance(search_output, dict):
        results = search_output.get("results", [])
        for item in results:
            if isinstance(item, dict):
                url = item.get("url") or item.get("link") or item.get("href")
                if url:
                    urls.append(url)
    return urls


# ─────────────────────────────────────────────────────────────────────────────
# Meta-Skill
# ─────────────────────────────────────────────────────────────────────────────

class MetaAutonomousResearchSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="meta_autonomous_research",
        version="1.0.0",
        description=(
            "Autonomous multi-source research pipeline. "
            "Searches the web for a topic, fetches and summarizes top result pages, "
            "consolidates findings into memory (ChromaDB), and writes a Markdown research report. "
            "Use for deep-dive research tasks where you need synthesized findings across multiple sources. "
            "Provide a clear 'topic' and optional 'focus_questions' to guide the research."
        ),
        category="data",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"net:fetch", "fs:read", "fs:write", "data:write"}),
        requires_confirmation=False,
        timeout_seconds=240,
        parameters={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Research topic or search query (e.g. 'WebGPU compute shaders performance 2024').",
                },
                "focus_questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional specific questions to answer during research.",
                    "default": [],
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Maximum number of web sources to fetch and summarize. Default 3.",
                    "default": 3,
                },
                "memory_key": {
                    "type": "string",
                    "description": "Key under which to store findings in long-term memory. Defaults to a slug of the topic.",
                    "default": "",
                },
                "report_path": {
                    "type": "string",
                    "description": "Output path for the research Markdown report.",
                    "default": "~/neuralclaw/reports/research.md",
                },
                "skip_steps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Steps to skip: search, fetch, summarize, memory_write.",
                    "default": [],
                },
                "plan_id": {
                    "type": "string",
                    "description": "TaskMemory plan ID (injected by orchestrator).",
                    "default": "",
                },
            },
            "required": ["topic"],
        },
    )

    async def validate(self, topic: str = "", **kwargs) -> None:
        if not topic or not topic.strip():
            raise SkillValidationError("meta_autonomous_research: 'topic' must be a non-empty string.")
        if len(topic) > 500:
            raise SkillValidationError("meta_autonomous_research: 'topic' must be under 500 characters.")

    async def execute(
        self,
        topic: str,
        focus_questions: list[str] | None = None,
        max_sources: int = 3,
        memory_key: str = "",
        report_path: str = "~/neuralclaw/reports/research.md",
        skip_steps: list[str] | None = None,
        plan_id: str = "",
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        skip = set(skip_steps or [])
        plan_id = plan_id or f"research-{uuid.uuid4().hex[:8]}"
        max_sources = max(1, min(max_sources, 10))  # clamp 1-10

        # Generate memory key from topic
        if not memory_key:
            memory_key = "research:" + topic[:60].lower().replace(" ", "_").replace("/", "_")

        bus = kwargs.get("_bus")
        session = kwargs.get("_session")
        task_store = kwargs.get("_task_store")

        if not bus:
            return SkillResult.fail(
                self.manifest.name, call_id,
                "meta_autonomous_research requires '_bus' injected.",
                "MetaSkillConfigError",
            )

        if task_store:
            task_store.create(plan_id=plan_id, goal=f"Research: {topic}")

        def log_step(step_id: str, desc: str) -> None:
            if task_store:
                task_store.log_step(plan_id, step_id, desc)

        def log_result(step_id: str, result: SkillResult) -> None:
            if task_store:
                task_store.update_result(
                    plan_id, step_id,
                    result_content=_result_summary(result),
                    is_error=not result.success,
                    duration_ms=result.duration_ms,
                )

        step_results: dict[str, SkillResult] = {}
        source_summaries: list[dict[str, str]] = []

        # Step 1: Web Search
        search_result: SkillResult | None = None
        if "search" not in skip:
            log_step("search", f"Searching web for: {topic}")
            search_result = await _dispatch(bus, _skill_call("web_search", {
                "query": topic,
                "num_results": max_sources + 2,  # fetch a few extra in case some fail
            }), session)
            log_result("search", search_result)
            step_results["search"] = search_result

        # Step 2: Fetch each URL
        urls: list[str] = []
        if search_result and search_result.success:
            urls = _extract_urls(search_result.output)[:max_sources]

        fetch_results: list[SkillResult] = []
        if "fetch" not in skip:
            for i, url in enumerate(urls):
                step_id = f"fetch_{i}"
                log_step(step_id, f"Fetching source {i+1}/{len(urls)}: {url[:80]}")
                r = await _dispatch(bus, _skill_call("web_fetch", {
                    "url": url,
                }), session)
                log_result(step_id, r)
                step_results[step_id] = r
                if r.success:
                    fetch_results.append(r)

        # Step 3: Summarize each fetched document
        if "summarize" not in skip:
            for i, fetch_r in enumerate(fetch_results):
                step_id = f"summarize_{i}"
                url = urls[i] if i < len(urls) else f"source_{i}"
                content = fetch_r.output if isinstance(fetch_r.output, str) else str(fetch_r.output)
                log_step(step_id, f"Summarizing source {i+1}: {url[:60]}")
                r = await _dispatch(bus, _skill_call("data_summarize_doc", {
                    "content": content[:20000],  # cap input to summarizer
                    "focus": focus_questions[0] if focus_questions else topic,
                }), session)
                log_result(step_id, r)
                step_results[step_id] = r
                summary_text = r.output if isinstance(r.output, str) else str(r.output) if r.success else f"(summarization failed: {r.error})"
                source_summaries.append({"url": url, "summary": summary_text})

        # Step 4: Write consolidated findings to memory
        if "memory_write" not in skip and source_summaries:
            consolidated = f"Research topic: {topic}\n\n"
            if focus_questions:
                consolidated += "Focus questions:\n" + "\n".join(f"- {q}" for q in focus_questions) + "\n\n"
            for i, s in enumerate(source_summaries, 1):
                consolidated += f"Source {i}: {s['url']}\n{s['summary']}\n\n"

            log_step("memory_write", f"Writing findings to memory under key: {memory_key}")
            r = await _dispatch(bus, _skill_call("memory_write", {
                "content": consolidated,
                "key": memory_key,
                "collection": "research",
            }), session)
            log_result("memory_write", r)
            step_results["memory_write"] = r

        # ── Build report sections ────────────────────────────────────────────
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).strftime("%A, %d %B %Y %H:%M UTC")

        sections: list[dict] = [
            {
                "heading": "Research Brief",
                "content": (
                    f"**Topic:** {topic}\n\n"
                    f"**Run at:** {now}\n\n"
                    f"**Sources fetched:** {len(source_summaries)}/{len(urls)}\n\n"
                    + (f"**Focus questions:**\n" + "\n".join(f"- {q}" for q in focus_questions) if focus_questions else "")
                ),
                "level": 2,
            }
        ]

        for i, s in enumerate(source_summaries, 1):
            sections.append({
                "heading": f"Source {i}",
                "content": f"**URL:** {s['url']}\n\n{s['summary']}",
                "level": 3,
            })

        if not source_summaries:
            sections.append({
                "heading": "Findings",
                "content": "_No sources were successfully fetched and summarized._",
                "level": 2,
            })

        if "memory_write" in step_results:
            mw = step_results["memory_write"]
            sections.append({
                "heading": "Memory",
                "content": f"**Key:** `{memory_key}`\n\n**Status:** {_result_summary(mw)}",
                "level": 2,
            })

        # ── Write report ─────────────────────────────────────────────────────
        log_step("report_render", "Writing research report")
        report_result = await _dispatch(bus, _skill_call("automation_report_render", {
            "title": f"Research: {topic}",
            "sections": sections,
            "output_path": report_path,
        }), session)
        log_result("report_render", report_result)

        if task_store:
            task_store.close(plan_id)

        duration_ms = (time.monotonic() - t_start) * 1000
        success_count = sum(1 for r in step_results.values() if r.success)

        return SkillResult.ok(
            skill_name=self.manifest.name,
            skill_call_id=call_id,
            output={
                "topic": topic,
                "sources_found": len(urls),
                "sources_summarized": len(source_summaries),
                "memory_key": memory_key,
                "steps_completed": success_count,
                "steps_total": len(step_results),
                "steps_skipped": list(skip),
                "report_path": report_path,
                "report_written": report_result.success,
                "summary": {k: _result_summary(v) for k, v in step_results.items()},
            },
            duration_ms=duration_ms,
        )