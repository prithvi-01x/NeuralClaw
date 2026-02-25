"""
skills/plugins/automation_report_render.py — Automation: Report Renderer

Renders a structured report to a Markdown file. Accepts a title, sections,
and optional metadata. This is the output sink used by meta-skills like
daily_assistant, repo_audit, and recon_pipeline.

Risk: LOW — fs:read + fs:write capabilities required.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_DEFAULT_REPORT_DIR = "~/neuralclaw/reports"
_MAX_SECTION_CONTENT_LEN = 50_000


class AutomationReportRenderSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="automation_report_render",
        version="1.0.0",
        description=(
            "Render a structured report to a Markdown (.md) file. "
            "Accepts a title, list of sections (heading + content), and optional metadata. "
            "Used by meta-skills to persist output as readable reports."
        ),
        category="automation",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read", "fs:write"}),
        requires_confirmation=False,
        timeout_seconds=15,
        parameters={
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Report title.",
                },
                "sections": {
                    "type": "array",
                    "description": "List of report sections.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "heading": {"type": "string", "description": "Section heading."},
                            "content": {"type": "string", "description": "Section body (Markdown supported)."},
                            "level": {"type": "integer", "description": "Heading level 1-4 (default 2).", "default": 2},
                        },
                        "required": ["heading", "content"],
                    },
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        f"Output file path. If omitted, saves to {_DEFAULT_REPORT_DIR}/<title>_<timestamp>.md"
                    ),
                    "default": "",
                },
                "metadata": {
                    "type": "object",
                    "description": "Optional key-value metadata added as a front-matter table.",
                    "default": {},
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite if file exists (default false — appends timestamp suffix).",
                    "default": False,
                },
            },
            "required": ["title", "sections"],
        },
    )

    async def validate(self, title: str, sections: list, **_) -> None:
        if not title or not title.strip():
            raise SkillValidationError("title must be non-empty.")
        if not sections:
            raise SkillValidationError("sections must contain at least one section.")
        for i, sec in enumerate(sections):
            if not isinstance(sec, dict):
                raise SkillValidationError(f"sections[{i}] must be an object with 'heading' and 'content'.")
            if not sec.get("heading"):
                raise SkillValidationError(f"sections[{i}] is missing 'heading'.")
            if "content" not in sec:
                raise SkillValidationError(f"sections[{i}] is missing 'content'.")

    async def execute(
        self,
        title: str,
        sections: list[dict],
        output_path: str = "",
        metadata: dict | None = None,
        overwrite: bool = False,
        **kwargs,
    ) -> SkillResult:
        call_id = kwargs.get("_skill_call_id", "")
        t_start = time.monotonic()
        metadata = metadata or {}

        try:
            now = datetime.now(tz=timezone.utc)
            timestamp = now.strftime("%Y%m%d_%H%M%S")
            timestamp_human = now.strftime("%Y-%m-%d %H:%M:%S UTC")

            # Resolve output path
            if output_path:
                out = Path(output_path).expanduser()
            else:
                report_dir = Path(_DEFAULT_REPORT_DIR).expanduser()
                safe_title = _safe_filename(title)
                out = report_dir / f"{safe_title}_{timestamp}.md"

            # Handle existing file
            if out.exists() and not overwrite:
                stem = out.stem
                out = out.with_name(f"{stem}_{timestamp}{out.suffix}")

            def _render() -> str:
                lines: list[str] = []

                # Title
                lines.append(f"# {title}")
                lines.append(f"*Generated: {timestamp_human}*")
                lines.append("")

                # Metadata table
                if metadata:
                    lines.append("## Metadata")
                    lines.append("")
                    lines.append("| Key | Value |")
                    lines.append("|-----|-------|")
                    for k, v in metadata.items():
                        lines.append(f"| {k} | {v} |")
                    lines.append("")

                # Sections
                for sec in sections:
                    level = max(1, min(4, int(sec.get("level", 2))))
                    heading = sec["heading"]
                    content = str(sec["content"])[:_MAX_SECTION_CONTENT_LEN]
                    lines.append(f"{'#' * level} {heading}")
                    lines.append("")
                    lines.append(content)
                    lines.append("")

                lines.append("---")
                lines.append(f"*NeuralClaw report · {timestamp_human}*")
                return "\n".join(lines)

            def _write(content: str) -> None:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(content, encoding="utf-8")

            loop = asyncio.get_event_loop()
            rendered = await loop.run_in_executor(None, _render)
            await loop.run_in_executor(None, _write, rendered)

            char_count = len(rendered)
            duration_ms = (time.monotonic() - t_start) * 1000
            return SkillResult.ok(
                skill_name=self.manifest.name,
                skill_call_id=call_id,
                output={
                    "output_path": str(out),
                    "title": title,
                    "sections": len(sections),
                    "char_count": char_count,
                    "generated_at": timestamp_human,
                },
                duration_ms=duration_ms,
            )

        except (OSError, PermissionError) as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )
        except BaseException as e:
            return SkillResult.fail(
                self.manifest.name, call_id,
                f"{type(e).__name__}: {e}", type(e).__name__,
                duration_ms=(time.monotonic() - t_start) * 1000,
            )


def _safe_filename(name: str) -> str:
    """Convert a title to a safe filename (lowercase, underscores, no special chars)."""
    import re
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    return name[:60]