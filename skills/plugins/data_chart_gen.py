"""
skills/plugins/data_chart_gen.py — Data: Chart Generator

Generates a chart image (PNG) from data using matplotlib.
Supports bar, line, pie, scatter, histogram.

Risk: LOW — fs:write
"""
from __future__ import annotations
import asyncio, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DataChartGenSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="data_chart_gen",
        version="1.0.0",
        description="Generate a chart image (PNG) from provided data. Supports bar, line, pie, scatter, and histogram charts. Requires matplotlib.",
        category="data",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:write"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "chart_type":{"type":"string","enum":["bar","line","pie","scatter","histogram"],"description":"Type of chart to generate."},
            "labels":{"type":"array","items":{"type":"string"},"description":"X-axis labels or pie slice labels.","default":[]},
            "values":{"type":"array","items":{"type":"number"},"description":"Y-axis values or pie sizes.","default":[]},
            "x_values":{"type":"array","items":{"type":"number"},"description":"X values for scatter chart.","default":[]},
            "title":{"type":"string","description":"Chart title.","default":""},
            "x_label":{"type":"string","description":"X-axis label.","default":""},
            "y_label":{"type":"string","description":"Y-axis label.","default":""},
            "output_path":{"type":"string","description":"Where to save the PNG (default: ~/neuralclaw/reports/chart_<ts>.png).","default":""},
        },"required":["chart_type","values"]},
    )

    async def validate(self, chart_type: str, values: list, labels: list|None=None, x_values: list|None=None, **_) -> None:
        if not values: raise SkillValidationError("values must be non-empty.")
        if chart_type == "scatter" and not x_values: raise SkillValidationError("x_values required for scatter chart.")

    async def execute(self, chart_type: str, values: list, labels: list|None=None, x_values: list|None=None,
                      title: str="", x_label: str="", y_label: str="", output_path: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        labels = labels or [str(i) for i in range(len(values))]

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return SkillResult.fail(self.manifest.name, call_id, "matplotlib not installed. Run: pip install matplotlib", "ImportError")

        def _generate():
            fig, ax = plt.subplots(figsize=(10,6))
            if chart_type == "bar":
                ax.bar(labels, values)
            elif chart_type == "line":
                ax.plot(labels, values, marker="o")
            elif chart_type == "pie":
                ax.pie(values, labels=labels, autopct="%1.1f%%")
            elif chart_type == "scatter":
                ax.scatter(x_values or list(range(len(values))), values)
            elif chart_type == "histogram":
                ax.hist(values, bins="auto")
            if title: ax.set_title(title)
            if x_label: ax.set_xlabel(x_label)
            if y_label: ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = Path(output_path).expanduser() if output_path else Path(f"~/neuralclaw/reports/chart_{ts}.png").expanduser()
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(str(out), dpi=150, bbox_inches="tight")
            plt.close(fig)
            return str(out)

        try:
            loop = asyncio.get_event_loop()
            out_path = await loop.run_in_executor(None, _generate)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"output_path":out_path,"chart_type":chart_type,"data_points":len(values)}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
