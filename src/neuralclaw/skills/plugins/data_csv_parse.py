"""
skills/plugins/data_csv_parse.py — Data: CSV Parse

Reads and analyses a CSV file. Returns schema, row count, column stats, and preview rows.

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, csv, io, time
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DataCsvParseSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="data_csv_parse",
        version="1.0.0",
        description="Read and analyse a CSV file. Returns column names, row count, data types, null counts, and preview rows.",
        category="data",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "file_path":{"type":"string","description":"Path to the CSV file."},
            "delimiter":{"type":"string","description":"Field delimiter (default: auto-detect).","default":""},
            "preview_rows":{"type":"integer","description":"Number of preview rows to return (default 5).","default":5},
            "max_rows":{"type":"integer","description":"Max rows to analyse (default 10000).","default":10000},
        },"required":["file_path"]},
    )

    async def validate(self, file_path: str, **_) -> None:
        p = Path(file_path).expanduser()
        if not p.exists(): raise SkillValidationError(f"File does not exist: '{file_path}'")
        if p.suffix.lower() not in (".csv",".tsv",".txt"): raise SkillValidationError(f"Expected .csv/.tsv file, got: '{p.suffix}'")

    async def execute(self, file_path: str, delimiter: str="", preview_rows: int=5, max_rows: int=10000, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        preview_rows = min(int(preview_rows), 20)
        max_rows = min(int(max_rows), 100_000)

        def _analyse():
            p = Path(file_path).expanduser().resolve()
            raw = p.read_text(errors="replace")
            # Auto-detect delimiter
            delim = delimiter or (csv.Sniffer().sniff(raw[:4096]).delimiter if not delimiter else delimiter)
            reader = csv.DictReader(io.StringIO(raw), delimiter=delim)
            headers = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                if i >= max_rows: break
                rows.append(dict(row))
            total = len(rows)
            # Column stats
            col_stats = {}
            for col in headers:
                values = [r.get(col,"") for r in rows]
                nulls = sum(1 for v in values if v in ("","None","null","NULL","N/A","NA"))
                numerics = []
                for v in values:
                    try: numerics.append(float(v))
                    except (ValueError, TypeError): pass
                col_stats[col] = {"null_count":nulls,"null_pct":round(nulls/total*100,1) if total else 0,
                    "numeric":len(numerics)>total*0.8,"min":min(numerics) if numerics else None,"max":max(numerics) if numerics else None}
            return {"file":str(p),"row_count":total,"column_count":len(headers),"columns":headers,
                    "delimiter":delim,"col_stats":col_stats,"preview":rows[:preview_rows]}

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _analyse)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, result, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
