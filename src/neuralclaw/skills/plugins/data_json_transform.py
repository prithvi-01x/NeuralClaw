"""
skills/plugins/data_json_transform.py — Data: JSON Transform

Read, query, filter, and transform JSON/JSONL files using JMESPath expressions.

Risk: LOW — fs:read, fs:write
"""
from __future__ import annotations
import asyncio, json, time
from pathlib import Path
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DataJsonTransformSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="data_json_transform",
        version="1.0.0",
        description="Read, query, and transform JSON or JSONL files. Supports JMESPath queries, key extraction, filtering, and writing transformed output.",
        category="data",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read","fs:write"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "file_path":{"type":"string","description":"Path to input JSON or JSONL file."},
            "action":{"type":"string","enum":["read","query","keys","write"],"description":"Action: read (return as-is), query (JMESPath), keys (list top-level keys), write (save transformed).","default":"read"},
            "jmespath_query":{"type":"string","description":"JMESPath expression for query action (e.g. 'items[?status==`active`].name').","default":""},
            "output_path":{"type":"string","description":"Output file path for write action.","default":""},
            "max_chars":{"type":"integer","description":"Max characters of output (default 10000).","default":10000},
        },"required":["file_path"]},
    )

    async def validate(self, file_path: str, **_) -> None:
        p = Path(file_path).expanduser()
        if not p.exists(): raise SkillValidationError(f"File does not exist: '{file_path}'")

    async def execute(self, file_path: str, action: str="read", jmespath_query: str="",
                      output_path: str="", max_chars: int=10000, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_chars = min(int(max_chars), 50_000)
        p = Path(file_path).expanduser().resolve()

        def _load():
            text = p.read_text(errors="replace")
            if p.suffix == ".jsonl":
                return [json.loads(l) for l in text.splitlines() if l.strip()]
            return json.loads(text)

        try:
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(None, _load)

            if action == "read":
                out = json.dumps(data, indent=2, default=str)
                truncated = len(out) > max_chars
                return SkillResult.ok(self.manifest.name, call_id,
                    {"type":type(data).__name__,"preview":out[:max_chars],"truncated":truncated})

            elif action == "keys":
                if isinstance(data, dict):
                    keys = list(data.keys())
                elif isinstance(data, list) and data and isinstance(data[0], dict):
                    keys = list(data[0].keys())
                else:
                    keys = []
                return SkillResult.ok(self.manifest.name, call_id, {"keys":keys,"type":type(data).__name__})

            elif action == "query":
                if not jmespath_query: return SkillResult.fail(self.manifest.name, call_id, "jmespath_query required for action='query'.")
                try: import jmespath; result = jmespath.search(jmespath_query, data)
                except ImportError:
                    # Fallback: simple key access
                    result = data
                    for key in jmespath_query.split("."):
                        if isinstance(result, dict): result = result.get(key)
                        else: break
                out = json.dumps(result, indent=2, default=str)[:max_chars]
                return SkillResult.ok(self.manifest.name, call_id, {"query":jmespath_query,"result":out})

            elif action == "write":
                if not output_path: return SkillResult.fail(self.manifest.name, call_id, "output_path required for action='write'.")
                out_p = Path(output_path).expanduser()
                def _write(): out_p.write_text(json.dumps(data, indent=2, default=str))
                await loop.run_in_executor(None, _write)
                return SkillResult.ok(self.manifest.name, call_id, {"written_to":str(out_p)})

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
