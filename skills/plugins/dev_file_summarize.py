"""
skills/plugins/dev_file_summarize.py — Developer: File Summarize

Reads a source file and returns its structure: imports, classes, functions,
constants, and a line count. No LLM call — pure AST/regex analysis.

Risk: LOW — fs:read
"""
from __future__ import annotations
import asyncio, ast, re, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevFileSummarizeSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_file_summarize",
        version="1.0.0",
        description="Summarize a source file's structure: imports, classes, functions, constants, line count. Supports Python (AST) and other languages (regex). Returns a quick structural overview.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:read"}),
        timeout_seconds=15,
        parameters={"type":"object","properties":{
            "file_path":{"type":"string","description":"Path to the source file."},
            "include_docstrings":{"type":"boolean","description":"Include docstrings in function/class summaries.","default":False},
        },"required":["file_path"]},
    )

    async def validate(self, file_path: str, **_) -> None:
        p = Path(file_path).expanduser()
        if not p.exists(): raise SkillValidationError(f"File does not exist: '{file_path}'")
        if not p.is_file(): raise SkillValidationError(f"'{file_path}' is not a file.")

    async def execute(self, file_path: str, include_docstrings: bool=False, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        p = Path(file_path).expanduser().resolve()

        def _analyze():
            src = p.read_text(errors="replace")
            lines = src.splitlines()
            result = {"file":str(p),"line_count":len(lines),"language":"unknown","imports":[],"classes":[],"functions":[],"constants":[]}

            if p.suffix == ".py":
                result["language"] = "python"
                try:
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.Import, ast.ImportFrom)):
                            if isinstance(node, ast.Import):
                                for alias in node.names: result["imports"].append(alias.name)
                            else:
                                mod = node.module or ""
                                for alias in node.names: result["imports"].append(f"{mod}.{alias.name}" if mod else alias.name)
                        elif isinstance(node, ast.ClassDef):
                            doc = ast.get_docstring(node) if include_docstrings else ""
                            bases = [ast.unparse(b) for b in node.bases] if hasattr(ast,"unparse") else []
                            methods = [n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef) and n.col_offset > node.col_offset]
                            result["classes"].append({"name":node.name,"line":node.lineno,"bases":bases,"methods":methods[:20],"docstring":doc[:200] if doc else ""})
                        elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                            doc = ast.get_docstring(node) if include_docstrings else ""
                            args = [a.arg for a in node.args.args]
                            result["functions"].append({"name":node.name,"line":node.lineno,"args":args,"docstring":doc[:200] if doc else ""})
                        elif isinstance(node, ast.Assign) and node.col_offset == 0:
                            for t in node.targets:
                                if isinstance(t, ast.Name) and t.id.isupper():
                                    result["constants"].append(t.id)
                except SyntaxError as e:
                    result["parse_error"] = str(e)
            else:
                # Generic regex analysis
                lang_map = {".js":".js",".ts":".ts",".go":"go",".rs":"rust",".rb":"ruby",".java":"java",".cpp":"cpp",".c":"c"}
                result["language"] = lang_map.get(p.suffix, p.suffix.lstrip(".") or "text")
                result["imports"] = list({m for m in re.findall(r'(?:import|require|#include)\s+["\']?([^\s"\';<>]+)',src)})[:30]
                result["functions"] = [{"name":m,"line":0} for m in re.findall(r'(?:func|def|function|fn)\s+(\w+)\s*\(',src)][:30]
                result["classes"] = [{"name":m} for m in re.findall(r'(?:class|struct|interface)\s+(\w+)',src)][:20]
            return result

        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, _analyze)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, summary, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
