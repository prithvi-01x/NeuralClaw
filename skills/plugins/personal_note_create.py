"""
skills/plugins/personal_note_create.py — Personal: Note Create

Creates a markdown note file in ~/neuralclaw/notes/.
Supports create, read, list, search, delete.

Risk: LOW — fs:write
"""
from __future__ import annotations
import asyncio, re, time
from datetime import datetime, timezone
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_NOTES_DIR = Path("~/neuralclaw/notes").expanduser()

class PersonalNoteCreateSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_note_create",
        version="1.0.0",
        description="Create, read, list, search, and delete markdown notes in ~/neuralclaw/notes/.",
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"fs:write"}),
        timeout_seconds=10,
        parameters={"type":"object","properties":{
            "action":{"type":"string","enum":["create","read","list","search","delete"],"description":"Action to perform."},
            "title":{"type":"string","description":"Note title (required for create).","default":""},
            "content":{"type":"string","description":"Note content in markdown (required for create).","default":""},
            "filename":{"type":"string","description":"Exact filename for read/delete (e.g. 'my_note.md').","default":""},
            "query":{"type":"string","description":"Search query for search action.","default":""},
            "tags":{"type":"array","items":{"type":"string"},"description":"Tags for the note (for create).","default":[]},
        },"required":["action"]},
    )

    async def validate(self, action: str, title: str="", content: str="", filename: str="", **_) -> None:
        if action == "create" and not title.strip():
            raise SkillValidationError("title is required for action='create'.")
        if action in ("read","delete") and not filename.strip():
            raise SkillValidationError(f"filename is required for action='{action}'.")

    async def execute(self, action: str, title: str="", content: str="", filename: str="",
                      query: str="", tags: list|None=None, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        tags = tags or []

        def _safe(name: str) -> str:
            return re.sub(r"[^\w\s-]","",name).strip().replace(" ","_")[:60]

        try:
            loop = asyncio.get_event_loop()

            if action == "create":
                ts = datetime.now(tz=timezone.utc)
                fname = f"{ts.strftime('%Y%m%d_%H%M%S')}_{_safe(title)}.md"
                tag_line = f"tags: {', '.join(tags)}\n" if tags else ""
                body = f"# {title}\n\n*Created: {ts.isoformat()}*\n{tag_line}\n{content}\n"
                out = _NOTES_DIR / fname
                def _write():
                    _NOTES_DIR.mkdir(parents=True, exist_ok=True)
                    out.write_text(body)
                await loop.run_in_executor(None, _write)
                return SkillResult.ok(self.manifest.name, call_id, {"created": str(out), "filename": fname})

            elif action == "read":
                p = _NOTES_DIR / filename
                if not p.exists(): return SkillResult.fail(self.manifest.name, call_id, f"Note '{filename}' not found.")
                text = await loop.run_in_executor(None, p.read_text)
                return SkillResult.ok(self.manifest.name, call_id, {"filename":filename,"content":text})

            elif action == "list":
                def _list():
                    if not _NOTES_DIR.exists(): return []
                    return [{"filename":f.name,"size":f.stat().st_size,"modified":datetime.fromtimestamp(f.stat().st_mtime,tz=timezone.utc).isoformat()}
                            for f in sorted(_NOTES_DIR.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)]
                notes = await loop.run_in_executor(None, _list)
                return SkillResult.ok(self.manifest.name, call_id, {"count":len(notes),"notes":notes})

            elif action == "search":
                q = query.lower()
                def _search():
                    if not _NOTES_DIR.exists(): return []
                    results = []
                    for f in _NOTES_DIR.glob("*.md"):
                        try:
                            text = f.read_text()
                            if q in text.lower():
                                results.append({"filename":f.name,"snippet":text[:200]})
                        except Exception: pass
                    return results
                matches = await loop.run_in_executor(None, _search)
                return SkillResult.ok(self.manifest.name, call_id, {"count":len(matches),"matches":matches})

            elif action == "delete":
                p = _NOTES_DIR / filename
                if not p.exists(): return SkillResult.fail(self.manifest.name, call_id, f"Note '{filename}' not found.")
                await loop.run_in_executor(None, p.unlink)
                return SkillResult.ok(self.manifest.name, call_id, {"deleted": filename})

            return SkillResult.fail(self.manifest.name, call_id, f"Unknown action: {action}")
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
