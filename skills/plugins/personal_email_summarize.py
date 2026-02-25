"""
skills/plugins/personal_email_summarize.py — Personal: Email Summarize

Reads and summarizes emails from a local mbox file or maildir.
No external API — reads standard Unix mail formats.

Risk: LOW — data:read
"""
from __future__ import annotations
import asyncio, mailbox, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class PersonalEmailSummarizeSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_email_summarize",
        version="1.0.0",
        description="Read and summarize emails from a local mbox file or Maildir. Returns subject, sender, date, and body snippet for recent messages.",
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"data:read"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "mailbox_path":{"type":"string","description":"Path to mbox file or Maildir directory."},
            "limit":{"type":"integer","description":"Max emails to return (default 10, max 50).","default":10},
            "unread_only":{"type":"boolean","description":"Only show unread messages (default false).","default":False},
            "snippet_chars":{"type":"integer","description":"Body snippet length in chars (default 200).","default":200},
        },"required":["mailbox_path"]},
    )

    async def validate(self, mailbox_path: str, **_) -> None:
        p = Path(mailbox_path).expanduser()
        if not p.exists():
            raise SkillValidationError(f"Mailbox path does not exist: '{mailbox_path}'")

    async def execute(self, mailbox_path: str, limit: int=10, unread_only: bool=False, snippet_chars: int=200, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        limit = min(int(limit), 50)
        snippet_chars = min(int(snippet_chars), 500)

        def _read_mailbox() -> list[dict]:
            p = Path(mailbox_path).expanduser()
            if p.is_dir():
                mb = mailbox.Maildir(str(p), create=False)
            else:
                mb = mailbox.mbox(str(p), create=False)

            messages = []
            for key, msg in mb.items():
                if unread_only:
                    flags = msg.get_flags() if hasattr(msg,'get_flags') else ""
                    if 'S' in flags:  # 'S' = seen/read in Maildir
                        continue
                subject = msg.get("Subject","(no subject)")
                sender  = msg.get("From","(unknown)")
                date    = msg.get("Date","")
                # Extract plain text body
                body = ""
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            try: body = part.get_payload(decode=True).decode("utf-8","replace")[:snippet_chars]
                            except Exception: pass
                            break
                else:
                    try: body = msg.get_payload(decode=True).decode("utf-8","replace")[:snippet_chars]
                    except Exception: body = str(msg.get_payload())[:snippet_chars]
                messages.append({"subject":subject,"from":sender,"date":date,"snippet":body.strip()[:snippet_chars],"key":str(key)})
                if len(messages) >= limit:
                    break
            mb.close()
            return messages

        try:
            loop = asyncio.get_event_loop()
            messages = await loop.run_in_executor(None, _read_mailbox)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {"count":len(messages),"messages":messages}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
