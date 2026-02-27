"""
skills/plugins/personal_news_digest.py — Personal: News Digest

Fetches top headlines from RSS feeds. No API key required.
Default feeds: BBC, Reuters, HN. User can supply custom feed URLs.

Risk: LOW — net:fetch
"""
from __future__ import annotations
import asyncio, time, xml.etree.ElementTree as ET
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_DEFAULT_FEEDS = {
    "bbc":     "https://feeds.bbci.co.uk/news/rss.xml",
    "reuters": "https://feeds.reuters.com/reuters/topNews",
    "hn":      "https://hnrss.org/frontpage",
    "techcrunch": "https://techcrunch.com/feed/",
}

class PersonalNewsDigestSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="personal_news_digest",
        version="1.0.0",
        description="Fetch top headlines from RSS feeds. Default sources: BBC, Reuters, HN, TechCrunch. Accepts custom feed URLs.",
        category="personal",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "sources":{"type":"array","items":{"type":"string"},"description":"Feed names (bbc/reuters/hn/techcrunch) or full RSS URLs. Default: all.","default":[]},
            "max_per_feed":{"type":"integer","description":"Max headlines per feed (default 5).","default":5},
            "topic_filter":{"type":"string","description":"Only include headlines containing this keyword.","default":""},
        },"required":[]},
    )

    async def execute(self, sources: list[str]|None=None, max_per_feed: int=5, topic_filter: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_per_feed = min(int(max_per_feed), 20)
        feeds: list[str] = []
        if not sources:
            feeds = list(_DEFAULT_FEEDS.values())
        else:
            for s in sources:
                feeds.append(_DEFAULT_FEEDS.get(s.lower(), s))

        try:
            import httpx
            results = []
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                async def fetch_feed(url: str) -> list[dict]:
                    items = []
                    try:
                        r = await client.get(url, headers={"User-Agent":"Mozilla/5.0"})
                        r.raise_for_status()
                        root = ET.fromstring(r.text)
                        ns = {"atom":"http://www.w3.org/2005/Atom"}
                        entries = root.findall(".//item") or root.findall(".//atom:entry", ns)
                        for entry in entries[:max_per_feed]:
                            title = (entry.findtext("title") or entry.findtext("atom:title",namespaces=ns) or "").strip()
                            link  = (entry.findtext("link")  or entry.findtext("atom:link",namespaces=ns) or "").strip()
                            pub   = (entry.findtext("pubDate") or entry.findtext("atom:published",namespaces=ns) or "").strip()
                            if topic_filter and topic_filter.lower() not in title.lower():
                                continue
                            if title:
                                items.append({"title":title,"link":link,"published":pub,"source":url})
                    except Exception:
                        pass
                    return items
                all_results = await asyncio.gather(*[fetch_feed(f) for f in feeds])
            for batch in all_results:
                results.extend(batch)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {"headline_count":len(results),"headlines":results}, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)
