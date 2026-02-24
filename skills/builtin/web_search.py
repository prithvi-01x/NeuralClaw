"""
skills/builtin/web_search.py — Web Search Skill

Migrated from tools/search.py to the SkillBase contract.
Uses DuckDuckGo HTML search — no API key required.
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Optional

from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult

_DDG_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}
_TIMEOUT = 15.0


class WebSearchSkill(SkillBase):
    manifest = SkillManifest(
        name="web_search",
        version="1.0.0",
        description=(
            "Search the web using DuckDuckGo and return the top results. "
            "Returns page titles, URLs, and short snippets. "
            "No API key required."
        ),
        category="web",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query"},
                "max_results": {"type": "integer", "description": "Max results (default 5, max 10)", "default": 5},
            },
            "required": ["query"],
        },
        timeout_seconds=20,
    )

    async def execute(self, query: str, max_results: int = 5, **_) -> SkillResult:
        call_id = _.get("_skill_call_id", "")
        max_results = min(max_results, 10)
        try:
            import httpx
            from bs4 import BeautifulSoup

            async with httpx.AsyncClient(headers=_HEADERS, timeout=_TIMEOUT, follow_redirects=True) as client:
                response = await client.post(_DDG_URL, data={"q": query, "b": "", "kl": "us-en"})
                response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            results = []
            for result in soup.select(".result"):
                if len(results) >= max_results:
                    break
                title_tag = result.select_one(".result__title a")
                if not title_tag:
                    continue
                title = title_tag.get_text(strip=True)
                raw_url = title_tag.get("href", "")
                url = _clean_url(raw_url)
                if not url or not title:
                    continue
                snippet_tag = result.select_one(".result__snippet")
                snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""
                results.append({"title": title, "url": url, "snippet": snippet})

            return SkillResult.ok(
                skill_name=self.manifest.name, skill_call_id=call_id,
                output={"query": query, "result_count": len(results), "results": results},
            )
        except Exception as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__)


def _clean_url(raw_url: str) -> Optional[str]:
    if not raw_url:
        return None
    if raw_url.startswith("/l/?"):
        parsed = urllib.parse.urlparse("https://duckduckgo.com" + raw_url)
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])
    if raw_url.startswith("http"):
        return raw_url
    return None