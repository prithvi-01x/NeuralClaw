"""
tools/search.py — Web Search Tool

Provides the agent with web search capability via DuckDuckGo.
No API key required — uses the public DuckDuckGo instant answer API
and HTML scraping via httpx + BeautifulSoup.

Registered tools:
  - web_search → search the web and return top results
"""

from __future__ import annotations

import json
import urllib.parse
from typing import Optional

import httpx
from bs4 import BeautifulSoup

from tools.tool_registry import registry
from tools.types import RiskLevel
from observability.logger import get_logger

log = get_logger(__name__)

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
_DEFAULT_MAX_RESULTS = 5


@registry.register(
    name="web_search",
    description=(
        "Search the web using DuckDuckGo and return the top results. "
        "Returns page titles, URLs, and short snippets. "
        "Use this to find current information, documentation, or research topics. "
        "No API key required."
    ),
    category="search",
    risk_level=RiskLevel.LOW,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (default: 5, max: 10)",
                "default": 5,
            },
        },
        "required": ["query"],
    },
)
async def web_search(query: str, max_results: int = _DEFAULT_MAX_RESULTS) -> str:
    """
    Search DuckDuckGo and return structured results.

    Returns a JSON string with a list of results, each containing:
    - title: page title
    - url: page URL
    - snippet: short description
    """
    max_results = min(max_results, 10)  # Hard cap

    log.debug("web_search.start", query=query, max_results=max_results)

    try:
        results = await _ddg_search(query, max_results)
    except httpx.TimeoutException:
        return json.dumps({"error": "Search timed out", "query": query})
    except httpx.HTTPError as e:
        return json.dumps({"error": f"HTTP error: {e}", "query": query})
    except Exception as e:
        log.error("web_search.error", query=query, error=str(e))
        return json.dumps({"error": str(e), "query": query})

    log.debug("web_search.complete", query=query, result_count=len(results))

    return json.dumps({
        "query": query,
        "result_count": len(results),
        "results": results,
    }, indent=2)


async def _ddg_search(query: str, max_results: int) -> list[dict]:
    """
    Perform a DuckDuckGo HTML search and parse results.
    """
    async with httpx.AsyncClient(
        headers=_HEADERS,
        timeout=_TIMEOUT,
        follow_redirects=True,
    ) as client:
        response = await client.post(
            _DDG_URL,
            data={"q": query, "b": "", "kl": "us-en"},
        )
        response.raise_for_status()

    return _parse_ddg_results(response.text, max_results)


def _parse_ddg_results(html: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo HTML search results page."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    for result in soup.select(".result"):
        if len(results) >= max_results:
            break

        # Title + URL
        title_tag = result.select_one(".result__title a")
        if not title_tag:
            continue

        title = title_tag.get_text(strip=True)
        raw_url = title_tag.get("href", "")
        url = _clean_ddg_url(raw_url)

        if not url or not title:
            continue

        # Snippet
        snippet_tag = result.select_one(".result__snippet")
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

        results.append({
            "title": title,
            "url": url,
            "snippet": snippet,
        })

    return results


def _clean_ddg_url(raw_url: str) -> Optional[str]:
    """
    DuckDuckGo wraps URLs in a redirect. Extract the actual URL.
    Handles both /l/?uddg=... format and direct URLs.
    """
    if not raw_url:
        return None

    # DDG redirect format: /l/?uddg=<encoded_url>&...
    if raw_url.startswith("/l/?"):
        parsed = urllib.parse.urlparse("https://duckduckgo.com" + raw_url)
        params = urllib.parse.parse_qs(parsed.query)
        if "uddg" in params:
            return urllib.parse.unquote(params["uddg"][0])

    # Already a direct URL
    if raw_url.startswith("http"):
        return raw_url

    return None