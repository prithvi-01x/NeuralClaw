"""
skills/plugins/dev_pr_summary.py — Developer: PR Summary

Fetches a GitHub pull request and returns a structured summary:
title, description, diff stats, reviewers, labels, CI status.
Uses GitHub public API — no auth needed for public repos.

Risk: LOW — net:fetch
"""
from __future__ import annotations
import asyncio, re, time
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class DevPrSummarySkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="dev_pr_summary",
        version="1.0.0",
        description="Fetch and summarize a GitHub pull request. Returns title, description, diff stats, labels, reviewers, and CI status. Works with public repos without authentication.",
        category="developer",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "pr_url":{"type":"string","description":"GitHub PR URL (e.g. 'https://github.com/owner/repo/pull/123') or 'owner/repo#123'."},
            "github_token":{"type":"string","description":"GitHub personal access token (optional, increases rate limit and enables private repos).","default":""},
        },"required":["pr_url"]},
    )

    async def validate(self, pr_url: str, **_) -> None:
        if not ("github.com" in pr_url or re.match(r"[\w-]+/[\w-]+#\d+", pr_url)):
            raise SkillValidationError("pr_url must be a GitHub PR URL or 'owner/repo#123' format.")

    async def execute(self, pr_url: str, github_token: str="", **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()

        # Parse URL to owner/repo/number
        m = re.search(r"github\.com/([^/]+)/([^/]+)/pull/(\d+)", pr_url)
        if not m:
            m2 = re.match(r"([\w-]+)/([\w-]+)#(\d+)", pr_url)
            if not m2: return SkillResult.fail(self.manifest.name, call_id, f"Cannot parse PR URL: '{pr_url}'")
            owner, repo, number = m2.group(1), m2.group(2), m2.group(3)
        else:
            owner, repo, number = m.group(1), m.group(2), m.group(3)

        try:
            import httpx
            headers = {"Accept":"application/vnd.github.v3+json","User-Agent":"NeuralClaw/1.0"}
            if github_token: headers["Authorization"] = f"token {github_token}"

            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/pulls/{number}", headers=headers)
                if r.status_code == 404: return SkillResult.fail(self.manifest.name, call_id, f"PR not found: {owner}/{repo}#{number}")
                if r.status_code == 403: return SkillResult.fail(self.manifest.name, call_id, "Rate limited by GitHub API. Pass a github_token to increase limits.")
                r.raise_for_status()
                pr = r.json()

                # Fetch CI checks
                checks_r = await client.get(f"https://api.github.com/repos/{owner}/{repo}/commits/{pr['head']['sha']}/check-runs", headers=headers)
                checks = []
                if checks_r.status_code == 200:
                    for c in checks_r.json().get("check_runs",[])[:10]:
                        checks.append({"name":c.get("name"),"status":c.get("status"),"conclusion":c.get("conclusion")})

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, {
                "url": pr.get("html_url"), "number": pr.get("number"), "title": pr.get("title"),
                "state": pr.get("state"), "draft": pr.get("draft"),
                "author": pr.get("user",{}).get("login"),
                "base": pr.get("base",{}).get("ref"), "head": pr.get("head",{}).get("ref"),
                "body": (pr.get("body") or "")[:500],
                "commits": pr.get("commits"), "additions": pr.get("additions"),
                "deletions": pr.get("deletions"), "changed_files": pr.get("changed_files"),
                "labels": [l.get("name") for l in pr.get("labels",[])],
                "reviewers": [r.get("login") for r in pr.get("requested_reviewers",[])],
                "merged": pr.get("merged"), "mergeable": pr.get("mergeable"),
                "ci_checks": checks,
            }, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
