"""
skills/plugins/cyber_cve_lookup.py — Cyber: CVE Lookup

Queries the NVD (NIST National Vulnerability Database) API for CVE details.
Free, no API key needed for basic queries.

Risk: LOW — net:fetch
"""
from __future__ import annotations
import time
from typing import ClassVar
from neuralclaw.skills.base import SkillBase
from neuralclaw.skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

class CyberCveLookupSkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="cyber_cve_lookup",
        version="1.0.0",
        description="Look up CVE details from the NIST National Vulnerability Database. Search by CVE ID (e.g. 'CVE-2024-1234') or by keyword/product name.",
        category="cybersecurity",
        risk_level=RiskLevel.LOW,
        capabilities=frozenset({"net:fetch"}),
        timeout_seconds=20,
        parameters={"type":"object","properties":{
            "cve_id":{"type":"string","description":"Specific CVE ID (e.g. 'CVE-2024-1234'). Takes priority over keyword.","default":""},
            "keyword":{"type":"string","description":"Product or vulnerability keyword to search for.","default":""},
            "limit":{"type":"integer","description":"Max CVEs to return for keyword search (default 5).","default":5},
        },"required":[]},
    )

    async def validate(self, cve_id: str="", keyword: str="", **_) -> None:
        if not cve_id and not keyword: raise SkillValidationError("Either cve_id or keyword must be provided.")

    async def execute(self, cve_id: str="", keyword: str="", limit: int=5, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        limit = min(int(limit), 20)

        try:
            import httpx
            base = "https://services.nvd.nist.gov/rest/json/cves/2.0"
            headers = {"User-Agent":"NeuralClaw/1.0"}

            async with httpx.AsyncClient(timeout=15.0) as client:
                if cve_id:
                    params = {"cveId": cve_id.upper()}
                else:
                    params = {"keywordSearch": keyword, "resultsPerPage": limit}

                r = await client.get(base, params=params, headers=headers)
                if r.status_code == 403: return SkillResult.fail(self.manifest.name, call_id, "NVD API rate limited. Wait a minute and retry.")
                r.raise_for_status()
                data = r.json()

            vulns = []
            for item in data.get("vulnerabilities", []):
                cve = item.get("cve", {})
                cve_id_val = cve.get("id","")
                descs = cve.get("descriptions",[])
                desc = next((d["value"] for d in descs if d.get("lang")=="en"), "")
                metrics = cve.get("metrics",{})
                score = None
                severity = None
                for key in ["cvssMetricV31","cvssMetricV30","cvssMetricV2"]:
                    if key in metrics and metrics[key]:
                        cvss = metrics[key][0].get("cvssData",{})
                        score = cvss.get("baseScore")
                        severity = cvss.get("baseSeverity")
                        break
                published = cve.get("published","")
                refs = [r.get("url","") for r in cve.get("references",[])[:3]]
                vulns.append({"cve_id":cve_id_val,"description":desc[:500],"cvss_score":score,
                               "severity":severity,"published":published,"references":refs})

            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id,
                {"query":cve_id or keyword,"total_results":data.get("totalResults",0),"returned":len(vulns),"cves":vulns},
                duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
