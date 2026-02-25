"""
skills/plugins/data_sql_query.py — Data: SQL Query

Execute SQL queries against a SQLite database file.
Defaults to SELECT-only. Write mode (INSERT/UPDATE/DELETE) requires allow_writes=True.

Risk: MED — data:read
"""
from __future__ import annotations
import asyncio, re, time
from pathlib import Path
from typing import ClassVar
from skills.base import SkillBase
from skills.types import RiskLevel, SkillManifest, SkillResult, SkillValidationError

_WRITE_PATTERN = re.compile(r"^\s*(INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|TRUNCATE|REPLACE)\b", re.IGNORECASE)

class DataSqlQuerySkill(SkillBase):
    manifest: ClassVar[SkillManifest] = SkillManifest(
        name="data_sql_query",
        version="1.0.0",
        description="Run SQL queries against a SQLite database file. SELECT-only by default. Set allow_writes=true to permit INSERT/UPDATE/DELETE. Returns rows as list of dicts.",
        category="data",
        risk_level=RiskLevel.MEDIUM,
        capabilities=frozenset({"data:read"}),
        timeout_seconds=30,
        parameters={"type":"object","properties":{
            "db_path":{"type":"string","description":"Path to SQLite .db file."},
            "query":{"type":"string","description":"SQL query to execute."},
            "allow_writes":{"type":"boolean","description":"Allow INSERT/UPDATE/DELETE (default false).","default":False},
            "max_rows":{"type":"integer","description":"Max rows to return (default 100).","default":100},
            "params":{"type":"array","description":"Query parameters for parameterised queries.","default":[]},
        },"required":["db_path","query"]},
    )

    async def validate(self, db_path: str, query: str, allow_writes: bool=False, **_) -> None:
        p = Path(db_path).expanduser()
        if not p.exists(): raise SkillValidationError(f"Database file does not exist: '{db_path}'")
        if not allow_writes and _WRITE_PATTERN.match(query):
            raise SkillValidationError("Write operations (INSERT/UPDATE/DELETE/DROP) are blocked. Set allow_writes=true to enable.")

    async def execute(self, db_path: str, query: str, allow_writes: bool=False, max_rows: int=100,
                      params: list|None=None, **kwargs) -> SkillResult:
        call_id = kwargs.get("_skill_call_id","")
        t_start = time.monotonic()
        max_rows = min(int(max_rows), 1000)
        params = params or []

        def _run_query():
            import sqlite3
            p = Path(db_path).expanduser().resolve()
            conn = sqlite3.connect(str(p))
            conn.row_factory = sqlite3.Row
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                if _WRITE_PATTERN.match(query):
                    conn.commit()
                    return {"rows_affected":cur.rowcount,"rows":[]}
                rows = [dict(r) for r in cur.fetchmany(max_rows)]
                columns = [d[0] for d in cur.description] if cur.description else []
                return {"column_count":len(columns),"columns":columns,"row_count":len(rows),"rows":rows}
            finally:
                conn.close()

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, _run_query)
            duration_ms = (time.monotonic()-t_start)*1000
            return SkillResult.ok(self.manifest.name, call_id, result, duration_ms=duration_ms)
        except BaseException as e:
            return SkillResult.fail(self.manifest.name, call_id, f"{type(e).__name__}: {e}", type(e).__name__,
                                    duration_ms=(time.monotonic()-t_start)*1000)
