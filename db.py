"""SQLite persistence for workflow runs and per-stage outputs."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    prospect_name TEXT NOT NULL,
    company_name TEXT NOT NULL,
    prospect_role TEXT NOT NULL,
    company_website TEXT,
    status TEXT NOT NULL,            -- RUNNING | COMPLETED | FAILED
    outcome TEXT,                    -- DRAFT_READY | DRAFT_WITH_CAVEATS | ABSTAIN_* | NEEDS_INPUT
    outcome_reason TEXT,
    review_status TEXT DEFAULT 'PENDING_REVIEW',  -- PENDING_REVIEW | APPROVED | REJECTED | N/A
    draft_subject TEXT,
    draft_body TEXT,
    edited_subject TEXT,
    edited_body TEXT,
    reviewer_note TEXT,
    error TEXT,
    providers TEXT
);
CREATE TABLE IF NOT EXISTS stages (
    run_id TEXT NOT NULL,
    stage_no INTEGER NOT NULL,
    name TEXT NOT NULL,
    status TEXT NOT NULL,            -- RUNNING | DONE | FAILED | SKIPPED
    started_at TEXT,
    finished_at TEXT,
    summary TEXT,
    output_json TEXT,
    PRIMARY KEY (run_id, stage_no)
);
"""


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class RunStore:
    def __init__(self, path: str):
        self.path = path
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- runs -------------------------------------------------------------
    def create_run(self, prospect_name: str, company_name: str, prospect_role: str,
                   company_website: str | None, providers: str) -> str:
        run_id = uuid.uuid4().hex[:10]
        ts = now()
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (id, created_at, updated_at, prospect_name, company_name, prospect_role,"
                " company_website, status, providers) VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, ts, ts, prospect_name, company_name, prospect_role, company_website or None,
                 "RUNNING", providers),
            )
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        fields["updated_at"] = now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE runs SET {cols} WHERE id = ?", (*fields.values(), run_id))

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- stages -----------------------------------------------------------
    def stage_started(self, run_id: str, stage_no: int, name: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO stages (run_id, stage_no, name, status, started_at) VALUES (?,?,?,?,?)",
                (run_id, stage_no, name, "RUNNING", now()),
            )

    def stage_finished(self, run_id: str, stage_no: int, status: str, summary: str, output: Any) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE stages SET status = ?, finished_at = ?, summary = ?, output_json = ?"
                " WHERE run_id = ? AND stage_no = ?",
                (status, now(), summary, json.dumps(output, default=str), run_id, stage_no),
            )

    def get_stages(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM stages WHERE run_id = ? ORDER BY stage_no", (run_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["output"] = json.loads(d.pop("output_json")) if d.get("output_json") else None
            out.append(d)
        return out
