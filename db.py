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
    providers TEXT,
    batch_id TEXT,
    parent_run_id TEXT
);
CREATE TABLE IF NOT EXISTS cache (
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    created_at TEXT NOT NULL,
    value_json TEXT NOT NULL,
    PRIMARY KEY (kind, key)
);
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    name TEXT,
    run_ids TEXT NOT NULL DEFAULT '[]'
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
            cols = {r["name"] for r in c.execute("PRAGMA table_info(runs)").fetchall()}
            for col in ("batch_id", "parent_run_id"):  # migrate databases created by older versions
                if col not in cols:
                    c.execute(f"ALTER TABLE runs ADD COLUMN {col} TEXT")

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

    # ---- cache ------------------------------------------------------------
    def cache_get(self, kind: str, key: str, max_age_hours: float):
        with self._conn() as c:
            row = c.execute("SELECT created_at, value_json FROM cache WHERE kind = ? AND key = ?", (kind, key)).fetchone()
        if not row:
            return None
        age = datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])
        if age.total_seconds() > max_age_hours * 3600:
            return None
        return json.loads(row["value_json"])

    def cache_put(self, kind: str, key: str, value) -> None:
        with self._conn() as c:
            c.execute("INSERT OR REPLACE INTO cache (kind, key, created_at, value_json) VALUES (?,?,?,?)",
                      (kind, key, now(), json.dumps(value, default=str)))

    # ---- batches ----------------------------------------------------------
    def create_batch(self, name: str) -> str:
        bid = uuid.uuid4().hex[:8]
        with self._conn() as c:
            c.execute("INSERT INTO batches (id, created_at, name) VALUES (?,?,?)", (bid, now(), name))
        return bid

    def add_to_batch(self, batch_id: str, run_id: str) -> None:
        with self._conn() as c:
            row = c.execute("SELECT run_ids FROM batches WHERE id = ?", (batch_id,)).fetchone()
            ids = json.loads(row["run_ids"]) if row else []
            ids.append(run_id)
            c.execute("UPDATE batches SET run_ids = ? WHERE id = ?", (json.dumps(ids), batch_id))

    def list_batches(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()
        return [dict(r) | {"run_ids": json.loads(r["run_ids"])} for r in rows]

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
