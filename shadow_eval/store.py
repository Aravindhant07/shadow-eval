"""SQLite persistence. Every write is also recorded in the audit log."""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Iterator

from . import audit
from .models import Adjudication, AgentRecord, Case, HumanRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agent_decisions (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS human_decisions (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS adjudications (case_id TEXT PRIMARY KEY, payload TEXT NOT NULL);
"""


class Store:
    def __init__(self, path: str | Path = "shadow.db"):
        self.path = str(path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA + audit.SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- generic helpers -------------------------------------------------
    def _put(self, table: str, case_id: str, payload: str, event: str, actor: str, audit_payload) -> None:
        with self._lock:
            self._conn.execute(
                f"INSERT INTO {table} (case_id, payload) VALUES (?, ?) "
                f"ON CONFLICT(case_id) DO UPDATE SET payload = excluded.payload",
                (case_id, payload),
            )
            audit.append(self._conn, event, case_id, actor, audit_payload)
            self._conn.commit()

    def _get(self, table: str, case_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(f"SELECT payload FROM {table} WHERE case_id = ?", (case_id,)).fetchone()
        return row[0] if row else None

    def _all(self, table: str) -> list[str]:
        with self._lock:
            return [r[0] for r in self._conn.execute(f"SELECT payload FROM {table} ORDER BY case_id")]

    # -- cases -----------------------------------------------------------
    def put_case(self, case: Case) -> None:
        self._put("cases", case.case_id, case.model_dump_json(), "case.recorded", case.source, case.model_dump())

    def get_case(self, case_id: str) -> Case | None:
        raw = self._get("cases", case_id)
        return Case.model_validate_json(raw) if raw else None

    def cases(self) -> list[Case]:
        return [Case.model_validate_json(r) for r in self._all("cases")]

    # -- agent decisions -------------------------------------------------
    def put_agent(self, rec: AgentRecord) -> None:
        self._put("agent_decisions", rec.case_id, rec.model_dump_json(), "agent.decided", rec.agent_name, rec.model_dump())

    def get_agent(self, case_id: str) -> AgentRecord | None:
        raw = self._get("agent_decisions", case_id)
        return AgentRecord.model_validate_json(raw) if raw else None

    def agent_records(self) -> dict[str, AgentRecord]:
        recs = (AgentRecord.model_validate_json(r) for r in self._all("agent_decisions"))
        return {r.case_id: r for r in recs}

    # -- human decisions -------------------------------------------------
    def put_human(self, rec: HumanRecord) -> None:
        self._put("human_decisions", rec.case_id, rec.model_dump_json(), "human.decided", rec.decided_by, rec.model_dump())

    def get_human(self, case_id: str) -> HumanRecord | None:
        raw = self._get("human_decisions", case_id)
        return HumanRecord.model_validate_json(raw) if raw else None

    def human_records(self) -> dict[str, HumanRecord]:
        recs = (HumanRecord.model_validate_json(r) for r in self._all("human_decisions"))
        return {r.case_id: r for r in recs}

    # -- adjudications ---------------------------------------------------
    def put_adjudication(self, adj: Adjudication, actor: str) -> None:
        event = "adjudication.reviewed" if adj.final_verdict else "adjudication.suggested"
        self._put("adjudications", adj.case_id, adj.model_dump_json(), event, actor, adj.model_dump(mode="json"))

    def get_adjudication(self, case_id: str) -> Adjudication | None:
        raw = self._get("adjudications", case_id)
        return Adjudication.model_validate_json(raw) if raw else None

    def adjudications(self) -> dict[str, Adjudication]:
        recs = (Adjudication.model_validate_json(r) for r in self._all("adjudications"))
        return {r.case_id: r for r in recs}

    # -- audit -----------------------------------------------------------
    def verify_audit(self) -> audit.VerifyResult:
        with self._lock:
            return audit.verify(self._conn)

    def audit_events(self, limit: int = 50) -> Iterator[tuple]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, ts, event, case_id, actor, hash FROM audit_log ORDER BY seq DESC LIMIT ?", (limit,)
            ).fetchall()
        return iter(rows)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                t: self._conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("cases", "agent_decisions", "human_decisions", "adjudications", "audit_log")
            }
