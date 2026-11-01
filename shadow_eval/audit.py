"""Tamper-evident audit log.

Every write to the store appends an event whose hash covers the previous event's
hash. Editing or deleting any past event breaks the chain, which `verify` detects.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from typing import Any

from .models import utcnow

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    event TEXT NOT NULL,
    case_id TEXT,
    actor TEXT NOT NULL,
    payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL,
    hash TEXT NOT NULL
);
"""


def _digest(prev_hash: str, ts: str, event: str, case_id: str | None, actor: str, payload: str) -> str:
    material = json.dumps([prev_hash, ts, event, case_id, actor, payload], separators=(",", ":"))
    return hashlib.sha256(material.encode()).hexdigest()


def append(conn: sqlite3.Connection, event: str, case_id: str | None, actor: str, payload: Any) -> str:
    """Append one event. Caller holds the store lock and commits."""
    row = conn.execute("SELECT hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
    prev_hash = row[0] if row else GENESIS
    ts = utcnow()
    body = json.dumps(payload, sort_keys=True, default=str)
    h = _digest(prev_hash, ts, event, case_id, actor, body)
    conn.execute(
        "INSERT INTO audit_log (ts, event, case_id, actor, payload, prev_hash, hash) VALUES (?,?,?,?,?,?,?)",
        (ts, event, case_id, actor, body, prev_hash, h),
    )
    return h


@dataclass
class VerifyResult:
    ok: bool
    events: int
    broken_at: int | None = None
    reason: str = ""


def verify(conn: sqlite3.Connection) -> VerifyResult:
    prev = GENESIS
    expected_seq = None
    n = 0
    for seq, ts, event, case_id, actor, payload, prev_hash, h in conn.execute(
        "SELECT seq, ts, event, case_id, actor, payload, prev_hash, hash FROM audit_log ORDER BY seq"
    ):
        n += 1
        if expected_seq is not None and seq != expected_seq:
            return VerifyResult(False, n, seq, f"gap in sequence: expected {expected_seq}, found {seq}")
        if prev_hash != prev:
            return VerifyResult(False, n, seq, "prev_hash does not match previous event (deleted or reordered event)")
        if _digest(prev_hash, ts, event, case_id, actor, payload) != h:
            return VerifyResult(False, n, seq, "event content does not match its hash (edited event)")
        prev = h
        expected_seq = seq + 1
    return VerifyResult(True, n)
