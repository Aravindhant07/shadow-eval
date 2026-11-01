"""Batch replay: run the agent in shadow mode over historical cases."""

from __future__ import annotations

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

from ..agents.base import Agent
from ..config import DomainConfig
from ..models import AgentRecord, Case, Decision, HumanRecord
from ..store import Store


def _coerce(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in ("true", "yes"):
        return True
    if s.lower() in ("false", "no"):
        return False
    if s == "":
        return None
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        rows = json.loads(path.read_text())
        if isinstance(rows, dict):
            rows = rows.get("records") or rows.get("cases") or list(rows.values())
        return list(rows)
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    with open(path, newline="") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_cases(store: Store, domain: DomainConfig, path: str | Path) -> int:
    n = 0
    for row in read_rows(path):
        case_id = str(row[domain.case_id_field])
        store.put_case(Case(case_id=case_id, data=row, source="batch"))
        n += 1
    return n


def human_from_row(domain: DomainConfig, row: dict[str, Any]) -> HumanRecord:
    values = {}
    for f in domain.decision:
        if f.name not in row:
            raise ValueError(f"human decision row is missing '{f.name}': {row}")
        v = row[f.name]
        if f.type == "enum":
            v = str(v).strip().lower()
        elif f.type == "number":
            v = float(v) if v not in (None, "") else 0.0
        values[f.name] = v
    return HumanRecord(
        case_id=str(row[domain.case_id_field]),
        decision=Decision(values=values, rationale=str(row.get("rationale") or "")),
        decided_by=str(row.get("decided_by") or "unknown"),
    )


def load_human(store: Store, domain: DomainConfig, path: str | Path) -> int:
    n = 0
    for row in read_rows(path):
        store.put_human(human_from_row(domain, row))
        n += 1
    return n


def run_agent_on_case(agent: Agent, case: Case) -> AgentRecord:
    start = time.perf_counter()
    try:
        r = agent.decide(case.data)
        return AgentRecord(
            case_id=case.case_id,
            agent_name=agent.name,
            decision=r.decision,
            model=r.model,
            latency_ms=int((time.perf_counter() - start) * 1000),
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            cost_usd=r.cost_usd,
        )
    except Exception as e:  # an agent failure is data, not a crash
        return AgentRecord(
            case_id=case.case_id,
            agent_name=getattr(agent, "name", "agent"),
            error=f"{type(e).__name__}: {e}",
            latency_ms=int((time.perf_counter() - start) * 1000),
        )


def run_shadow(
    store: Store,
    agent: Agent,
    workers: int = 4,
    limit: int | None = None,
    retry_errors: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, int]:
    """Run the agent on every recorded case that doesn't have an agent decision yet.

    Resumable: re-running skips cases already decided.
    """
    existing = store.agent_records()
    todo = [
        c for c in store.cases()
        if c.case_id not in existing or (retry_errors and existing[c.case_id].error)
    ]
    if limit is not None:
        todo = todo[:limit]
    stats = {"ran": 0, "errors": 0, "skipped": len(existing)}

    def work(case: Case) -> AgentRecord:
        rec = run_agent_on_case(agent, case)
        store.put_agent(rec)
        return rec

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for rec in pool.map(work, todo):
            stats["ran"] += 1
            stats["errors"] += 1 if rec.error else 0
            if on_progress:
                on_progress(stats["ran"], len(todo))
    return stats
