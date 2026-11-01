"""Build the offline demo database end to end (no API key, no cost)."""

from __future__ import annotations

import csv
from pathlib import Path

from ..adjudicate import record_review, review_queue, suggest_all
from ..capture.batch import load_cases, load_human, run_shadow
from ..config import DomainConfig, load_domain
from ..store import Store
from .generate import generate, write_dataset
from .simulated import PolicyRulesAdjudicator, SimulatedAgent, simulated_sme

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOMAIN = ROOT / "domains" / "insurance_claims.yaml"
DEFAULT_DATA = ROOT / "data" / "demo"


def build_demo(
    db_path: str | Path,
    domain: DomainConfig | None = None,
    data_dir: str | Path = DEFAULT_DATA,
    n: int = 600,
    review_share: float = 0.85,
    workers: int = 8,
) -> Store:
    """Generate (if needed) the synthetic claims, run the simulated agent, get
    suggestions, and let the simulated expert review the most costly share of the
    queue — leaving the rest pending so the review UI has work in it."""
    domain = domain or load_domain(DEFAULT_DOMAIN)
    data_dir = Path(data_dir)
    if not (data_dir / "cases.json").exists():
        write_dataset(generate(n=n), data_dir)
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    store = Store(db_path)
    load_cases(store, domain, data_dir / "cases.json")
    load_human(store, domain, data_dir / "human.csv")
    run_shadow(store, SimulatedAgent(), workers=workers)
    suggest_all(store, domain, PolicyRulesAdjudicator(domain), workers=workers)
    with open(data_dir / "truth.csv", newline="") as fh:
        truth = {r["claim_id"]: r for r in csv.DictReader(fh)}
    queue = review_queue(store, domain)
    for item in queue[: int(len(queue) * review_share)]:
        cid = item["case_id"]
        a, h = store.get_agent(cid), store.get_human(cid)
        assert a and a.decision and h
        verdict, correct = simulated_sme(domain, truth[cid], a.decision, h.decision)
        record_review(store, domain, cid, verdict, reviewer="demo-expert", notes="simulated expert (demo)", correct_values=correct)
    return store
