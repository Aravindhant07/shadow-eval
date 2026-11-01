"""Run a real open-source model (downloaded from Hugging Face, run locally on CPU)
in shadow mode on a stratified sample of the demo claims, then score it.

    python scripts/real_model_run.py --model Qwen/Qwen2.5-1.5B-Instruct --n 20
"""

import argparse
import csv
import json
from pathlib import Path

from shadow_eval.adjudicate import record_review, review_queue, suggest_all
from shadow_eval.agents import make_agent
from shadow_eval.capture.batch import human_from_row, run_shadow
from shadow_eval.config import load_domain
from shadow_eval.demo.simulated import PolicyRulesAdjudicator, simulated_sme
from shadow_eval.models import Case
from shadow_eval.report import write_report
from shadow_eval.score import scorecard
from shadow_eval.store import Store

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
p.add_argument("--n", type=int, default=20)
p.add_argument("--db", default="real_run.db")
p.add_argument("--out", default="docs/results")
a = p.parse_args()

domain = load_domain(ROOT / "domains" / "insurance_claims.yaml")
cases = json.loads((ROOT / "data/demo/cases.json").read_text())
humans = {r["claim_id"]: r for r in csv.DictReader(open(ROOT / "data/demo/human.csv"))}
truth = {r["claim_id"]: r for r in csv.DictReader(open(ROOT / "data/demo/truth.csv"))}

# Stratified sample: equal share per policy type, in dataset order.
by_type: dict[str, list] = {}
for c in cases:
    by_type.setdefault(c["policy_type"], []).append(c)
per = -(-a.n // len(by_type))
sample = [c for t in sorted(by_type) for c in by_type[t][:per]][: a.n]

Path(a.db).unlink(missing_ok=True)
store = Store(a.db)
for c in sample:
    store.put_case(Case(case_id=c["claim_id"], data=c))
    store.put_human(human_from_row(domain, {**humans[c["claim_id"]], "payout": float(humans[c["claim_id"]]["payout"])}))

agent = make_agent("local", domain, a.model)
print(f"Running {agent.name} on {len(sample)} claims…", flush=True)
stats = run_shadow(store, agent, workers=1, on_progress=lambda d, t: print(f"  {d}/{t}", flush=True))
print(stats, flush=True)

suggest_all(store, domain, PolicyRulesAdjudicator(domain))
for item in review_queue(store, domain):
    cid = item["case_id"]
    ag, h = store.get_agent(cid), store.get_human(cid)
    v, cv = simulated_sme(domain, truth[cid], ag.decision, h.decision)
    record_review(store, domain, cid, v, "ground-truth", correct_values=cv)

sc = scorecard(store, domain)
m = sc.overall.metrics
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
slug = a.model.split("/")[-1]
write_report(sc, out / f"report_{slug}.html", title=f"Readiness: {a.model} on {len(sample)} claims")
rows = []
for r in sorted(sc.rows, key=lambda r: r.case_id):
    rows.append({
        "claim_id": r.case_id, "policy_type": r.case["policy_type"], "claim_type": r.case["claim_type"],
        "correct": truth[r.case_id]["action"], "model": r.agent.values["action"], "model_payout": r.agent.values["payout"],
        "confidence": r.agent.confidence, "model_correct": r.agent_correct, "error": r.agent_cost.label,
        "rationale": r.agent.rationale,
    })
failed = [rec.case_id for rec in store.agent_records().values() if rec.error]
summary = {
    "model": a.model, "runtime": "local CPU via Hugging Face transformers", "cases": len(sample), "failed_to_answer": failed,
    "accuracy": m.accuracy, "acc_ci": [m.acc_low, m.acc_high], "critical_errors": m.critical,
    "cost_per_100": m.cost_per_100, "readiness": sc.readiness.value,
    "latency_p50_ms": sc.run.latency_p50_ms, "rows": rows,
}
(out / f"results_{slug}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1))
