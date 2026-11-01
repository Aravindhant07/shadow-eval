"""Live shadow mode in five lines.

Wrap your existing production function. Its behaviour never changes; the agent
decides each case in the background, and human decisions are joined in later.

    python examples/live_capture.py                      # simulated agent (free, offline)
    python examples/live_capture.py --agent ollama       # local open-source model via Ollama
"""

import argparse
import json
from pathlib import Path

from shadow_eval.agents import make_agent
from shadow_eval.capture.live import ShadowObserver
from shadow_eval.config import load_domain
from shadow_eval.score import scorecard
from shadow_eval.store import Store

ROOT = Path(__file__).resolve().parents[1]

parser = argparse.ArgumentParser()
parser.add_argument("--agent", default="simulated", help="simulated | ollama | hf | claude")
parser.add_argument("--model", default=None)
args = parser.parse_args()

domain = load_domain(ROOT / "domains" / "insurance_claims.yaml")
store = Store("live_demo.db")
observer = ShadowObserver(store, domain, make_agent(args.agent, domain, args.model))


@observer.observe
def triage_claim(claim: dict) -> str:
    """Stand-in for your production code: it just routes the claim to the team."""
    return f"{claim['claim_id']} routed to the claims team"


cases = json.loads((ROOT / "data" / "demo" / "cases.json").read_text())[:40]
for case in cases:
    print(triage_claim(case))  # production behaves exactly as before

# Later, when adjusters have decided (here: from the demo's human decisions file)
import csv

with open(ROOT / "data" / "demo" / "human.csv", newline="") as fh:
    humans = {r["claim_id"]: r for r in csv.DictReader(fh)}
for case in cases:
    h = humans[case["claim_id"]]
    observer.record_human(case["claim_id"], {"action": h["action"], "payout": float(h["payout"])}, decided_by=h["decided_by"])

observer.drain()
sc = scorecard(store, domain)
m = sc.overall.metrics
print(f"\n{m.n} cases compared · agreement {m.agreement_rate:.0%} · {m.disagreements} disagreements to review")
print("Review them with: shadow serve --db live_demo.db")
