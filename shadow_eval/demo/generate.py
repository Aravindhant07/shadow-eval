"""Synthetic insurance-claims dataset with known ground truth.

Produces three tables:
* cases            — what both the human and the agent see
* human decisions  — what the claims team decided, with ~8% deliberately seeded
                     mistakes so we can check the evaluator finds them
* ground truth     — the policy-correct decision and which human errors were seeded
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

from .policy import COVERAGE, apply_policy, compute_payout

CLAIM_RANGES = {
    "windshield": (250, 1800),
    "collision": (800, 32000),
    "theft": (400, 38000),
    "water_damage": (900, 42000),
    "fire": (3000, 90000),
    "trip_cancellation": (300, 7000),
    "lost_baggage": (100, 3500),
    "medical_abroad": (400, 30000),
}
LIMITS = {"auto": [25000, 50000], "home": [150000, 300000], "travel": [5000, 10000, 20000]}
DEDUCTIBLES = {"auto": [250, 500, 1000], "home": [500, 1000, 2500], "travel": [0, 100, 250]}
POLICY_MIX = [("auto", 0.45), ("home", 0.35), ("travel", 0.20)]

NARRATIVES = {
    "windshield": "Stone chip on the highway cracked the windshield across the driver's side.",
    "collision": "Rear-ended at a junction; bumper, boot and rear sensors damaged.",
    "theft": "Customer reports the item was stolen overnight from {place}.",
    "water_damage": "Burst pipe under the kitchen sink flooded the ground floor.",
    "fire": "Kitchen fire spread to the dining room; fire service attended.",
    "trip_cancellation": "Trip cancelled after a family member was hospitalised.",
    "lost_baggage": "Checked bag lost by the airline on a connecting flight.",
    "medical_abroad": "Emergency treatment abroad after a fall while hiking.",
}


def _skewed_amount(rng: random.Random, lo: float, hi: float) -> float:
    # Most claims are small, a long tail is large.
    return round(lo + (hi - lo) * (rng.random() ** 2.2), 2)


def _make_case(rng: random.Random, idx: int) -> dict[str, Any]:
    policy_type = rng.choices([p for p, _ in POLICY_MIX], [w for _, w in POLICY_MIX])[0]
    covered = sorted(COVERAGE[policy_type])
    if rng.random() < 0.05:  # occasionally a claim type the policy doesn't cover
        claim_type = rng.choice(sorted(set(CLAIM_RANGES) - set(covered)))
    else:
        claim_type = rng.choice(covered)
    lo, hi = CLAIM_RANGES[claim_type]
    amount = _skewed_amount(rng, lo, hi)
    days = rng.randint(1, 29) if rng.random() < 0.07 else rng.randint(30, 2500)
    prior = rng.choices([0, 1, 2, 3, 4], [0.70, 0.17, 0.09, 0.03, 0.01])[0]
    delay = rng.randint(61, 120) if rng.random() < 0.04 else rng.randint(0, 45)
    police = rng.random() < (0.85 if claim_type == "theft" else 0.3)
    case = {
        "claim_id": f"CLM-{10000 + idx}",
        "policy_type": policy_type,
        "claim_type": claim_type,
        "claim_amount": amount,
        "coverage_limit": float(rng.choice(LIMITS[policy_type])),
        "deductible": float(rng.choice(DEDUCTIBLES[policy_type])),
        "policy_active": rng.random() > 0.04,
        "documents_complete": rng.random() > 0.09,
        "police_report": police,
        "days_since_policy_start": days,
        "prior_claims_12m": prior,
        "reported_delay_days": delay,
    }
    place = rng.choice(["a driveway", "a shared garage", "a hotel car park", "outside the house"])
    case["description"] = NARRATIVES[claim_type].format(place=place) + (
        " Police report attached." if police else ""
    )
    return case


def _seed_human_error(rng: random.Random, case: dict[str, Any], action: str, payout: float) -> tuple[str, float, str] | None:
    """Return a realistic human mistake for this case, or None if none applies."""
    options = []
    if action == "flag_fraud":
        p = compute_payout(case)
        if p > 0:
            options.append(("approve", p, "missed_fraud_signal"))
    if action == "approve":
        no_deductible = round(min(case["claim_amount"], case["coverage_limit"]), 2)
        if case["deductible"] > max(50, 0.06 * payout):
            options.append(("approve", no_deductible, "forgot_deductible"))
        if case["claim_type"] in ("water_damage", "medical_abroad"):
            options.append(("deny", 0.0, "wrongly_denied"))
        if case["claim_amount"] > 12000:
            options.append(("escalate", 0.0, "needless_escalation"))
    if action == "request_info":
        options.append(("approve", compute_payout(case), "paid_without_documents"))
    if not options:
        return None
    return rng.choice(options)


def generate(n: int = 600, seed: int = 7, human_error_rate: float = 0.08) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    cases, humans, truth = [], [], []
    adjusters = ["a.patel", "j.okafor", "m.garcia", "s.chen", "l.novak"]
    for i in range(n):
        case = _make_case(rng, i)
        action, payout, rule = apply_policy(case)
        h_action, h_payout, err = action, payout, ""
        if rng.random() < human_error_rate * 1.6:  # not every case has an applicable mistake
            seeded = _seed_human_error(rng, case, action, payout)
            if seeded:
                h_action, h_payout, err = seeded
        cases.append(case)
        humans.append(
            {"claim_id": case["claim_id"], "action": h_action, "payout": max(0.0, h_payout), "decided_by": rng.choice(adjusters)}
        )
        truth.append(
            {"claim_id": case["claim_id"], "action": action, "payout": payout, "rule": rule, "seeded_human_error": err}
        )
    return {"cases": cases, "human": humans, "truth": truth}


def write_dataset(data: dict[str, list[dict[str, Any]]], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.json").write_text(json.dumps(data["cases"], indent=1))
    for name in ("human", "truth"):
        rows = data[name]
        with open(out / f"{name}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
