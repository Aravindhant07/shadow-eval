"""The demo insurer's claims policy as code (mirrors domains/claims_policy.md).

Used to create ground truth for the synthetic dataset and by the demo's
rules-based adjudicator. Real deployments use an LLM adjudicator instead.
"""

from __future__ import annotations

from typing import Any

COVERAGE = {
    "auto": {"collision", "theft", "windshield"},
    "home": {"water_damage", "fire", "theft"},
    "travel": {"trip_cancellation", "lost_baggage", "medical_abroad"},
}


def fraud_signals(c: dict[str, Any]) -> list[str]:
    signals = []
    if c["days_since_policy_start"] < 30 and c["claim_amount"] > 5000:
        signals.append("new policy (<30 days) with claim over $5,000")
    if c["prior_claims_12m"] >= 3:
        signals.append(f"{c['prior_claims_12m']} prior claims in 12 months")
    if c["claim_type"] == "theft" and not c["police_report"]:
        signals.append("theft without a police report")
    if c["reported_delay_days"] > 60:
        signals.append(f"reported {c['reported_delay_days']} days after the incident")
    return signals


def compute_payout(c: dict[str, Any]) -> float:
    return round(min(c["claim_amount"], c["coverage_limit"]) - c["deductible"], 2)


def apply_policy(c: dict[str, Any]) -> tuple[str, float, str]:
    """Return (action, payout, rule explanation)."""
    if not c["policy_active"]:
        return "deny", 0.0, "Rule 1: policy is not active"
    if c["claim_type"] not in COVERAGE[c["policy_type"]]:
        return "deny", 0.0, f"Rule 2: {c['claim_type']} is not covered by a {c['policy_type']} policy"
    if not c["documents_complete"]:
        return "request_info", 0.0, "Rule 3: required documents are missing"
    signals = fraud_signals(c)
    if signals:
        return "flag_fraud", 0.0, "Rule 4: fraud signal — " + "; ".join(signals)
    if c["claim_amount"] > 25000:
        return "escalate", 0.0, "Rule 5: claim over $25,000 needs a senior adjuster"
    payout = compute_payout(c)
    if payout <= 0:
        return "deny", 0.0, "Rule 6: claim is below the deductible"
    return "approve", payout, f"Rule 6: covered — min(claim, limit) − deductible = ${payout:,.2f}"
