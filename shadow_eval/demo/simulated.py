"""Demo-only stand-ins so the full pipeline runs offline, for free.

* SimulatedAgent — behaves like a realistic LLM agent on the demo claims domain:
  strong on auto claims, over-cautious on some home claims, weak on travel
  medical claims, with confidence that is informative but imperfect.
* PolicyRulesAdjudicator — suggests verdicts by applying the demo policy as code.
* simulated_sme — plays the subject-matter expert using the dataset's ground truth.

None of these are used when you plug in a real model (Claude, Ollama, Hugging Face).
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

from ..agents.base import AgentResult
from ..compare import Outcome, diff
from ..config import DomainConfig
from ..models import Adjudication, Decision, Verdict
from .policy import apply_policy, compute_payout


def _rng_for(case_id: str, salt: str) -> random.Random:
    seed = int(hashlib.sha256(f"{salt}:{case_id}".encode()).hexdigest()[:12], 16)
    return random.Random(seed)


class SimulatedAgent:
    name = "simulated-agent (demo)"

    def __init__(self, seed: str = "v1"):
        self.seed = seed

    def decide(self, case: dict[str, Any]) -> AgentResult:
        rng = _rng_for(case["claim_id"], self.seed)
        action, payout, rule = apply_policy(case)
        err = None

        # Weak spot 1: travel medical claims — denies valid ones, often confidently.
        if case["claim_type"] == "medical_abroad" and action == "approve" and rng.random() < 0.30:
            err = ("deny", 0.0, "Misread the medical-abroad exclusion and denied the claim.", rng.uniform(0.84, 0.97))
        # Weak spot 2: travel claims — misapplies the coverage limit.
        elif case["policy_type"] == "travel" and action == "approve" and rng.random() < 0.12:
            wrong = round(case["claim_amount"], 2)
            if abs(wrong - payout) > max(50, 0.05 * payout):
                err = ("approve", wrong, "Paid the full claimed amount without applying limit/deductible.", rng.uniform(0.55, 0.85))
        # Weak spot 3: home claims — over-cautious fraud flags on late-ish reports.
        elif case["policy_type"] == "home" and action == "approve" and case["reported_delay_days"] > 25 and rng.random() < 0.45:
            err = ("flag_fraud", 0.0, "Treated the reporting delay as a fraud signal.", rng.uniform(0.50, 0.72))
        # Weak spot 4: theft without police report — occasionally approves anyway.
        elif action == "flag_fraud" and case["claim_type"] == "theft" and not case["police_report"] and rng.random() < 0.15:
            p = compute_payout(case)
            if p > 0:
                err = ("approve", p, "Did not notice the missing police report.", rng.uniform(0.55, 0.80))
        # Background noise everywhere.
        elif rng.random() < 0.015:
            alt = rng.choice([a for a in ["approve", "deny", "request_info", "escalate", "flag_fraud"] if a != action])
            err = (alt, compute_payout(case) if alt == "approve" else 0.0, "Misapplied the rule order.", rng.uniform(0.45, 0.80))

        if err:
            a, p, why, conf = err
            decision = Decision(
                values={"action": a, "payout": max(0.0, p)},
                rationale=why,
                confidence=round(conf, 2),
                citations=[rule.split(":")[0]],
            )
        else:
            conf = rng.uniform(0.86, 0.99) if rng.random() > 0.05 else rng.uniform(0.6, 0.85)
            decision = Decision(
                values={"action": action, "payout": payout},
                rationale=rule,
                confidence=round(conf, 2),
                citations=[rule.split(":")[0]],
            )
        tokens_in = 900 + rng.randint(0, 200)
        tokens_out = 120 + rng.randint(0, 80)
        return AgentResult(decision, "simulated-demo", tokens_in, tokens_out, 0.0)


def _policy_decision(case: dict[str, Any]) -> tuple[Decision, str]:
    action, payout, rule = apply_policy(case)
    return Decision(values={"action": action, "payout": payout}, rationale=rule), rule


def verdict_against(domain: DomainConfig, agent: Decision, human: Decision, correct: Decision) -> Verdict:
    agent_ok = diff(domain, agent, correct).outcome is Outcome.MATCH
    human_ok = diff(domain, human, correct).outcome is Outcome.MATCH
    if agent_ok and human_ok:
        return Verdict.BOTH_ACCEPTABLE
    if agent_ok:
        return Verdict.HUMAN_WRONG
    if human_ok:
        return Verdict.AGENT_WRONG
    return Verdict.BOTH_WRONG


class PolicyRulesAdjudicator:
    """Suggests a verdict by re-deciding the case with the policy coded as rules."""

    name = "policy-rules (demo)"

    def __init__(self, domain: DomainConfig):
        self.domain = domain

    def suggest(self, case: dict[str, Any], agent: Decision, human: Decision) -> Adjudication:
        correct, rule = _policy_decision(case)
        v = verdict_against(self.domain, agent, human, correct)
        reason = f"Policy check: {rule}. Correct decision: {correct.values['action']}"
        if correct.values["action"] == "approve":
            reason += f" ${correct.values['payout']:,.2f}"
        return Adjudication(
            case_id=case["claim_id"],
            suggested_verdict=v,
            suggested_reason=reason + ".",
            suggested_by=self.name,
            suggested_correct=correct.values if v is Verdict.BOTH_WRONG else None,
        )


def simulated_sme(domain: DomainConfig, truth_row: dict[str, Any], agent: Decision, human: Decision) -> tuple[Verdict, dict | None]:
    """The 'expert' answer from ground truth (demo only)."""
    correct = Decision(values={"action": truth_row["action"], "payout": float(truth_row["payout"])})
    v = verdict_against(domain, agent, human, correct)
    return v, (correct.values if v is Verdict.BOTH_WRONG else None)
