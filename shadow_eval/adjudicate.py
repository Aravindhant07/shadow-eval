"""Adjudication: decide who was right when the agent and the human disagree.

An adjudicator *suggests* a verdict; a subject-matter expert confirms or
overrides it in the review UI. The LLM adjudicator works blind: it sees the two
decisions as "A" and "B" in a random order, so it cannot favour the agent.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Protocol

from .compare import Outcome, diff, error_cost
from .config import DomainConfig
from .llm import JSONBackend, LLMError
from .models import Adjudication, Decision, Verdict
from .store import Store


class Adjudicator(Protocol):
    name: str

    def suggest(self, case: dict[str, Any], agent: Decision, human: Decision) -> Adjudication: ...


_BLIND = {"a_wrong", "b_wrong", "both_acceptable", "both_wrong", "ambiguous"}


class LLMAdjudicator:
    def __init__(self, domain: DomainConfig, backend: JSONBackend):
        self.domain = domain
        self.backend = backend
        self.name = f"llm-adjudicator:{backend.name}"

    def _schema(self) -> dict[str, Any]:
        props: dict[str, Any] = {
            "verdict": {
                "type": "string",
                "enum": sorted(_BLIND),
                "description": "a_wrong: only A is wrong. b_wrong: only B is wrong. both_acceptable: both comply with policy. "
                "both_wrong: neither complies. ambiguous: the policy does not settle it.",
            },
            "reasoning": {"type": "string", "description": "Apply the policy step by step and explain the verdict."},
        }
        for f in self.domain.decision:
            t = {"enum": "string", "number": "number", "boolean": "boolean", "text": "string"}[f.type]
            spec: dict[str, Any] = {"type": t, "description": f"Correct {f.name} under the policy."}
            if f.values:
                spec["enum"] = list(f.values)
            props[f"correct_{f.name}"] = spec
        return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}

    def _system(self) -> str:
        return (
            "You are an impartial senior reviewer. Two reviewers, A and B, decided the same case. "
            "Determine, strictly from the policy, which decision complies. Small numeric differences "
            "within tolerance count as equal. Also state the correct decision.\n\n"
            f"<policy>\n{self.domain.policy.strip()}\n</policy>"
        )

    @staticmethod
    def _fmt(d: Decision) -> str:
        return json.dumps({"decision": d.values, "rationale": d.rationale}, default=str)

    def suggest(self, case: dict[str, Any], agent: Decision, human: Decision) -> Adjudication:
        case_id = str(case.get(self.domain.case_id_field))
        agent_is_a = int(hashlib.sha256(case_id.encode()).hexdigest(), 16) % 2 == 0
        a, b = (agent, human) if agent_is_a else (human, agent)
        user = (
            "Case:\n```json\n" + json.dumps(case, indent=2, sort_keys=True, default=str) + "\n```\n\n"
            f"Decision A: {self._fmt(a)}\nDecision B: {self._fmt(b)}"
        )
        try:
            r = self.backend.complete(self._system(), user, self._schema())
        except LLMError as e:
            return Adjudication(case_id=case_id, suggested_verdict=None, suggested_reason=f"Adjudicator error: {e}", suggested_by=self.name)
        blind = str(r.data.get("verdict", "ambiguous"))
        mapping = {
            "a_wrong": Verdict.AGENT_WRONG if agent_is_a else Verdict.HUMAN_WRONG,
            "b_wrong": Verdict.HUMAN_WRONG if agent_is_a else Verdict.AGENT_WRONG,
            "both_acceptable": Verdict.BOTH_ACCEPTABLE,
            "both_wrong": Verdict.BOTH_WRONG,
            "ambiguous": Verdict.AMBIGUOUS,
        }
        verdict = mapping.get(blind, Verdict.AMBIGUOUS)
        correct = {f.name: r.data.get(f"correct_{f.name}") for f in self.domain.decision}
        return Adjudication(
            case_id=case_id,
            suggested_verdict=verdict,
            suggested_reason=str(r.data.get("reasoning", "")),
            suggested_by=self.name,
            suggested_correct=correct if verdict is Verdict.BOTH_WRONG else None,
        )


def is_spot_check(case_id: str, rate: float) -> bool:
    """Deterministic sample of agreed cases sent to the expert anyway."""
    if rate <= 0:
        return False
    h = int(hashlib.sha256(f"spot:{case_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < rate


def review_queue(store: Store, domain: DomainConfig) -> list[dict[str, Any]]:
    """Everything the expert should look at.

    * disagreements — agent and human differ, most costly first
    * spot checks   — a random sample of agreements, to catch shared mistakes
    Unreviewed items come first.
    """
    agents, humans = store.agent_records(), store.human_records()
    adjs = store.adjudications()
    rate = domain.readiness.spot_check_rate
    out = []
    for case_id, h in humans.items():
        a = agents.get(case_id)
        if not a or not a.decision:
            continue
        d = diff(domain, a.decision, h.decision)
        if d.outcome is Outcome.MATCH:
            if not is_spot_check(case_id, rate):
                continue
            kind = "spot_check"
        else:
            kind = "disagreement"
        # Potential cost if the agent's decision were wrong (human assumed right until reviewed).
        risk = error_cost(domain, a.decision, h.decision)
        out.append({"case_id": case_id, "kind": kind, "diff": d, "risk": risk, "adjudication": adjs.get(case_id)})
    out.sort(
        key=lambda x: (
            x["adjudication"] is not None and x["adjudication"].final_verdict is not None,
            x["kind"] == "spot_check",
            -x["risk"].cost,
            x["case_id"],
        )
    )
    return out


def suggest_all(
    store: Store,
    domain: DomainConfig,
    adjudicator: Adjudicator,
    workers: int = 4,
    redo: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> int:
    todo = []
    for item in review_queue(store, domain):
        adj = item["adjudication"]
        if adj and adj.suggested_verdict and not redo:
            continue
        todo.append(item["case_id"])

    def work(case_id: str) -> None:
        case = store.get_case(case_id)
        a, h = store.get_agent(case_id), store.get_human(case_id)
        assert case and a and a.decision and h
        sug = adjudicator.suggest(case.data, a.decision, h.decision)
        existing = store.get_adjudication(case_id)
        if existing and existing.final_verdict:  # keep the expert's decision
            existing.suggested_verdict = sug.suggested_verdict
            existing.suggested_reason = sug.suggested_reason
            existing.suggested_by = sug.suggested_by
            existing.suggested_correct = sug.suggested_correct
            sug = existing
        store.put_adjudication(sug, actor=adjudicator.name)

    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for _ in pool.map(work, todo):
            done += 1
            if on_progress:
                on_progress(done, len(todo))
    return done


def record_review(
    store: Store,
    domain: DomainConfig,
    case_id: str,
    verdict: Verdict,
    reviewer: str,
    notes: str = "",
    correct_values: dict[str, Any] | None = None,
) -> Adjudication:
    from .models import utcnow

    if verdict is Verdict.BOTH_WRONG:
        if not correct_values:
            raise ValueError("a 'both wrong' verdict needs the correct decision")
        for f in domain.decision:
            if f.name not in correct_values:
                raise ValueError(f"correct decision is missing '{f.name}'")
    adj = store.get_adjudication(case_id) or Adjudication(case_id=case_id)
    adj.final_verdict = verdict
    adj.reviewer = reviewer
    adj.notes = notes
    adj.correct_values = correct_values if verdict is Verdict.BOTH_WRONG else None
    adj.reviewed_at = utcnow()
    store.put_adjudication(adj, actor=reviewer)
    return adj
