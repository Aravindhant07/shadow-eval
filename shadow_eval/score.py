"""Readiness scoring: segment metrics, confidence intervals, guardrails, verdicts."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .compare import ErrorCost, Outcome, diff, error_cost
from .config import DomainConfig, ReadinessGates
from .models import Decision, Verdict
from .store import Store


class Readiness(str, Enum):
    GO = "GO"
    GO_WITH_GUARDRAILS = "GO WITH GUARDRAILS"
    NO_GO = "NO-GO"
    INSUFFICIENT_DATA = "INSUFFICIENT DATA"


@dataclass
class Row:
    case_id: str
    case: dict[str, Any]
    segments: dict[str, str]
    agent: Decision
    human: Decision
    truth: Decision | None  # None = ambiguous, excluded from accuracy
    status: str  # agreed | reviewed | unreviewed | ambiguous
    verdict: Verdict | None
    agent_cost: ErrorCost
    human_cost: ErrorCost | None

    @property
    def agreed(self) -> bool:
        return self.status == "agreed"

    @property
    def agent_correct(self) -> bool | None:
        return None if self.truth is None else self.agent_cost.cost == 0

    @property
    def human_correct(self) -> bool | None:
        return None if self.truth is None or self.human_cost is None else self.human_cost.cost == 0

    @property
    def confidence(self) -> float | None:
        return self.agent.confidence


ZERO = ErrorCost(0.0, False, "")


def build_rows(store: Store, domain: DomainConfig) -> tuple[list[Row], dict[str, int]]:
    """Join cases, agent and human decisions, and adjudications; resolve ground truth."""
    cases = {c.case_id: c for c in store.cases()}
    agents, humans, adjs = store.agent_records(), store.human_records(), store.adjudications()
    rows: list[Row] = []
    info = {"cases": len(cases), "awaiting_agent": 0, "awaiting_human": 0, "agent_errors": 0}
    for case_id, case in cases.items():
        a, h = agents.get(case_id), humans.get(case_id)
        if a is None:
            info["awaiting_agent"] += 1
            continue
        if a.decision is None:
            info["agent_errors"] += 1
            continue
        if h is None:
            info["awaiting_human"] += 1
            continue
        segs = domain.segments_for(case.data)
        adj = adjs.get(case_id)
        v = adj.final_verdict if adj else None
        if diff(domain, a.decision, h.decision).outcome is Outcome.MATCH:
            if v is Verdict.BOTH_WRONG and adj and adj.correct_values:  # spot check found a shared mistake
                truth = Decision(values=adj.correct_values)
                c = error_cost(domain, a.decision, truth)
                rows.append(Row(case_id, case.data, segs, a.decision, h.decision, truth, "agreed", v, c, error_cost(domain, h.decision, truth)))
            else:
                rows.append(Row(case_id, case.data, segs, a.decision, h.decision, h.decision, "agreed", v, ZERO, ZERO))
            continue
        if v is None:
            truth, status = h.decision, "unreviewed"
        elif v is Verdict.AGENT_WRONG:
            truth, status = h.decision, "reviewed"
        elif v is Verdict.HUMAN_WRONG:
            truth, status = a.decision, "reviewed"
        elif v is Verdict.BOTH_ACCEPTABLE:
            truth, status = a.decision, "reviewed"
        elif v is Verdict.BOTH_WRONG:
            truth, status = Decision(values=adj.correct_values or {}), "reviewed"
        else:
            truth, status = None, "ambiguous"
        if truth is None:
            rows.append(Row(case_id, case.data, segs, a.decision, h.decision, None, status, v, ZERO, None))
            continue
        if v is Verdict.BOTH_ACCEPTABLE:
            a_cost, h_cost = ZERO, ZERO
        else:
            a_cost, h_cost = error_cost(domain, a.decision, truth), error_cost(domain, h.decision, truth)
        rows.append(Row(case_id, case.data, segs, a.decision, h.decision, truth, status, v, a_cost, h_cost))
    return rows, info


def wilson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z = statistics.NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass
class Metrics:
    n: int = 0
    scored: int = 0
    agreement_rate: float = 0.0
    accuracy: float = 0.0
    acc_low: float = 0.0
    acc_high: float = 0.0
    cost_total: float = 0.0
    cost_per_100: float = 0.0
    critical: int = 0
    critical_rate: float = 0.0
    disagreements: int = 0
    reviewed: int = 0
    unreviewed: int = 0
    ambiguous: int = 0
    agent_right_when_disagreed: float | None = None
    human_errors: int = 0
    human_error_cost: float = 0.0
    human_accuracy: float = 0.0
    avg_confidence: float | None = None
    spot_checked: int = 0  # agreements reviewed by the expert
    shared_errors: int = 0  # ...where agent and human were both wrong
    shared_error_rate: float | None = None
    shared_error_high: float | None = None  # upper bound of the CI


def metrics(rows: list[Row], gates: ReadinessGates) -> Metrics:
    m = Metrics(n=len(rows))
    if not rows:
        return m
    scored = [r for r in rows if r.truth is not None]
    m.scored = len(scored)
    m.agreement_rate = sum(r.agreed for r in rows) / len(rows)
    correct = sum(1 for r in scored if r.agent_correct)
    m.accuracy = correct / m.scored if m.scored else 0.0
    m.acc_low, m.acc_high = wilson(correct, m.scored, gates.confidence_level)
    m.cost_total = sum(r.agent_cost.cost for r in scored)
    m.cost_per_100 = 100 * m.cost_total / m.scored if m.scored else 0.0
    m.critical = sum(1 for r in scored if r.agent_cost.critical)
    m.critical_rate = m.critical / m.scored if m.scored else 0.0
    dis = [r for r in rows if not r.agreed]
    m.disagreements = len(dis)
    m.reviewed = sum(1 for r in dis if r.status == "reviewed")
    m.unreviewed = sum(1 for r in dis if r.status == "unreviewed")
    m.ambiguous = sum(1 for r in dis if r.status == "ambiguous")
    rev = [r for r in dis if r.status == "reviewed"]
    if rev:
        m.agent_right_when_disagreed = sum(1 for r in rev if r.agent_correct) / len(rev)
    m.human_errors = sum(1 for r in scored if r.human_correct is False)
    m.human_error_cost = sum(r.human_cost.cost for r in scored if r.human_cost)
    m.human_accuracy = sum(1 for r in scored if r.human_correct) / m.scored if m.scored else 0.0
    confs = [r.confidence for r in rows if r.confidence is not None]
    m.avg_confidence = sum(confs) / len(confs) if confs else None
    checked = [r for r in rows if r.agreed and r.verdict is not None]
    m.spot_checked = len(checked)
    m.shared_errors = sum(1 for r in checked if r.verdict is Verdict.BOTH_WRONG)
    if checked:
        m.shared_error_rate = m.shared_errors / len(checked)
        m.shared_error_high = wilson(m.shared_errors, len(checked), gates.confidence_level)[1]
    return m


def passes(m: Metrics, gates: ReadinessGates) -> list[str]:
    """Return the reasons a slice fails the gates (empty list = passes)."""
    reasons = []
    if m.acc_low < gates.min_accuracy:
        reasons.append(
            f"accuracy {m.accuracy:.1%} (95% CI lower bound {m.acc_low:.1%}) is below the {gates.min_accuracy:.0%} target"
        )
    if m.critical_rate > gates.max_critical_rate:
        reasons.append(f"critical error rate {m.critical_rate:.1%} exceeds {gates.max_critical_rate:.1%}")
    if m.cost_per_100 > gates.max_cost_per_100:
        reasons.append(f"error cost {m.cost_per_100:.0f} per 100 cases exceeds {gates.max_cost_per_100:.0f}")
    return reasons


@dataclass
class Guardrail:
    threshold: float
    coverage: float
    auto: Metrics


def find_guardrail(rows: list[Row], gates: ReadinessGates) -> Guardrail | None:
    """Lowest confidence threshold where the auto-handled slice passes every gate."""
    confs = sorted({round(r.confidence, 2) for r in rows if r.confidence is not None})
    n = len(rows)
    for t in confs:
        auto = [r for r in rows if r.confidence is not None and r.confidence >= t]
        if len(auto) < gates.min_cases:
            break
        coverage = len(auto) / n
        if coverage < gates.min_guardrail_coverage:
            break
        m = metrics(auto, gates)
        if not passes(m, gates):
            return Guardrail(t, coverage, m)
    return None


@dataclass
class SegmentResult:
    dimension: str
    value: str
    metrics: Metrics
    readiness: Readiness
    reasons: list[str] = field(default_factory=list)
    guardrail: Guardrail | None = None


def assess(rows: list[Row], gates: ReadinessGates, dimension: str, value: str) -> SegmentResult:
    m = metrics(rows, gates)
    if m.scored < gates.min_cases:
        return SegmentResult(dimension, value, m, Readiness.INSUFFICIENT_DATA, [f"only {m.scored} scored cases (need {gates.min_cases})"])
    reasons = passes(m, gates)
    if not reasons:
        return SegmentResult(dimension, value, m, Readiness.GO)
    g = find_guardrail(rows, gates)
    if g:
        return SegmentResult(dimension, value, m, Readiness.GO_WITH_GUARDRAILS, reasons, g)
    return SegmentResult(dimension, value, m, Readiness.NO_GO, reasons)


@dataclass
class CalibrationBucket:
    label: str
    n: int
    accuracy: float
    avg_confidence: float


def calibration(rows: list[Row]) -> list[CalibrationBucket]:
    edges = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
    out = []
    for lo, hi in edges:
        b = [r for r in rows if r.truth is not None and r.confidence is not None and lo <= r.confidence < hi]
        label = f"{lo:.0%}–{min(hi, 1.0):.0%}"
        if b:
            out.append(CalibrationBucket(label, len(b), sum(1 for r in b if r.agent_correct) / len(b), sum(r.confidence for r in b) / len(b)))
        else:
            out.append(CalibrationBucket(label, 0, 0.0, 0.0))
    return out


@dataclass
class RunStats:
    agent_names: list[str]
    models: list[str]
    decided: int
    errors: int
    cost_usd: float
    input_tokens: int
    output_tokens: int
    latency_p50_ms: int | None
    latency_p95_ms: int | None


def run_stats(store: Store) -> RunStats:
    recs = list(store.agent_records().values())
    lat = sorted(r.latency_ms for r in recs if r.latency_ms is not None and r.decision)

    def pct(p: float) -> int | None:
        return lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None

    return RunStats(
        agent_names=sorted({r.agent_name for r in recs}),
        models=sorted({r.model for r in recs if r.model}),
        decided=sum(1 for r in recs if r.decision),
        errors=sum(1 for r in recs if r.error),
        cost_usd=round(sum(r.cost_usd for r in recs), 4),
        input_tokens=sum(r.input_tokens for r in recs),
        output_tokens=sum(r.output_tokens for r in recs),
        latency_p50_ms=pct(0.5),
        latency_p95_ms=pct(0.95),
    )


@dataclass
class Scorecard:
    domain: DomainConfig
    rows: list[Row]
    info: dict[str, int]
    overall: SegmentResult
    readiness: Readiness
    segments: dict[str, list[SegmentResult]]
    calibration: list[CalibrationBucket]
    top_risks: list[Row]
    human_catches: list[Row]
    run: RunStats

    @property
    def provisional(self) -> bool:
        return self.overall.metrics.unreviewed > 0


def scorecard(store: Store, domain: DomainConfig) -> Scorecard:
    gates = domain.readiness
    rows, info = build_rows(store, domain)
    overall = assess(rows, gates, "overall", "All cases")
    segments: dict[str, list[SegmentResult]] = {}
    for spec in domain.segments:
        values: dict[str, list[Row]] = {}
        for r in rows:
            values.setdefault(r.segments[spec.name], []).append(r)
        order = spec.labels or sorted(values)
        segments[spec.name] = [assess(values[v], gates, spec.name, v) for v in order if v in values]

    seg_verdicts = [s.readiness for segs in segments.values() for s in segs if s.readiness is not Readiness.INSUFFICIENT_DATA]
    if overall.readiness is Readiness.INSUFFICIENT_DATA:
        readiness = Readiness.INSUFFICIENT_DATA
    elif overall.readiness is Readiness.GO and all(v is Readiness.GO for v in seg_verdicts):
        readiness = Readiness.GO
    elif overall.readiness in (Readiness.GO, Readiness.GO_WITH_GUARDRAILS) or any(
        v in (Readiness.GO, Readiness.GO_WITH_GUARDRAILS) for v in seg_verdicts
    ):
        readiness = Readiness.GO_WITH_GUARDRAILS
    else:
        readiness = Readiness.NO_GO

    risks = sorted((r for r in rows if r.agent_correct is False), key=lambda r: (-r.agent_cost.cost, -(r.confidence or 0)))
    catches = sorted(
        (r for r in rows if r.human_correct is False and r.agent_correct), key=lambda r: -(r.human_cost.cost if r.human_cost else 0)
    )
    return Scorecard(domain, rows, info, overall, readiness, segments, calibration(rows), risks[:10], catches[:10], run_stats(store))
