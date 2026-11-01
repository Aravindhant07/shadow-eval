"""Deterministic, tolerance-aware comparison of two decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .config import DomainConfig, FieldSpec
from .models import Decision


class Outcome(str, Enum):
    MATCH = "match"  # every compared field agrees
    PARTIAL = "partial"  # primary field agrees, a secondary field differs
    MISMATCH = "mismatch"  # primary field differs


@dataclass
class FieldResult:
    name: str
    a: Any
    b: Any
    equal: bool
    note: str = ""


@dataclass
class Diff:
    outcome: Outcome
    fields: list[FieldResult] = field(default_factory=list)

    @property
    def differing(self) -> list[FieldResult]:
        return [f for f in self.fields if not f.equal]


def _norm(v: Any) -> Any:
    if isinstance(v, str):
        return v.strip().lower()
    return v


def _to_float(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def field_equal(spec: FieldSpec, a: Any, b: Any) -> tuple[bool, str]:
    if spec.match == "ignore":
        return True, "ignored"
    if spec.type == "number":
        fa, fb = _to_float(a), _to_float(b)
        if fa is None or fb is None:
            return (fa is None and fb is None), "missing value"
        if spec.match == "tolerance":
            allowed = max(spec.tolerance_abs, spec.tolerance_pct * abs(fb))
            delta = abs(fa - fb)
            return delta <= allowed + 1e-9, f"Δ={delta:,.2f} (allowed {allowed:,.2f})"
        return fa == fb, ""
    if spec.type == "boolean":
        return bool(a) == bool(b), ""
    return _norm(a) == _norm(b), ""


def diff(domain: DomainConfig, a: Decision, b: Decision) -> Diff:
    """Compare decision `a` against decision `b`."""
    results = []
    for spec in domain.decision:
        va, vb = a.values.get(spec.name), b.values.get(spec.name)
        eq, note = field_equal(spec, va, vb)
        results.append(FieldResult(spec.name, va, vb, eq, note))
    primary_ok = next(r.equal for r in results if r.name == domain.primary_field)
    if not primary_ok:
        outcome = Outcome.MISMATCH
    elif all(r.equal for r in results):
        outcome = Outcome.MATCH
    else:
        outcome = Outcome.PARTIAL
    return Diff(outcome, results)


@dataclass
class ErrorCost:
    cost: float
    critical: bool
    label: str


def error_cost(domain: DomainConfig, predicted: Decision, truth: Decision) -> ErrorCost:
    """Business cost of `predicted` when `truth` was correct. Zero if they agree."""
    d = diff(domain, predicted, truth)
    if d.outcome is Outcome.MATCH:
        return ErrorCost(0.0, False, "")
    if d.outcome is Outcome.PARTIAL:
        names = ", ".join(f.name for f in d.differing)
        return ErrorCost(domain.partial_mismatch_cost, False, f"{names} off")
    pv = _norm(predicted.values.get(domain.primary_field))
    tv = _norm(truth.values.get(domain.primary_field))
    rule = domain.cost_rule(str(pv), str(tv))
    if rule is None:
        return ErrorCost(domain.default_mismatch_cost, False, f"{pv} instead of {tv}")
    return ErrorCost(rule.cost, rule.critical, rule.label or f"{pv} instead of {tv}")
