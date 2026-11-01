"""Executive readiness report (single self-contained HTML file)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .score import Readiness, Row, Scorecard, SegmentResult

TEMPLATES = Path(__file__).parent / "templates"

HEADLINES = {
    Readiness.GO: "Ready to go live.",
    Readiness.GO_WITH_GUARDRAILS: "Ready for a phased go-live with guardrails.",
    Readiness.NO_GO: "Not ready to go live.",
    Readiness.INSUFFICIENT_DATA: "Not enough shadow data to decide yet.",
}


def money(points: float, sc: Scorecard) -> str | None:
    unit = sc.domain.cost_unit_value
    return f"${points * unit:,.0f}" if unit else None


def failure_modes(rows: list[Row], top: int = 3) -> list[tuple[str, int]]:
    c = Counter(r.agent_cost.label for r in rows if r.agent_correct is False)
    return c.most_common(top)


def _seg_name(s: SegmentResult) -> str:
    return f"{s.dimension.replace('_', ' ')}: {s.value}"


def recommendations(sc: Scorecard) -> list[dict[str, str]]:
    recs: list[dict[str, str]] = []
    all_segs = [s for segs in sc.segments.values() for s in segs]
    rows_by_seg = {
        (s.dimension, s.value): [r for r in sc.rows if r.segments.get(s.dimension) == s.value] for s in all_segs
    }
    go = [s for s in all_segs if s.readiness is Readiness.GO]
    if go:
        recs.append({
            "tone": "go",
            "title": "Automate where the agent is already as good as the team",
            "body": "Let the agent decide on its own for " + ", ".join(_seg_name(s) for s in go)
            + ". In each of these, accuracy clears the target even at the bottom of the 95% confidence interval.",
        })
    for s in all_segs:
        if s.readiness is Readiness.GO_WITH_GUARDRAILS and s.guardrail:
            g = s.guardrail
            recs.append({
                "tone": "guard",
                "title": f"Guardrail for {_seg_name(s)}",
                "body": f"Let the agent act alone only when its confidence is at least {g.threshold:.0%}. That covers "
                f"{g.coverage:.0%} of these cases at {g.auto.accuracy:.1%} accuracy; route the rest to people.",
            })
    for s in all_segs:
        if s.readiness is Readiness.NO_GO:
            modes = failure_modes(rows_by_seg[(s.dimension, s.value)])
            mode_txt = f" Most common failure: {modes[0][0].lower()} ({modes[0][1]}×)." if modes else ""
            recs.append({
                "tone": "nogo",
                "title": f"Keep {_seg_name(s)} with people for now",
                "body": f"{s.reasons[0][0].upper() + s.reasons[0][1:]}.{mode_txt} Fix the agent, then re-run shadow mode for this segment.",
            })
    m = sc.overall.metrics
    if m.human_errors:
        value = money(m.human_error_cost, sc)
        recs.append({
            "tone": "info",
            "title": "Use the agent as a second pair of eyes",
            "body": f"Where they disagreed, the agent was right {m.human_errors} times when the team was wrong"
            + (f" (about {value} of avoidable cost)" if value else "")
            + ". Even outside autonomous segments, flagging disagreements for review catches real mistakes.",
        })
    if m.shared_error_high is not None:
        recs.append({
            "tone": "info",
            "title": "Shared mistakes are bounded, not zero",
            "body": f"Experts spot-checked {m.spot_checked} cases where agent and team agreed and found {m.shared_errors} "
            f"shared mistakes. With 95% confidence, shared mistakes affect at most {m.shared_error_high:.1%} of agreements. "
            "Raise the spot-check rate to tighten this bound.",
        })
    if sc.provisional:
        recs.append({
            "tone": "warn",
            "title": "Finish the expert review",
            "body": f"{m.unreviewed} disagreements are not yet reviewed, so they assume the team was right. "
            "This verdict is provisional until they are reviewed.",
        })
    return recs


def rollout_plan(sc: Scorecard) -> list[dict[str, str]]:
    all_segs = [s for segs in sc.segments.values() for s in segs]
    go = [_seg_name(s) for s in all_segs if s.readiness is Readiness.GO]
    guard = [_seg_name(s) for s in all_segs if s.readiness is Readiness.GO_WITH_GUARDRAILS]
    nogo = [_seg_name(s) for s in all_segs if s.readiness is Readiness.NO_GO]
    gates = sc.domain.readiness
    plan = []
    if go:
        plan.append({"phase": "Phase 1 · weeks 1–4", "what": "Autonomous in: " + ", ".join(go) + ". Experts QA a 10% random sample weekly."})
    if guard:
        plan.append({"phase": "Phase 2 · weeks 5–8", "what": "Guardrailed autonomy in: " + ", ".join(guard) + ". Low-confidence cases go to people."})
    if nogo:
        plan.append({"phase": "Phase 3 · after fixes", "what": "Improve the agent for: " + ", ".join(nogo) + ", then re-run shadow mode before any go-live."})
    plan.append({
        "phase": "Always",
        "what": f"Kill switch: return a segment to people if its weekly critical-error rate exceeds {gates.max_critical_rate:.1%}.",
    })
    return plan


def render_report(sc: Scorecard, title: str | None = None) -> str:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x, d=1: "—" if x is None else f"{x * 100:.{d}f}%"
    env.filters["num"] = lambda x, d=0: "—" if x is None else f"{x:,.{d}f}"
    tpl = env.get_template("report.html")
    ctx: dict[str, Any] = {
        "sc": sc,
        "m": sc.overall.metrics,
        "title": title or f"Agent readiness — {sc.domain.name.replace('_', ' ')}",
        "headline": HEADLINES[sc.readiness],
        "recs": recommendations(sc),
        "plan": rollout_plan(sc),
        "money": lambda p: money(p, sc),
        "generated": datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC"),
        "Readiness": Readiness,
        "summary_field": sc.domain.summary_field,
        "primary": sc.domain.primary_field,
        "decision_fields": [f.name for f in sc.domain.decision],
    }
    return tpl.render(**ctx)


def write_report(sc: Scorecard, path: str | Path, title: str | None = None) -> Path:
    path = Path(path)
    path.write_text(render_report(sc, title))
    return path
