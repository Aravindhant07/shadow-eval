"""Export a static website (free to host anywhere, e.g. a Hugging Face Static Space
or GitHub Pages): a landing page with results, an interactive "try a claim"
widget, a read-only review sample, and the full readiness report."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import DomainConfig
from .report import HEADLINES, render_report
from .score import Scorecard

TEMPLATES = Path(__file__).parent / "templates"


def _stats_by_type(sc: Scorecard) -> dict[str, Any]:
    """Agent accuracy by (claim_type, correct action) from the shadow run, for the widget."""
    groups: dict[str, list] = defaultdict(list)
    for r in sc.rows:
        if r.truth is None:
            continue
        key = f"{r.case.get('claim_type')}|{r.truth.values.get(sc.domain.primary_field)}"
        groups[key].append(r)
    out = {}
    for key, rows in groups.items():
        wrong = [r for r in rows if r.agent_correct is False]
        top = Counter(r.agent_cost.label for r in wrong).most_common(1)
        out[key] = {
            "n": len(rows),
            "correct": len(rows) - len(wrong),
            "top_mistake": top[0][0] if top else None,
            "top_mistake_n": top[0][1] if top else 0,
        }
    return out


def _review_sample(sc: Scorecard, n: int = 12) -> list[dict[str, Any]]:
    rows = [r for r in sc.rows if not r.agreed]
    rows.sort(key=lambda r: (-(r.agent_cost.cost if r.agent_correct is False else (r.human_cost.cost if r.human_cost else 0)), r.case_id))
    out = []
    for r in rows[:n]:
        out.append({
            "case_id": r.case_id,
            "summary": r.case.get(sc.domain.summary_field or "", ""),
            "segments": " · ".join(r.segments.values()),
            "agent": r.agent.values,
            "agent_conf": r.agent.confidence,
            "agent_why": r.agent.rationale,
            "human": r.human.values,
            "verdict": r.verdict.value.replace("_", " ") if r.verdict else "awaiting review",
            "who_right": "agent" if r.agent_correct and r.human_correct is False else ("team" if r.human_correct and r.agent_correct is False else "both/neither"),
        })
    return out


def export_static(sc: Scorecard, out_dir: str | Path, repo_url: str = "https://github.com/Aravindhant07/shadow-eval") -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.html").write_text(render_report(sc))
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["pct"] = lambda x, d=1: "—" if x is None else f"{x * 100:.{d}f}%"
    env.filters["num"] = lambda x, d=0: "—" if x is None else f"{x:,.{d}f}"
    domain: DomainConfig = sc.domain
    html = env.get_template("site.html").render(
        sc=sc,
        m=sc.overall.metrics,
        headline=HEADLINES[sc.readiness],
        repo_url=repo_url,
        review=_review_sample(sc),
        stats_json=json.dumps(_stats_by_type(sc)),
        cost_json=json.dumps([r.model_dump() for r in domain.cost_matrix]),
    )
    (out / "index.html").write_text(html)
    return out
