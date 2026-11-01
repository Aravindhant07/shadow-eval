"""Web app: expert review queue, live dashboard, readiness report and REST API."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from ..adjudicate import Adjudicator, record_review, review_queue
from ..agents.base import Agent
from ..capture.batch import run_agent_on_case
from ..config import DomainConfig
from ..models import VERDICT_LABELS, Case, Decision, HumanRecord, Verdict
from ..report import render_report
from ..score import Readiness, scorecard
from ..store import Store

TEMPLATES = Path(__file__).parent / "templates"


def _pct(x: Any, d: int = 1) -> str:
    return "—" if x is None else f"{x * 100:.{d}f}%"


def _num(x: Any, d: int = 0) -> str:
    return "—" if x is None else f"{x:,.{d}f}"


def create_app(
    store: Store,
    domain: DomainConfig,
    agent: Agent | None = None,
    adjudicator: Adjudicator | None = None,
    banner: str | None = None,
) -> FastAPI:
    app = FastAPI(title="shadow-eval", docs_url="/api/docs", redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES))
    templates.env.filters["pct"] = _pct
    templates.env.filters["num"] = _num
    templates.env.globals.update(
        domain=domain, banner=banner, VERDICT_LABELS=VERDICT_LABELS, Verdict=Verdict, Readiness=Readiness
    )

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, ctx)

    # -- pages -----------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/dashboard")

    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard(request: Request) -> HTMLResponse:
        return page(request, "dashboard.html", sc=scorecard(store, domain), counts=store.counts(), nav="dashboard")

    @app.get("/partials/stats", response_class=HTMLResponse)
    def stats_partial(request: Request) -> HTMLResponse:
        return page(request, "_stats.html", sc=scorecard(store, domain), counts=store.counts())

    @app.get("/review", response_class=HTMLResponse)
    def queue(request: Request, show: str = "pending") -> HTMLResponse:
        items = review_queue(store, domain)
        pending = [i for i in items if not (i["adjudication"] and i["adjudication"].final_verdict)]
        shown = {"pending": pending, "reviewed": [i for i in items if i not in pending], "all": items}.get(show, pending)
        return page(request, "queue.html", items=shown, show=show, n_pending=len(pending), n_all=len(items), nav="review")

    @app.get("/review/{case_id}", response_class=HTMLResponse)
    def review_case(request: Request, case_id: str) -> HTMLResponse:
        case = store.get_case(case_id)
        a, h = store.get_agent(case_id), store.get_human(case_id)
        if not case or not a or not h or not a.decision:
            raise HTTPException(404, "case not found or not ready for review")
        item = next((i for i in review_queue(store, domain) if i["case_id"] == case_id), None)
        return page(
            request, "case.html", case=case, agent=a, human=h, item=item,
            adj=store.get_adjudication(case_id), nav="review",
        )

    @app.post("/review/{case_id}")
    async def submit_review(request: Request, case_id: str, verdict: str = Form(...), reviewer: str = Form("expert"), notes: str = Form("")):
        form = await request.form()
        try:
            v = Verdict(verdict)
        except ValueError:
            raise HTTPException(400, f"unknown verdict '{verdict}'")
        correct = None
        if v is Verdict.BOTH_WRONG:
            correct = {}
            for f in domain.decision:
                raw = form.get(f"correct_{f.name}")
                if raw in (None, ""):
                    raise HTTPException(400, f"'both wrong' needs the correct {f.name}")
                correct[f.name] = float(raw) if f.type == "number" else str(raw)
        record_review(store, domain, case_id, v, reviewer.strip() or "expert", notes, correct)
        nxt = next((i for i in review_queue(store, domain) if not (i["adjudication"] and i["adjudication"].final_verdict)), None)
        return RedirectResponse(f"/review/{nxt['case_id']}" if nxt else "/review?show=reviewed", status_code=303)

    @app.get("/report", response_class=HTMLResponse)
    def report() -> HTMLResponse:
        return HTMLResponse(render_report(scorecard(store, domain)))

    # -- API -------------------------------------------------------------
    @app.post("/api/cases", status_code=202)
    def api_case(payload: dict[str, Any], background: BackgroundTasks) -> dict[str, Any]:
        """Live capture: record a case; the agent decides it in the background (shadow mode)."""
        if domain.case_id_field not in payload:
            raise HTTPException(422, f"payload needs '{domain.case_id_field}'")
        case = Case(case_id=str(payload[domain.case_id_field]), data=payload, source="live")
        store.put_case(case)
        if agent is not None:
            background.add_task(lambda: store.put_agent(run_agent_on_case(agent, case)))
        return {"case_id": case.case_id, "status": "recorded", "agent": agent.name if agent else None}

    @app.post("/api/human-decisions/{case_id}")
    def api_human(case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Record what the human decided. Can arrive hours after the case."""
        if store.get_case(case_id) is None:
            raise HTTPException(404, "unknown case")
        values = {}
        for f in domain.decision:
            if f.name not in payload:
                raise HTTPException(422, f"missing '{f.name}'")
            v = payload[f.name]
            values[f.name] = float(v) if f.type == "number" else (str(v).lower() if f.type == "enum" else v)
        store.put_human(HumanRecord(
            case_id=case_id,
            decision=Decision(values=values, rationale=str(payload.get("rationale", ""))),
            decided_by=str(payload.get("decided_by", "unknown")),
        ))
        return {"case_id": case_id, "status": "recorded"}

    @app.post("/api/suggest/{case_id}")
    def api_suggest(case_id: str) -> dict[str, Any]:
        if adjudicator is None:
            raise HTTPException(400, "no adjudicator configured")
        case, a, h = store.get_case(case_id), store.get_agent(case_id), store.get_human(case_id)
        if not case or not a or not a.decision or not h:
            raise HTTPException(404, "case not ready")
        sug = adjudicator.suggest(case.data, a.decision, h.decision)
        existing = store.get_adjudication(case_id)
        if existing and existing.final_verdict:
            existing.suggested_verdict, existing.suggested_reason = sug.suggested_verdict, sug.suggested_reason
            existing.suggested_by, existing.suggested_correct = sug.suggested_by, sug.suggested_correct
            sug = existing
        store.put_adjudication(sug, actor=adjudicator.name)
        return sug.model_dump(mode="json")

    @app.get("/api/metrics")
    def api_metrics() -> JSONResponse:
        sc = scorecard(store, domain)
        m = sc.overall.metrics
        return JSONResponse({
            "readiness": sc.readiness.value,
            "provisional": sc.provisional,
            "overall": m.__dict__,
            "segments": {
                dim: [{"value": s.value, "readiness": s.readiness.value, "n": s.metrics.n, "accuracy": s.metrics.accuracy,
                       "acc_low": s.metrics.acc_low, "critical_rate": s.metrics.critical_rate, "cost_per_100": s.metrics.cost_per_100,
                       "guardrail": ({"threshold": s.guardrail.threshold, "coverage": s.guardrail.coverage} if s.guardrail else None)}
                      for s in segs]
                for dim, segs in sc.segments.items()
            },
            "info": sc.info,
        })

    @app.get("/api/audit/verify")
    def api_verify() -> dict[str, Any]:
        r = store.verify_audit()
        return {"ok": r.ok, "events": r.events, "broken_at": r.broken_at, "reason": r.reason}

    return app
