"""End-to-end: synthetic data → shadow run → adjudication → scoring → report → web."""

import csv

import pytest
from fastapi.testclient import TestClient

from shadow_eval.adjudicate import record_review, review_queue
from shadow_eval.capture.live import ShadowObserver
from shadow_eval.demo.build import build_demo
from shadow_eval.demo.simulated import SimulatedAgent
from shadow_eval.models import Verdict
from shadow_eval.report import render_report
from shadow_eval.score import Readiness, scorecard
from shadow_eval.web.app import create_app

from .conftest import claim


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from shadow_eval.config import load_domain
    from shadow_eval.demo.build import DEFAULT_DOMAIN

    d = tmp_path_factory.mktemp("demo")
    domain = load_domain(DEFAULT_DOMAIN)
    store = build_demo(d / "demo.db", domain, data_dir=d / "data", n=600)
    with open(d / "data" / "truth.csv", newline="") as fh:
        truth = {r["claim_id"]: r for r in csv.DictReader(fh)}
    return store, domain, truth


def test_finds_seeded_human_errors(demo):
    store, domain, truth = demo
    sc = scorecard(store, domain)
    seeded = {k for k, r in truth.items() if r["seeded_human_error"]}
    found = {r.case_id for r in sc.rows if r.human_correct is False}
    assert len(seeded & found) / len(seeded) >= 0.90  # success criterion
    assert not (found - seeded)  # no false accusations


def test_finds_agent_errors(demo):
    store, domain, truth = demo
    sc = scorecard(store, domain)
    wrong = set()
    for cid, a in store.agent_records().items():
        t = truth[cid]
        if a.decision.values["action"] != t["action"] or abs(a.decision.values["payout"] - float(t["payout"])) > max(50, 0.05 * float(t["payout"])):
            wrong.add(cid)
    flagged = {r.case_id for r in sc.rows if r.agent_correct is False}
    assert len(wrong & flagged) / len(wrong) >= 0.90


def test_verdicts_tell_a_coherent_story(demo):
    store, domain, _ = demo
    sc = scorecard(store, domain)
    by = {s.value: s for s in sc.segments["policy_type"]}
    assert by["auto"].readiness is Readiness.GO
    assert by["travel"].readiness is Readiness.NO_GO
    assert sc.readiness is Readiness.GO_WITH_GUARDRAILS
    assert any(s.guardrail for segs in sc.segments.values() for s in segs)
    assert store.verify_audit().ok


def test_report_renders(demo):
    store, domain, _ = demo
    html = render_report(scorecard(store, domain))
    assert "GO WITH GUARDRAILS" in html and "Top risks" in html and "Rollout plan" in html


def test_unreviewed_disagreements_make_verdict_provisional(tmp_path, domain):
    from shadow_eval.capture.batch import run_shadow
    from shadow_eval.models import Case, Decision, HumanRecord
    from shadow_eval.store import Store

    s = Store(tmp_path / "p.db")
    s.put_case(Case(case_id="CLM-1", data=claim()))
    s.put_human(HumanRecord(case_id="CLM-1", decision=Decision(values={"action": "deny", "payout": 0})))
    run_shadow(s, SimulatedAgent())
    sc = scorecard(s, domain)
    assert sc.provisional
    record_review(s, domain, "CLM-1", Verdict.HUMAN_WRONG, "tester")
    sc = scorecard(s, domain)
    assert not sc.provisional and sc.rows[0].agent_correct and sc.rows[0].human_correct is False
    with pytest.raises(ValueError):
        record_review(s, domain, "CLM-1", Verdict.BOTH_WRONG, "tester")


def test_live_observer_never_changes_production(tmp_path, domain):
    from shadow_eval.store import Store

    s = Store(tmp_path / "live.db")
    obs = ShadowObserver(s, domain, SimulatedAgent())

    @obs.observe
    def production(c):
        return "routed-to-adjuster"

    assert production(claim(claim_id="CLM-77")) == "routed-to-adjuster"
    assert production({"no_id": True}) == "routed-to-adjuster"  # bad input: shadow fails silently

    class Boom:
        name = "boom"

        def decide(self, case):
            raise RuntimeError("model down")

    obs2 = ShadowObserver(s, domain, Boom())

    @obs2.observe
    def production2(c):
        return "routed-to-adjuster"

    assert production2(claim(claim_id="CLM-78")) == "routed-to-adjuster"
    obs.drain()
    obs2.drain()
    obs.record_human("CLM-77", {"action": "approve", "payout": 3500}, decided_by="j.okafor")
    assert s.get_agent("CLM-77").decision is not None
    assert s.get_agent("CLM-78").error.startswith("RuntimeError")
    assert s.get_human("CLM-77").decided_by == "j.okafor"


def test_web_review_flow_and_api(tmp_path, demo):
    store, domain, _ = demo
    client = TestClient(create_app(store, domain, agent=SimulatedAgent()))
    assert client.get("/dashboard").status_code == 200
    assert "Review queue" in client.get("/review").text
    pending = [i for i in review_queue(store, domain) if not (i["adjudication"] and i["adjudication"].final_verdict)]
    cid = pending[0]["case_id"]
    page = client.get(f"/review/{cid}")
    assert page.status_code == 200 and cid in page.text
    r = client.post(f"/review/{cid}", data={"verdict": "both_acceptable", "reviewer": "tester"}, follow_redirects=False)
    assert r.status_code == 303
    assert store.get_adjudication(cid).final_verdict is Verdict.BOTH_ACCEPTABLE
    bad = client.post(f"/review/{cid}", data={"verdict": "both_wrong"})
    assert bad.status_code == 400

    # live capture through the API: case first, human decision later
    assert client.post("/api/cases", json=claim(claim_id="LIVE-1")).status_code == 202
    assert store.get_agent("LIVE-1").decision is not None  # background task ran
    assert client.post("/api/human-decisions/LIVE-1", json={"action": "approve", "payout": 3500, "decided_by": "s.chen"}).status_code == 200
    assert client.post("/api/human-decisions/NOPE", json={"action": "approve", "payout": 1}).status_code == 404
    m = client.get("/api/metrics").json()
    assert m["readiness"] in {r.value for r in Readiness}
    assert client.get("/api/audit/verify").json()["ok"]
    assert "Rollout plan" in client.get("/report").text


def test_static_site_export(tmp_path, demo):
    from shadow_eval.static_site import export_static

    store, domain, _ = demo
    out = export_static(scorecard(store, domain), tmp_path / "site")
    index = (out / "index.html").read_text()
    assert "Try a claim" in index and "const STATS" in index and "medical_abroad|approve" in index
    assert "Rollout plan" in (out / "report.html").read_text()
