import sqlite3

import pytest

from shadow_eval.compare import Outcome, diff, error_cost
from shadow_eval.config import DomainConfig, SegmentSpec
from shadow_eval.models import Case, Decision
from shadow_eval.score import wilson

from .conftest import claim


def D(action, payout=0.0):
    return Decision(values={"action": action, "payout": payout})


def test_domain_loads_policy_and_validates(domain):
    assert domain.primary_field == "action"
    assert "Fraud signals" in domain.policy
    with pytest.raises(ValueError):
        DomainConfig(name="x", decision=[{"name": "a", "type": "number"}], primary_field="a")


def test_segment_bins(domain):
    band = next(s for s in domain.segments if s.name == "value_band")
    assert band.value_for({"claim_amount": 500}) == "< $2k"
    assert band.value_for({"claim_amount": 2000}) == "$2k–10k"
    assert band.value_for({"claim_amount": 99999}) == "> $25k"
    with pytest.raises(ValueError):
        SegmentSpec(name="b", field="x", bins=[2, 1], labels=["a", "b", "c"])


def test_tolerance_matching(domain):
    assert diff(domain, D("approve", 1180), D("approve", 1200)).outcome is Outcome.MATCH  # within $50
    assert diff(domain, D("approve", 10400), D("approve", 10000)).outcome is Outcome.MATCH  # within 5%
    assert diff(domain, D("approve", 10600), D("approve", 10000)).outcome is Outcome.PARTIAL
    assert diff(domain, D("deny"), D("approve", 10000)).outcome is Outcome.MISMATCH
    assert diff(domain, D("APPROVE ", 100), D("approve", 100)).outcome is Outcome.MATCH


def test_cost_matrix_specificity(domain):
    c = error_cost(domain, D("approve", 900), D("flag_fraud"))
    assert c.cost == 25 and c.critical
    c = error_cost(domain, D("deny"), D("approve", 900))
    assert c.cost == 20 and c.critical
    assert error_cost(domain, D("deny"), D("escalate")).cost == 8  # deny/* rule
    assert error_cost(domain, D("escalate"), D("approve", 50)).cost == 1
    assert error_cost(domain, D("approve", 2000), D("approve", 1000)).cost == domain.partial_mismatch_cost
    assert error_cost(domain, D("approve", 1000), D("approve", 1000)).cost == 0


def test_wilson_interval():
    lo, hi = wilson(90, 100)
    assert 0.82 < lo < 0.84 and 0.94 < hi < 0.95
    assert wilson(0, 0) == (0.0, 0.0)


def test_audit_chain_detects_tampering(store):
    store.put_case(Case(case_id="A", data=claim(claim_id="A")))
    store.put_case(Case(case_id="B", data=claim(claim_id="B")))
    assert store.verify_audit().ok
    conn = sqlite3.connect(store.path)
    conn.execute("UPDATE audit_log SET payload = '{\"edited\": true}' WHERE seq = 1")
    conn.commit()
    r = store.verify_audit()
    assert not r.ok and r.broken_at == 1 and "edited" in r.reason


def test_audit_chain_detects_deletion(store):
    for i in range(3):
        store.put_case(Case(case_id=str(i), data=claim(claim_id=str(i))))
    conn = sqlite3.connect(store.path)
    conn.execute("DELETE FROM audit_log WHERE seq = 2")
    conn.commit()
    assert not store.verify_audit().ok
