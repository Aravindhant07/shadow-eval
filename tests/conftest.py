from pathlib import Path

import pytest

from shadow_eval.config import load_domain
from shadow_eval.store import Store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def domain():
    return load_domain(ROOT / "domains" / "insurance_claims.yaml")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.db")
    yield s
    s.close()


def claim(**overrides):
    c = {
        "claim_id": "CLM-1",
        "policy_type": "auto",
        "claim_type": "collision",
        "claim_amount": 4000.0,
        "coverage_limit": 50000.0,
        "deductible": 500.0,
        "policy_active": True,
        "documents_complete": True,
        "police_report": True,
        "days_since_policy_start": 400,
        "prior_claims_12m": 0,
        "reported_delay_days": 3,
        "description": "Rear-ended at a junction.",
    }
    c.update(overrides)
    return c
