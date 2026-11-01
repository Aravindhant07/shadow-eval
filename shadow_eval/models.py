"""Records that flow through the pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Decision(BaseModel):
    """A decision made by either the agent or the human."""

    values: dict[str, Any]
    rationale: str = ""
    confidence: float | None = None  # 0..1, agents only
    citations: list[str] = Field(default_factory=list)


class Case(BaseModel):
    case_id: str
    data: dict[str, Any]
    received_at: str = Field(default_factory=utcnow)
    source: str = "batch"  # batch | live


class AgentRecord(BaseModel):
    case_id: str
    agent_name: str
    decision: Decision | None = None
    error: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: str = Field(default_factory=utcnow)


class HumanRecord(BaseModel):
    case_id: str
    decision: Decision
    decided_by: str = "unknown"
    created_at: str = Field(default_factory=utcnow)


class Verdict(str, Enum):
    AGENT_WRONG = "agent_wrong"
    HUMAN_WRONG = "human_wrong"
    BOTH_ACCEPTABLE = "both_acceptable"
    BOTH_WRONG = "both_wrong"  # reviewer supplies the correct decision
    AMBIGUOUS = "ambiguous"


VERDICT_LABELS = {
    Verdict.AGENT_WRONG: "Agent wrong",
    Verdict.HUMAN_WRONG: "Human wrong",
    Verdict.BOTH_ACCEPTABLE: "Both acceptable",
    Verdict.BOTH_WRONG: "Both wrong",
    Verdict.AMBIGUOUS: "Ambiguous",
}


class Adjudication(BaseModel):
    case_id: str
    suggested_verdict: Verdict | None = None
    suggested_reason: str = ""
    suggested_by: str | None = None
    suggested_correct: dict[str, Any] | None = None  # for both_wrong suggestions
    final_verdict: Verdict | None = None
    correct_values: dict[str, Any] | None = None  # required when final_verdict is both_wrong
    reviewer: str | None = None
    notes: str = ""
    reviewed_at: str | None = None
