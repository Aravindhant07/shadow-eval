"""Domain configuration: everything domain-specific lives in one YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class FieldSpec(BaseModel):
    """One field of the decision the agent and the human both make."""

    name: str
    type: Literal["enum", "number", "boolean", "text"]
    description: str = ""
    values: list[str] | None = None  # allowed values for enum fields
    match: Literal["exact", "tolerance", "ignore"] = "exact"
    tolerance_pct: float = 0.0  # relative tolerance, e.g. 0.05 = 5%
    tolerance_abs: float = 0.0  # absolute tolerance, e.g. 50 (dollars)

    @model_validator(mode="after")
    def _check(self) -> "FieldSpec":
        if self.type == "enum" and not self.values:
            raise ValueError(f"enum field '{self.name}' needs 'values'")
        if self.match == "tolerance" and self.type != "number":
            raise ValueError(f"tolerance matching only applies to number fields ('{self.name}')")
        return self


class CostRule(BaseModel):
    """Cost of the agent choosing `agent` when the correct answer was `truth`.

    Values refer to the primary field. "*" matches any value.
    """

    agent: str
    truth: str
    cost: float
    critical: bool = False
    label: str = ""


class SegmentSpec(BaseModel):
    """A way of slicing cases for readiness, e.g. by policy type or value band."""

    name: str
    field: str  # case field to segment on
    bins: list[float] | None = None  # numeric thresholds (ascending); creates bands
    labels: list[str] | None = None  # one more label than bins

    @model_validator(mode="after")
    def _check(self) -> "SegmentSpec":
        if self.bins is not None:
            if self.bins != sorted(self.bins):
                raise ValueError(f"segment '{self.name}': bins must be ascending")
            if self.labels is None or len(self.labels) != len(self.bins) + 1:
                raise ValueError(f"segment '{self.name}': need len(bins)+1 labels")
        return self

    def value_for(self, case: dict[str, Any]) -> str:
        raw = case.get(self.field)
        if self.bins is None:
            return str(raw) if raw is not None else "unknown"
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return "unknown"
        assert self.labels is not None
        for threshold, label in zip(self.bins, self.labels):
            if x < threshold:
                return label
        return self.labels[-1]


class ReadinessGates(BaseModel):
    min_cases: int = 30
    min_accuracy: float = 0.90  # applied to the Wilson lower bound
    max_critical_rate: float = 0.01
    max_cost_per_100: float = 50.0
    min_guardrail_coverage: float = 0.30  # guardrail must auto-handle >= 30% of volume
    confidence_level: float = 0.95
    # Share of *agreements* sent to the expert as spot checks. Agent and human can
    # make the same mistake; without sampling agreements, shared errors are invisible.
    spot_check_rate: float = 0.05


class AgentSettings(BaseModel):
    system_prompt: str = "You are a careful operations agent. Decide each case by following the policy exactly."
    claude_model: str = "claude-opus-5-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    ollama_model: str = "qwen2.5:7b"
    ollama_url: str = "http://localhost:11434"


class DomainConfig(BaseModel):
    name: str
    description: str = ""
    case_id_field: str = "case_id"
    display_fields: list[str] = Field(default_factory=list)  # case fields shown in the UI/report
    summary_field: str | None = None  # free-text field shown as the case narrative
    decision: list[FieldSpec]
    primary_field: str  # the field the cost matrix is keyed on (e.g. "action")
    cost_matrix: list[CostRule] = Field(default_factory=list)
    default_mismatch_cost: float = 5.0
    partial_mismatch_cost: float = 2.0  # primary field matches, a secondary field doesn't
    segments: list[SegmentSpec] = Field(default_factory=list)
    readiness: ReadinessGates = Field(default_factory=ReadinessGates)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    policy: str = ""  # policy text handed to agents and adjudicators
    currency: str = "USD"
    cost_unit_value: float | None = None  # optional: $ per cost point, for the report

    @model_validator(mode="after")
    def _check(self) -> "DomainConfig":
        names = [f.name for f in self.decision]
        if len(set(names)) != len(names):
            raise ValueError("duplicate decision field names")
        if self.primary_field not in names:
            raise ValueError(f"primary_field '{self.primary_field}' is not a decision field")
        primary = self.field(self.primary_field)
        if primary.type != "enum":
            raise ValueError("primary_field must be an enum field")
        allowed = set(primary.values or []) | {"*"}
        for rule in self.cost_matrix:
            for v in (rule.agent, rule.truth):
                if v not in allowed:
                    raise ValueError(f"cost rule uses unknown value '{v}'")
        return self

    def field(self, name: str) -> FieldSpec:
        for f in self.decision:
            if f.name == name:
                return f
        raise KeyError(name)

    def segments_for(self, case: dict[str, Any]) -> dict[str, str]:
        return {s.name: s.value_for(case) for s in self.segments}

    def cost_rule(self, agent_value: str, truth_value: str) -> CostRule | None:
        """Most specific matching rule wins (exact/exact > exact/* > */exact > */*)."""
        best: tuple[int, CostRule] | None = None
        for rule in self.cost_matrix:
            if rule.agent not in (agent_value, "*") or rule.truth not in (truth_value, "*"):
                continue
            score = (2 if rule.agent == agent_value else 0) + (1 if rule.truth == truth_value else 0)
            if best is None or score > best[0]:
                best = (score, rule)
        return best[1] if best else None


def load_domain(path: str | Path) -> DomainConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    policy_file = raw.pop("policy_file", None)
    if policy_file and not raw.get("policy"):
        raw["policy"] = (path.parent / policy_file).read_text()
    return DomainConfig.model_validate(raw)
