"""Agent adapter interface and the built-in LLM agent."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ..config import DomainConfig
from ..llm import JSONBackend, LLMError
from ..models import Decision


class AgentError(RuntimeError):
    pass


@dataclass
class AgentResult:
    decision: Decision
    model: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class Agent(Protocol):
    """Anything with a `name` and `decide(case) -> AgentResult`.

    Agents only *propose* decisions; shadow mode never executes them. Wrap your
    production agent in a class like this to evaluate it.
    """

    name: str

    def decide(self, case: dict[str, Any]) -> AgentResult: ...


def decision_schema(domain: DomainConfig) -> dict[str, Any]:
    """JSON schema for an agent's answer, built from the domain config."""
    props: dict[str, Any] = {}
    for f in domain.decision:
        if f.type == "enum":
            props[f.name] = {"type": "string", "enum": list(f.values or []), "description": f.description}
        elif f.type == "number":
            props[f.name] = {"type": "number", "description": f.description}
        elif f.type == "boolean":
            props[f.name] = {"type": "boolean", "description": f.description}
        else:
            props[f.name] = {"type": "string", "description": f.description}
    props["rationale"] = {"type": "string", "description": "Short step-by-step explanation against the policy."}
    props["confidence"] = {"type": "number", "description": "Probability between 0 and 1 that this decision is correct."}
    props["policy_citations"] = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Policy rules relied on, e.g. 'Rule 4: fraud signals'.",
    }
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def agent_system_prompt(domain: DomainConfig) -> str:
    fields = "\n".join(
        f"- {f.name} ({f.type}{': ' + ', '.join(f.values) if f.values else ''}): {f.description}" for f in domain.decision
    )
    return (
        f"{domain.agent.system_prompt.strip()}\n\n"
        f"<policy>\n{domain.policy.strip()}\n</policy>\n\n"
        f"Your decision has these fields:\n{fields}\n\n"
        "Also give a short rationale, a calibrated confidence between 0 and 1, and the policy rules "
        "you relied on. Respond only with the JSON object."
    )


def case_prompt(case: dict[str, Any]) -> str:
    return "Decide this case:\n```json\n" + json.dumps(case, indent=2, sort_keys=True, default=str) + "\n```"


def parse_decision(domain: DomainConfig, obj: dict[str, Any]) -> Decision:
    """Validate a model's JSON answer against the domain's decision schema."""
    values: dict[str, Any] = {}
    for f in domain.decision:
        if f.name not in obj:
            raise AgentError(f"missing field '{f.name}'")
        v = obj[f.name]
        if f.type == "enum":
            v = str(v).strip().lower()
            if v not in (f.values or []):
                raise AgentError(f"'{v}' is not an allowed value for '{f.name}'")
        elif f.type == "number":
            try:
                v = float(v)
            except (TypeError, ValueError) as e:
                raise AgentError(f"'{f.name}' is not a number: {v!r}") from e
        elif f.type == "boolean":
            v = bool(v)
        else:
            v = str(v)
        values[f.name] = v
    try:
        conf = obj.get("confidence")
        conf = None if conf is None else min(1.0, max(0.0, float(conf)))
    except (TypeError, ValueError):
        conf = None
    cites = obj.get("policy_citations") or []
    if not isinstance(cites, list):
        cites = [cites]
    return Decision(values=values, rationale=str(obj.get("rationale", "")), confidence=conf, citations=[str(c) for c in cites])


class LLMAgent:
    """An agent backed by any JSON backend: Claude, Ollama or Hugging Face."""

    def __init__(self, domain: DomainConfig, backend: JSONBackend):
        self.domain = domain
        self.backend = backend
        self.name = backend.name
        self._system = agent_system_prompt(domain)
        self._schema = decision_schema(domain)

    def decide(self, case: dict[str, Any]) -> AgentResult:
        try:
            r = self.backend.complete(self._system, case_prompt(case), self._schema)
        except LLMError as e:
            raise AgentError(str(e)) from e
        return AgentResult(parse_decision(self.domain, r.data), r.model, r.input_tokens, r.output_tokens, r.cost_usd)
