from .base import Agent, AgentError, AgentResult, LLMAgent, decision_schema, parse_decision

__all__ = ["Agent", "AgentError", "AgentResult", "LLMAgent", "decision_schema", "parse_decision", "make_agent"]


def make_agent(kind: str, domain, model: str | None = None, **kwargs):
    """Build an agent by name: claude | ollama | hf | local | simulated."""
    from ..llm import make_backend

    kind = kind.lower()
    if kind == "simulated":
        from ..demo.simulated import SimulatedAgent

        return SimulatedAgent(**kwargs)
    if kind == "claude":
        model = model or domain.agent.claude_model
        kwargs.setdefault("effort", domain.agent.claude_effort)
    elif kind == "ollama":
        model = model or domain.agent.ollama_model
        kwargs.setdefault("url", domain.agent.ollama_url)
    return LLMAgent(domain, make_backend(kind, model, **kwargs))
