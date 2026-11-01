"""Live shadow mode: observe production traffic without affecting it.

    observer = ShadowObserver(store, domain, agent)

    @observer.observe
    def triage_claim(claim: dict) -> str:
        ...  # existing production code, unchanged
        return route_to_adjuster(claim)

    # later, when the human has decided:
    observer.record_human("CLM-10042", {"action": "approve", "payout": 1180}, decided_by="j.okafor")

The decorated function's behaviour and return value never change. The agent runs
in a background thread; if it fails or is slow, production is unaffected.
"""

from __future__ import annotations

import functools
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from ..agents.base import Agent
from ..config import DomainConfig
from ..models import Case, Decision, HumanRecord
from ..store import Store
from .batch import run_agent_on_case

log = logging.getLogger(__name__)
F = TypeVar("F", bound=Callable[..., Any])


class ShadowObserver:
    def __init__(self, store: Store, domain: DomainConfig, agent: Agent, workers: int = 2):
        self.store = store
        self.domain = domain
        self.agent = agent
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="shadow")

    def submit(self, case_data: dict[str, Any]) -> Future | None:
        """Record a case and run the agent on it in the background. Never raises."""
        try:
            case_id = str(case_data[self.domain.case_id_field])
            case = Case(case_id=case_id, data=dict(case_data), source="live")
            self.store.put_case(case)
            return self._pool.submit(self._run, case)
        except Exception:  # never let shadow mode break production
            log.exception("shadow capture failed")
            return None

    def _run(self, case: Case) -> None:
        try:
            self.store.put_agent(run_agent_on_case(self.agent, case))
        except Exception:
            log.exception("shadow agent run failed for %s", case.case_id)

    def observe(self, fn: F | None = None, *, case_arg: int | str = 0) -> F:
        """Decorator. `case_arg` is the position or name of the case argument."""

        def wrap(func: F) -> F:
            @functools.wraps(func)
            def inner(*args: Any, **kwargs: Any) -> Any:
                try:
                    case = kwargs[case_arg] if isinstance(case_arg, str) else args[case_arg]
                    self.submit(case)
                except Exception:
                    log.exception("shadow observe failed")
                return func(*args, **kwargs)

            return inner  # type: ignore[return-value]

        return wrap(fn) if fn is not None else wrap  # type: ignore[return-value]

    def record_human(self, case_id: str, values: dict[str, Any], decided_by: str = "unknown", rationale: str = "") -> None:
        self.store.put_human(HumanRecord(case_id=str(case_id), decision=Decision(values=values, rationale=rationale), decided_by=decided_by))

    def drain(self, timeout: float | None = None) -> None:
        """Wait for in-flight agent runs (useful in tests and scripts)."""
        self._pool.shutdown(wait=True)
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="shadow")
