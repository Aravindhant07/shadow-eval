# Requirements

## Users

| User | Needs |
|---|---|
| **FDE** (deploys the agent) | Plug in any agent and any decision domain with minimal code |
| **SME reviewer** (e.g. senior claims adjuster) | Resolve agent-vs-human disagreements quickly |
| **Business owner** (e.g. Head of Claims Ops) | A clear verdict: ready / ready for some segments / not ready |
| **Compliance / Risk** | Full audit trail; worst failures surfaced first |

## Confirmed decisions

| Question | Decision |
|---|---|
| Input mode | **Both** historical batch replay and live capture in v1 |
| Scope | **Any domain via YAML config**; insurance claims is the first example |
| Disagreement review | **Web review UI**; AI suggests, SME confirms or overrides |
| Report audience | **Business leaders first**, technical appendix after |
| Models | **Model-agnostic**: Claude (optional), **open-source models via Ollama** (free, local), or any custom agent |
| Cost | The default demo path must cost **$0** — no paid API required |

## Functional requirements

| ID | Requirement |
|---|---|
| R1 | Define the decision schema in YAML (fields, types, allowed values, match tolerance) |
| R2 | Batch replay: load cases and human decisions from CSV/JSON |
| R3 | Live capture: Python decorator + REST API; human decisions can arrive later and are joined by case ID |
| R4 | Agent adapter interface; built-in adapters for Claude, Ollama (open-source models) and a simulated demo agent. The agent never executes actions |
| R5 | Tolerance-aware comparison: field-level rules → match / partial / mismatch |
| R6 | Business cost matrix per (agent decision, correct decision) pair, with critical-error flags |
| R7 | Adjudication: an adjudicator (LLM or rules) suggests agent-wrong / human-wrong / both-acceptable / ambiguous with reasoning; SME confirms in the web UI. Queue sorted by cost |
| R8 | Segment-level scoring: agreement, accuracy with confidence intervals, cost per 100 cases, critical error rate, agent-right-when-disagreed, confidence calibration |
| R9 | Readiness verdict per segment: GO / GO WITH GUARDRAILS / NO-GO / INSUFFICIENT DATA, against configurable gates |
| R10 | Guardrail recommendation: confidence threshold above which the agent can act autonomously, with coverage |
| R11 | Executive HTML report: verdict, segment heatmap, top risks, guardrails, rollout plan, technical appendix |
| R12 | Tamper-evident (hash-chained) audit log of every write; verification command |
| R13 | Synthetic insurance-claims dataset with known ground truth and seeded human errors |
| R14 | Offline demo with no API key |

## Non-functional requirements

- Python 3.11+, installable with `pip install -e .`
- Tests run offline (no network, no API key)
- Agent failures never affect the production path in live mode
- Resumable batch runs (re-running skips already-recorded cases)
- No secrets in the repository; API keys only from the environment

## Out of scope (v1)

- Executing real actions
- Training or fine-tuning the agent
- Replacing production observability platforms
- Multi-tenant auth on the web UI
