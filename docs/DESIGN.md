# Design

## Architecture

```
            ┌──────────── Domain config (YAML) ─────────────┐
            │ decision schema · tolerances · cost matrix    │
            │ segments · readiness gates · policy text      │
            └───────────────────────┬───────────────────────┘
                                    │
   CAPTURE                          ▼
 ┌──────────────────┐     ┌──────────────────┐
 │ Batch replay     │     │ Live capture     │
 │ CSV/JSON cases + │     │ @observe + REST; │
 │ human decisions  │     │ human decisions  │
 └────────┬─────────┘     │ joined later     │
          └───────┬───────┴────────┬─────────┘
                  ▼
        ┌───────────────────┐   Agent adapter: decide(case) -> Decision
        │ Shadow runner     │◄─ ClaudeAgent · OllamaAgent (open-source)
        │ (never acts)      │   · SimulatedAgent · your own
        └─────────┬─────────┘
                  ▼
   ┌──────────────────────────────────────────────┐
   │ SQLite store + hash-chained audit log        │
   └──────────────────────┬───────────────────────┘
                          ▼
   Comparator ─► Adjudicator ─► Scorer ─► Readiness report (HTML)
   (field rules)  (LLM/rules      (segments, costs,
                   suggest; SME    Wilson CIs,
                   confirms)       calibration)
                          ▲
                  Web review UI (FastAPI)
```

## Key decisions

| Decision | Choice | Why |
|---|---|---|
| Model access | Adapter interface; Claude and Ollama built in | Model-agnostic; open-source path is free and local |
| Structured output | Claude: JSON-schema `output_config.format`; Ollama: `format` JSON schema | Decisions always parse into the configured schema |
| Comparison | Deterministic code, not an LLM | Cheap, reproducible, auditable |
| Ground truth | SME-confirmed verdict; otherwise human decision, flagged *unreviewed* | Never silently assume the human is right |
| Statistics | Wilson score interval; GO requires the **lower bound** to clear the gate | Small segments can't fake readiness |
| Storage | SQLite, JSON payload columns | Zero setup; portable demo |
| Audit | Append-only table, SHA-256 chain over (prev_hash, event) | Any edit or deletion is detectable |
| Web | FastAPI + Jinja2 + HTMX (dashboard refresh only) | No frontend build step |

## Ground-truth resolution

| SME verdict | Correct decision used for scoring |
|---|---|
| `agent_wrong` | Human decision |
| `human_wrong` | Agent decision |
| `both_acceptable` | Both correct (cost 0) |
| `ambiguous` | Excluded from accuracy, reported separately |
| *(not reviewed)* | Human decision, flagged unreviewed; the report says the verdict is provisional |

## Readiness logic (per segment)

1. `n < min_cases` → **INSUFFICIENT DATA**
2. Accuracy lower bound ≥ `min_accuracy`, critical-error rate ≤ `max_critical_rate`,
   cost per 100 ≤ `max_cost_per_100` → **GO**
3. A confidence threshold exists where the auto-handled slice passes the same gates
   with coverage ≥ `min_guardrail_coverage` → **GO WITH GUARDRAILS**
4. Otherwise → **NO-GO**

Overall: GO if every segment is GO; NO-GO if none is GO or GO WITH GUARDRAILS;
otherwise GO WITH GUARDRAILS (phased rollout).

## Module map

| Module | Responsibility |
|---|---|
| `config.py` | Load and validate the domain YAML |
| `models.py` | Pydantic records: Decision, AgentRecord, HumanRecord, Adjudication |
| `store.py` / `audit.py` | SQLite persistence and the hash-chained audit log |
| `capture/batch.py` / `capture/live.py` | Batch replay; live decorator and observer |
| `agents/` | `ClaudeAgent`, `OllamaAgent`, `SimulatedAgent`, base protocol |
| `compare.py` | Field-level diff and error cost |
| `adjudicators/` | `LLMAdjudicator` (Claude or Ollama), domain rules adjudicator |
| `score.py` | Segment metrics, calibration, guardrails, verdicts |
| `report.py` | Executive HTML report |
| `web/` | Review UI, dashboard, REST API |
| `cli.py` | `shadow` command |
| `demo/` | Synthetic claims generator, policy rules, simulated agent and SME |
