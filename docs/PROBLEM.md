# Problem Statement

## Context

An enterprise has built an AI agent — for example, one that triages insurance
claims. The pilot demo went well. But the Head of Operations, Compliance and Risk
will not let it act on real customers yet. They all ask the same question:

> **"How do we know it decides as well as our people do?"**

## Why today's answers fail

| Today's practice | Why it fails |
|---|---|
| Evaluate on a synthetic or hand-picked test set | Doesn't reflect the real, messy distribution of daily cases |
| Score by exact match | "Approve $1,180" vs "Approve $1,200" is not a real disagreement |
| Treat every error the same | Wrongly denying a valid claim costs far more than an unnecessary escalation |
| Assume the human is the ground truth | When agent and human disagree, the **agent is sometimes right** — nobody measures this |
| Present an accuracy number and a demo | Executives need a decision: *go live, for which cases, with what guardrails* |

## Shadow mode

The agent sees the **same real cases** as the human team and records what it
*would* do — but never acts. Its decisions are then compared against what the
humans actually did.

## The gap

Shadow mode exists only inside proprietary platforms (e.g. Microsoft Dynamics 365
Case Management). Open-source tools only **record** agent actions; none compare
them against human decisions, adjudicate who was right, weight errors by business
cost, or produce a go-live readiness verdict.

## What we build

An open-source, model-agnostic tool that:

1. Runs any agent (Claude, local open-source models via Ollama, or your own) in
   shadow mode — from historical data **or** live traffic.
2. Compares agent vs human decisions with tolerance-aware matching.
3. Weights every disagreement by its business cost.
4. Adjudicates who was right — an AI suggests, a subject-matter expert confirms.
5. Produces a **go-live readiness report** a business leader can act on.

## Success criteria

- A business owner reads the report and can answer: *Can we go live? For which
  segments? With what guardrails?*
- On the synthetic benchmark, the tool finds ≥ 90% of deliberately seeded human
  errors and ≥ 90% of agent errors.
- The full demo runs offline, free, with no API key.
