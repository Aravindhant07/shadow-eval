"""Hugging Face Space entry point (Gradio SDK, free CPU tier).

Builds the offline demo on start-up (synthetic claims, simulated agent and
expert; no API key, no cost) and serves:

  /dashboard  /review  /report   the Shadow Eval web app
  /try                           an interactive Gradio page: decide one claim

Run locally with:  python app.py   then open http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402
import uvicorn  # noqa: E402

from shadow_eval.compare import Outcome, diff, error_cost  # noqa: E402
from shadow_eval.config import load_domain  # noqa: E402
from shadow_eval.demo.build import DEFAULT_DOMAIN, build_demo  # noqa: E402
from shadow_eval.demo.policy import COVERAGE, apply_policy  # noqa: E402
from shadow_eval.demo.simulated import SimulatedAgent  # noqa: E402
from shadow_eval.models import Decision  # noqa: E402
from shadow_eval.web.app import create_app  # noqa: E402

domain = load_domain(DEFAULT_DOMAIN)
store = build_demo(os.environ.get("SHADOW_DB", "/tmp/shadow_demo.db"), domain)
agent = SimulatedAgent()

app = create_app(
    store,
    domain,
    agent=agent,
    banner="Live demo · synthetic insurance claims · simulated agent and expert · resets on restart",
)

CLAIM_TYPES = sorted({t for types in COVERAGE.values() for t in types})


def decide(policy_type, claim_type, amount, limit, deductible, active, docs, police, days, prior, delay):
    case = {
        "claim_id": f"TRY-{abs(hash((policy_type, claim_type, amount, days, prior, delay))) % 10**6}",
        "policy_type": policy_type,
        "claim_type": claim_type,
        "claim_amount": float(amount),
        "coverage_limit": float(limit),
        "deductible": float(deductible),
        "policy_active": bool(active),
        "documents_complete": bool(docs),
        "police_report": bool(police),
        "days_since_policy_start": int(days),
        "prior_claims_12m": int(prior),
        "reported_delay_days": int(delay),
    }
    a = agent.decide(case).decision
    action, payout, rule = apply_policy(case)
    correct = Decision(values={"action": action, "payout": payout})
    agreed = diff(domain, a, correct).outcome is Outcome.MATCH
    cost = error_cost(domain, a, correct)

    def fmt(d: Decision) -> str:
        v = d.values
        return f"**{v['action']}**" + (f" · ${v['payout']:,.2f}" if v["action"] == "approve" else "")

    verdict = "✅ Agent matches the policy." if agreed else (
        f"❌ Agent is wrong: {cost.label} (cost {cost.cost:g}{', critical' if cost.critical else ''})."
    )
    return (
        f"### Agent (shadow mode)\n{fmt(a)} · confidence {a.confidence:.0%}\n\n> {a.rationale}\n\n"
        f"### Correct under policy\n{fmt(correct)}\n\n> {rule}\n\n### Result\n{verdict}\n\n"
        "In shadow mode this decision is only recorded, never executed. "
        "See [/dashboard](/dashboard) for how hundreds of these add up to a go-live verdict."
    )


with gr.Blocks(title="Shadow Eval · try a claim") as ui:
    gr.Markdown(
        "# Shadow Eval: try a claim\n"
        "Is your AI agent ready to go live? Enter a claim to see what the demo agent *would* do in shadow mode, "
        "compared with the policy-correct decision.\n\n"
        "**[Dashboard](/dashboard)** · **[Review queue](/review)** · **[Readiness report](/report)** · "
        "**[GitHub](https://github.com/Aravindhant07/shadow-eval)**"
    )
    with gr.Row():
        with gr.Column():
            policy_type = gr.Dropdown(["auto", "home", "travel"], value="travel", label="Policy type")
            claim_type = gr.Dropdown(CLAIM_TYPES, value="medical_abroad", label="Claim type")
            amount = gr.Number(4800, label="Claim amount ($)")
            limit = gr.Number(10000, label="Coverage limit ($)")
            deductible = gr.Number(100, label="Deductible ($)")
        with gr.Column():
            active = gr.Checkbox(True, label="Policy active")
            docs = gr.Checkbox(True, label="Documents complete")
            police = gr.Checkbox(False, label="Police report")
            days = gr.Number(400, label="Days since policy start", precision=0)
            prior = gr.Number(0, label="Prior claims (12 months)", precision=0)
            delay = gr.Number(5, label="Days between incident and report", precision=0)
    go = gr.Button("Decide in shadow mode", variant="primary")
    out = gr.Markdown()
    inputs = [policy_type, claim_type, amount, limit, deductible, active, docs, police, days, prior, delay]
    go.click(decide, inputs, out)

app = gr.mount_gradio_app(app, ui, path="/try")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
