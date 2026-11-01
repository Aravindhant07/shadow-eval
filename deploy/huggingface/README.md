---
title: Shadow Eval
emoji: 🌓
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "6.31.0"
app_file: app.py
pinned: false
license: mit
short_description: Is your AI agent ready to go live? Shadow-mode evaluator.
---

# Shadow Eval: live demo

Runs the offline demo of [shadow-eval](https://github.com/Aravindhant07/shadow-eval) on 600 synthetic
insurance claims, with a simulated agent and a simulated expert. No API key and no cost (free CPU Space).

- **/try**: enter a claim and see what the agent would decide in shadow mode versus the policy
- **/dashboard**: readiness by segment
- **/review**: settle agent-vs-team disagreements
- **/report**: the executive go-live verdict

## Deploying this Space (free Gradio SDK)

```bash
pip install -U huggingface_hub && hf auth login
hf repos create <your-username>/shadow-eval --type space --space-sdk gradio
hf upload <your-username>/shadow-eval . . --repo-type space \
  --exclude ".git/*" "tests/*" "*.db" "docs/images/*" "README.md"
hf upload <your-username>/shadow-eval deploy/huggingface/README.md README.md --repo-type space
```
