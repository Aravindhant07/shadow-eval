---
title: Shadow Eval
emoji: 🌓
colorFrom: indigo
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Is your AI agent ready to go live? Shadow-mode readiness evaluator.
---

# Shadow Eval: live demo

Runs the offline demo of [shadow-eval](https://github.com/Aravindhant07/shadow-eval) on 600 synthetic
insurance claims, with a simulated agent and a simulated expert. No API key and no cost.

- **Dashboard:** readiness by segment
- **Review queue:** settle agent-vs-team disagreements
- **Readiness report:** the executive go-live verdict

## Deploying this Space

1. Create a new Space on huggingface.co. Choose **Docker** as the SDK and the free **CPU basic** hardware.
2. Push the whole repository to the Space, with this file as the Space's `README.md`:

```bash
git clone https://github.com/Aravindhant07/shadow-eval && cd shadow-eval
cp deploy/huggingface/README.md README.md
git remote add space https://huggingface.co/spaces/<your-username>/shadow-eval
git commit -am "Hugging Face Space config" && git push space main
```
