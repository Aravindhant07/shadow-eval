"""Model backends that return a JSON object matching a schema.

One interface, three providers:

* ClaudeBackend      — Anthropic API (optional, paid). `pip install anthropic`;
                       credentials from ANTHROPIC_API_KEY or an `ant auth login` profile.
* OllamaBackend      — open-source models running locally via Ollama (free, private).
* HuggingFaceBackend — open-source models via Hugging Face Inference Providers.
                       Needs HF_TOKEN; free accounts get a small monthly credit.

Agents and adjudicators are built on top of these, so every component works with
every provider.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    data: dict[str, Any]
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class JSONBackend(Protocol):
    name: str

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult: ...


# -- helpers --------------------------------------------------------------

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating code fences or stray prose around it."""
    text = (text or "").strip()
    for candidate in (text, _FENCE.sub("", text).strip()):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    raise LLMError(f"model did not return a JSON object: {text[:200]!r}")


def post_json(url: str, body: dict[str, Any], headers: dict[str, str] | None = None, timeout: float = 120) -> dict[str, Any]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")[:300]
        raise LLMError(f"HTTP {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"cannot reach {url}: {e.reason}") from e


# -- Claude ---------------------------------------------------------------

# USD per million tokens (input, output).
CLAUDE_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


class ClaudeBackend:
    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low", client: Any = None, max_tokens: int = 8000):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.name = f"claude:{model}"
        if client is None:
            try:
                import anthropic
            except ImportError as e:  # pragma: no cover
                raise LLMError("Install the Claude extra: pip install 'shadow-eval[claude]'") from e
            client = anthropic.Anthropic()
        self.client = client

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            # The system prompt (instructions + policy) is identical for every case: cache it.
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": schema}},
            # If a safety classifier declines, the API re-runs on its recommended fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            category = getattr(getattr(response, "stop_details", None), "category", None)
            raise LLMError(f"model declined (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError("response hit max_tokens before finishing")
        text = next((b.text for b in response.content if getattr(b, "type", None) == "text"), None)
        if text is None:
            raise LLMError("no text block in response")
        served = getattr(response, "model", None) or self.model
        usage = response.usage
        pin, pout = CLAUDE_PRICES.get(served, CLAUDE_PRICES.get(self.model, (0.0, 0.0)))
        cost = (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000
        return LLMResult(parse_json_object(text), served, usage.input_tokens, usage.output_tokens, round(cost, 6))


# -- Ollama (local open-source models) ------------------------------------

class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:7b", url: str | None = None, timeout: float = 300):
        self.model = model
        url = (url or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        self.url = url if url.startswith("http") else "http://" + url
        self.timeout = timeout
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        data = post_json(
            f"{self.url}/api/chat",
            {
                "model": self.model,
                "stream": False,
                "format": schema,  # structured outputs: constrain generation to the schema
                "options": {"temperature": 0},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            timeout=self.timeout,
        )
        if "error" in data:
            raise LLMError(f"ollama: {data['error']}")
        content = (data.get("message") or {}).get("content", "")
        return LLMResult(
            parse_json_object(content),
            data.get("model", self.model),
            int(data.get("prompt_eval_count") or 0),
            int(data.get("eval_count") or 0),
            0.0,
        )


# -- Hugging Face Inference Providers -------------------------------------

HF_ROUTER = "https://router.huggingface.co/v1/chat/completions"


class HuggingFaceBackend:
    def __init__(self, model: str = "Qwen/Qwen2.5-72B-Instruct", token: str | None = None, url: str = HF_ROUTER, timeout: float = 120):
        self.model = model
        self.token = token or os.environ.get("HF_TOKEN")
        if not self.token:
            raise LLMError("Set HF_TOKEN to a Hugging Face access token (huggingface.co/settings/tokens)")
        self.url = url
        self.timeout = timeout
        self.name = f"hf:{model}"

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        data = post_json(
            self.url,
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": 1024,
                "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "schema": schema, "strict": True}},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=self.timeout,
        )
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"unexpected response from Hugging Face: {str(data)[:200]}") from e
        usage = data.get("usage") or {}
        return LLMResult(
            parse_json_object(content),
            data.get("model", self.model),
            int(usage.get("prompt_tokens") or 0),
            int(usage.get("completion_tokens") or 0),
            0.0,  # billed against HF credits; not tracked here
        )


# -- Local open-source models via Hugging Face transformers ---------------

class TransformersBackend:
    """Downloads an open model from the Hugging Face Hub and runs it locally.

    Free, offline after the first download, no token or credits needed.
    `pip install torch transformers` (CPU wheels are fine for small models).
    """

    def __init__(self, model: str = "Qwen/Qwen2.5-1.5B-Instruct", max_new_tokens: int = 400):
        self.model_id = model
        self.max_new_tokens = max_new_tokens
        self.name = f"local:{model}"
        self._tok = None
        self._model = None
        import threading

        self._lock = threading.Lock()  # one generation at a time

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover
            raise LLMError("Install the local extra: pip install 'shadow-eval[local]'") from e
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype="auto")
        self._model.eval()

    def complete(self, system: str, user: str, schema: dict[str, Any]) -> LLMResult:
        import torch

        keys = ", ".join(
            f'"{k}"' + (f" (one of: {', '.join(v['enum'])})" if "enum" in v else f" ({v.get('type')})")
            for k, v in schema.get("properties", {}).items()
        )
        sys_prompt = f"{system}\n\nRespond with ONLY a JSON object with exactly these keys: {keys}."
        with self._lock:
            self._load()
            assert self._tok is not None and self._model is not None
            prompt = self._tok.apply_chat_template(
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                add_generation_prompt=True,
                tokenize=False,
            )
            inputs = self._tok(prompt, return_tensors="pt")
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=self._tok.eos_token_id
                )
            new = out[0][inputs["input_ids"].shape[1]:]
            text = self._tok.decode(new, skip_special_tokens=True)
        return LLMResult(parse_json_object(text), self.model_id, int(inputs["input_ids"].shape[1]), int(len(new)), 0.0)


def make_backend(provider: str, model: str | None = None, **kwargs: Any) -> JSONBackend:
    provider = provider.lower()
    if provider in ("local", "transformers"):
        return TransformersBackend(model=model or "Qwen/Qwen2.5-1.5B-Instruct", **kwargs)
    if provider == "claude":
        return ClaudeBackend(model=model or "claude-opus-5-5", **kwargs)
    if provider == "ollama":
        return OllamaBackend(model=model or "qwen2.5:7b", **kwargs)
    if provider in ("hf", "huggingface"):
        return HuggingFaceBackend(model=model or "Qwen/Qwen2.5-72B-Instruct", **kwargs)
    raise LLMError(f"unknown provider '{provider}' (use claude, ollama or hf)")
