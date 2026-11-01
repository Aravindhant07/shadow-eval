"""Model adapters, tested offline with stubbed providers."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace

import pytest

from shadow_eval.adjudicate import LLMAdjudicator
from shadow_eval.agents import AgentError, LLMAgent, decision_schema, parse_decision
from shadow_eval.llm import ClaudeBackend, HuggingFaceBackend, LLMError, LLMResult, OllamaBackend, parse_json_object
from shadow_eval.models import Decision, Verdict

from .conftest import claim

ANSWER = {"action": "approve", "payout": 3500, "rationale": "Rule 6", "confidence": 0.93, "policy_citations": ["Rule 6"]}


def test_schema_is_strict(domain):
    s = decision_schema(domain)
    assert s["additionalProperties"] is False
    assert set(s["required"]) == {"action", "payout", "rationale", "confidence", "policy_citations"}
    assert s["properties"]["action"]["enum"] == ["approve", "deny", "request_info", "escalate", "flag_fraud"]


def test_parse_decision_validates(domain):
    d = parse_decision(domain, {**ANSWER, "action": "Approve", "confidence": 1.7})
    assert d.values == {"action": "approve", "payout": 3500.0} and d.confidence == 1.0
    with pytest.raises(AgentError):
        parse_decision(domain, {**ANSWER, "action": "pay"})
    with pytest.raises(AgentError):
        parse_decision(domain, {"action": "deny"})


def test_parse_json_tolerates_fences():
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! {"a": 2} hope that helps') == {"a": 2}
    with pytest.raises(LLMError):
        parse_json_object("no json here")


# -- Claude (stubbed SDK client) ---------------------------------------------

class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def fake_claude(text=None, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text or json.dumps(ANSWER))]
    resp = SimpleNamespace(
        content=content, stop_reason=stop_reason, stop_details=SimpleNamespace(category="cyber"),
        model="claude-opus-5-5", usage=SimpleNamespace(input_tokens=1000, output_tokens=200),
    )
    msgs = FakeMessages(resp)
    return SimpleNamespace(beta=SimpleNamespace(messages=msgs)), msgs


def test_claude_agent_request_and_cost(domain):
    client, msgs = fake_claude()
    agent = LLMAgent(domain, ClaudeBackend(client=client, effort="low"))
    r = agent.decide(claim())
    assert r.decision.values["action"] == "approve" and r.decision.confidence == 0.93
    k = msgs.kwargs
    assert k["model"] == "claude-opus-5-5"
    assert k["output_config"]["format"]["type"] == "json_schema"
    assert k["output_config"]["effort"] == "low"
    assert k["fallbacks"] == "default" and k["betas"] == ["server-side-fallback-2026-07-01"]
    assert k["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "Fraud signals" in k["system"][0]["text"]
    assert r.cost_usd == pytest.approx((1000 * 4 + 200 * 20) / 1e6)


def test_claude_refusal_becomes_agent_error(domain):
    client, _ = fake_claude(stop_reason="refusal")
    with pytest.raises(AgentError, match="declined"):
        LLMAgent(domain, ClaudeBackend(client=client)).decide(claim())


# -- Ollama and Hugging Face (local stub HTTP server) ------------------------

class Stub(BaseHTTPRequestHandler):
    last = {}

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.last = {"path": self.path, "body": body, "auth": self.headers.get("Authorization")}
        if self.path == "/api/chat":
            out = {"model": body["model"], "message": {"content": json.dumps(ANSWER)}, "prompt_eval_count": 800, "eval_count": 90}
        else:
            out = {"model": body["model"], "choices": [{"message": {"content": "```json\n" + json.dumps(ANSWER) + "\n```"}}],
                   "usage": {"prompt_tokens": 700, "completion_tokens": 80}}
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def stub_server():
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def test_ollama_agent(domain, stub_server):
    r = LLMAgent(domain, OllamaBackend(model="qwen2.5:7b", url=stub_server)).decide(claim())
    assert r.decision.values == {"action": "approve", "payout": 3500.0}
    assert r.cost_usd == 0 and r.input_tokens == 800
    body = Stub.last["body"]
    assert Stub.last["path"] == "/api/chat" and body["format"]["type"] == "object" and body["options"]["temperature"] == 0


def test_huggingface_agent(domain, stub_server):
    hf = HuggingFaceBackend(model="Qwen/Qwen2.5-72B-Instruct", token="hf_test", url=stub_server + "/v1/chat/completions")
    r = LLMAgent(domain, hf).decide(claim())
    assert r.decision.values["action"] == "approve" and r.output_tokens == 80
    assert Stub.last["auth"] == "Bearer hf_test"
    assert Stub.last["body"]["response_format"]["type"] == "json_schema"


def test_huggingface_requires_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(LLMError, match="HF_TOKEN"):
        HuggingFaceBackend()


def test_unreachable_ollama_is_agent_error(domain):
    with pytest.raises(AgentError, match="cannot reach"):
        LLMAgent(domain, OllamaBackend(url="http://127.0.0.1:9", timeout=2)).decide(claim())


# -- blind LLM adjudicator ----------------------------------------------------

class ScriptedBackend:
    name = "scripted"

    def __init__(self, verdict):
        self.verdict = verdict
        self.user = None

    def complete(self, system, user, schema):
        self.user = user
        return LLMResult({"verdict": self.verdict, "reasoning": "per policy", "correct_action": "flag_fraud", "correct_payout": 0}, "m")


@pytest.mark.parametrize("case_id", ["CLM-1", "CLM-2", "CLM-3", "CLM-4"])
def test_adjudicator_maps_blind_labels_back(domain, case_id):
    agent, human = Decision(values={"action": "approve", "payout": 1}), Decision(values={"action": "deny", "payout": 0})
    b = ScriptedBackend("a_wrong")
    adj = LLMAdjudicator(domain, b).suggest(claim(claim_id=case_id), agent, human)
    agent_was_a = b.user.index('"approve"') < b.user.index('"deny"')
    assert adj.suggested_verdict == (Verdict.AGENT_WRONG if agent_was_a else Verdict.HUMAN_WRONG)
    assert "agent" not in b.user.lower()  # blind: never told which is the agent


def test_adjudicator_both_wrong_carries_correct_decision(domain):
    adj = LLMAdjudicator(domain, ScriptedBackend("both_wrong")).suggest(
        claim(), Decision(values={"action": "approve", "payout": 1}), Decision(values={"action": "deny", "payout": 0})
    )
    assert adj.suggested_verdict is Verdict.BOTH_WRONG and adj.suggested_correct == {"action": "flag_fraud", "payout": 0}


def test_local_transformers_backend_with_stub_model(domain):
    torch = pytest.importorskip("torch")
    from shadow_eval.llm import TransformersBackend

    answer = json.dumps(ANSWER)

    class Tok:
        eos_token_id = 0

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert "exactly these keys" in messages[0]["content"]
            return "PROMPT"

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1, 2, 3]])}

        def decode(self, ids, skip_special_tokens):
            assert ids.tolist() == [7, 8]
            return "```json\n" + answer + "\n```"

    class Model:
        def generate(self, input_ids, max_new_tokens, do_sample, pad_token_id):
            assert do_sample is False
            return torch.tensor([[1, 2, 3, 7, 8]])

    backend = TransformersBackend(model="Qwen/Qwen2.5-1.5B-Instruct")
    backend._tok, backend._model = Tok(), Model()
    r = LLMAgent(domain, backend).decide(claim())
    assert r.decision.values == {"action": "approve", "payout": 3500.0}
    assert r.input_tokens == 3 and r.output_tokens == 2 and r.cost_usd == 0
