import json

import httpx
import pytest
from pydantic import BaseModel
from scientist_one.config import Config, LLMConfig
from scientist_one.llm import FakeBackend, LLMClient, _ollama_backend


class Verdict(BaseModel):
    flagged: bool
    reason: str


def make_client(tmp_path, responses):
    return LLMClient(Config(), tmp_path, backend=FakeBackend(responses))


def test_chat_returns_text_and_logs(tmp_path):
    client = make_client(tmp_path, ["hello"])
    assert client.chat("reasoning", "sys", "usr") == "hello"
    lines = (tmp_path / "llm_calls.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["model"] == "gemma4:26b"
    assert entry["response"] == "hello"


def test_chat_json_parses_model(tmp_path):
    client = make_client(tmp_path, ['{"flagged": true, "reason": "hardcoded"}'])
    v = client.chat_json("judging", "sys", "usr", Verdict)
    assert v.flagged is True


def test_chat_json_retries_then_none(tmp_path):
    client = make_client(tmp_path, ["not json", "still not", "nope"])
    assert client.chat_json("judging", "sys", "usr", Verdict) is None


def test_chat_json_recovers_on_retry(tmp_path):
    client = make_client(tmp_path, ["bad", '{"flagged": false, "reason": "ok"}'])
    v = client.chat_json("judging", "sys", "usr", Verdict)
    assert v.reason == "ok"


class RaisingBackend:
    """Backend stub that raises like a hung/unreachable ollama server."""

    def __init__(self, exc: Exception):
        self.exc = exc

    def __call__(self, model, system, user, format):
        raise self.exc


def test_backend_timeout_degrades_to_empty_response(tmp_path):
    client = LLMClient(Config(), tmp_path,
                       backend=RaisingBackend(httpx.ReadTimeout("timed out")))
    assert client.chat("reasoning", "sys", "usr") == ""
    entry = json.loads((tmp_path / "llm_calls.jsonl").read_text().splitlines()[0])
    assert entry["response"] == ""
    assert "timed out" in entry["error"]


def test_backend_timeout_error_also_degrades(tmp_path):
    client = LLMClient(Config(), tmp_path,
                       backend=RaisingBackend(TimeoutError("deadline exceeded")))
    assert client.chat_json("judging", "sys", "usr", Verdict) is None


def test_backend_exhaustion_still_raises(tmp_path):
    # FakeBackend's IndexError on exhaustion must NOT be swallowed by the
    # same degrade-on-error path used for real backend timeouts — that
    # would silently mask a test/script that called the client too many times.
    client = make_client(tmp_path, ["only one reply"])
    client.chat("reasoning", "sys", "usr")
    with pytest.raises(IndexError):
        client.chat("reasoning", "sys", "usr")


def test_ollama_backend_passes_timeout_and_sampling_options(monkeypatch):
    calls = {}

    class FakeOllamaClient:
        def __init__(self, host, timeout):
            calls["host"] = host
            calls["timeout"] = timeout

        def chat(self, model, messages, format, options):
            calls["options"] = options
            return {"message": {"content": "ok"}}

    class FakeOllamaModule:
        Client = FakeOllamaClient

    import sys
    monkeypatch.setitem(sys.modules, "ollama", FakeOllamaModule())

    llm_config = LLMConfig(timeout_s=42, max_output_tokens=777, temperature=0.5,
                           repeat_penalty=1.5, frequency_penalty=0.6,
                           presence_penalty=0.6)
    backend = _ollama_backend("http://localhost:11434", llm_config)
    result = backend("gemma4:26b", "sys", "usr", None)

    assert result == "ok"
    assert calls["timeout"] == 42
    assert calls["options"] == {
        "num_predict": 777, "temperature": 0.5, "repeat_penalty": 1.5,
        "frequency_penalty": 0.6, "presence_penalty": 0.6,
    }
