import json
from pydantic import BaseModel
from scientist_one.config import Config
from scientist_one.llm import FakeBackend, LLMClient


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
