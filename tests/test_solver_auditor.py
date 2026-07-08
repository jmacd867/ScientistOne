import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.auditor import audit_solution
from scientist_one.discovery.solver import solve
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))
IDEA = {"title": "FFD", "approach": "sort desc", "rationale": "classic"}
CODE = "def pack(items, capacity):\n    return [[i] for i in items]\n"


def make_llm(tmp_path, responses):
    return LLMClient(Config(), tmp_path, backend=FakeBackend(responses))


def test_solve_extracts_fenced_code(tmp_path):
    llm = make_llm(tmp_path, [f"Here you go:\n```python\n{CODE}```\ndone"])
    assert solve(llm, TASK, IDEA, None) == CODE


def test_solve_returns_none_on_garbage(tmp_path):
    llm = make_llm(tmp_path, ["I cannot write code today ((("])
    assert solve(llm, TASK, IDEA, feedback="prior score 1.4") is None


def test_audit_flagged(tmp_path):
    llm = make_llm(tmp_path, [json.dumps({"flagged": True, "reason": "hardcodes answers"})])
    v = audit_solution(llm, TASK, CODE)
    assert v.flagged and "hardcodes" in v.reason


def test_audit_degrades_unflagged(tmp_path):
    llm = make_llm(tmp_path, ["junk", "junk", "junk"])
    v = audit_solution(llm, TASK, CODE)
    assert v.flagged is False
    assert v.reason == "audit unavailable"
