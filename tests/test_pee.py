import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import run_discovery
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

FFD_CODE = ("def pack(items, capacity):\n"
            "    bins = []\n"
            "    for item in sorted(items, reverse=True):\n"
            "        for b in bins:\n"
            "            if sum(b) + item <= capacity:\n"
            "                b.append(item)\n"
            "                break\n"
            "        else:\n"
            "            bins.append([item])\n"
            "    return bins\n")
FF_NOSORT = FFD_CODE.replace("sorted(items, reverse=True)", "items")

IDEAS_A = json.dumps({"ideas": [{"title": "FFD", "approach": "sort desc + first fit",
                                 "rationale": "classic"}]})
IDEAS_B = json.dumps({"ideas": []})
SCORES = json.dumps({"scores": [{"index": 0, "novelty": 3, "feasibility": 5}]})
NOT_FLAGGED = json.dumps({"flagged": False, "reason": "legitimate heuristic"})
COMPONENTS = json.dumps({"components": ["descending sort"]})


def small_config():
    return Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                  solver={"timeout_s": 30})


def test_full_discovery_run(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [
        IDEAS_A, IDEAS_B, SCORES,                      # ideator
        f"```python\n{FFD_CODE}```",                   # solve b0 i0
        NOT_FLAGGED,                                   # audit b0 i0
        COMPONENTS,                                    # ablation component list
        f"```python\n{FF_NOSORT}```",                  # ablation variant code
    ]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    result = run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", [])
    assert result is not None
    assert Path(result.best_solution_path).read_text() == FFD_CODE
    assert store.get(result.best_eval_id).payload["ok"] is True
    assert len(result.ablation_ids) == 1
    abl = store.get(result.ablation_ids[0])
    assert abl.payload["component"] == "descending sort"
    assert abl.sources == [result.best_solution_id]


def test_no_valid_solution_returns_none(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [IDEAS_A, IDEAS_B, SCORES, "no code here ((("]  # solver garbage
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    assert run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", []) is None


def test_flagged_solution_excluded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [IDEAS_A, IDEAS_B, SCORES,
                 f"```python\n{FFD_CODE}```",
                 json.dumps({"flagged": True, "reason": "gaming"})]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    assert run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", []) is None
    assert len(store.by_type("audit-flag")) == 1
