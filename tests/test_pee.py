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


def test_pruning_refills_worst_branch_with_next_idea(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    one_per_bin_code = "def pack(items, capacity):\n    return [[i] for i in items]\n"
    ideas_conservative = json.dumps({"ideas": [
        {"title": "FFD", "approach": "sort desc + first fit", "rationale": "classic"},
        {"title": "OneItemPerBin", "approach": "one bin per item", "rationale": "trivial"},
    ]})
    ideas_unconventional = json.dumps({"ideas": [
        {"title": "ThirdIdea", "approach": "random shuffle first fit",
         "rationale": "exploratory"},
    ]})
    scores = json.dumps({"scores": [
        {"index": 0, "novelty": 4, "feasibility": 5},   # FFD: 9
        {"index": 1, "novelty": 4, "feasibility": 4},   # OneItemPerBin: 8
        {"index": 2, "novelty": 3, "feasibility": 4},   # ThirdIdea: 7
    ]})
    responses = [
        ideas_conservative, ideas_unconventional, scores,     # ideator
        f"```python\n{FFD_CODE}```", NOT_FLAGGED,              # i0 b0 (FFD): solve + audit
        f"```python\n{one_per_bin_code}```", NOT_FLAGGED,      # i0 b1 (OneItemPerBin)
        f"```python\n{FFD_CODE}```", NOT_FLAGGED,              # i1 surviving branch (FFD)
        f"```python\n{FFD_CODE}```", NOT_FLAGGED,              # i1 refilled branch (ThirdIdea)
        json.dumps({"components": []}),                       # ablation components
    ]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    config = Config(discovery={"branches": 2, "iterations": 2, "survivors": 1},
                     solver={"timeout_s": 60})

    result = run_discovery(llm, config, TASK, store, tmp_path, "brief", [])

    assert result is not None

    solutions = store.by_type("solution")
    ideas = store.by_type("idea")
    third_idea_id = next(r.id for r in ideas if r.payload["title"] == "ThirdIdea")
    one_per_bin_id = next(r.id for r in ideas if r.payload["title"] == "OneItemPerBin")

    iter1_from_third = [s for s in solutions
                        if s.payload["iteration"] == 1 and s.sources == [third_idea_id]]
    iter1_from_pruned = [s for s in solutions
                         if s.payload["iteration"] == 1 and s.sources == [one_per_bin_id]]

    assert len(iter1_from_third) >= 1
    assert len(iter1_from_pruned) == 0

    one_per_bin_solution_id = next(
        s.id for s in solutions if s.sources == [one_per_bin_id])
    one_per_bin_score = next(
        e.payload["score"] for e in store.by_type("eval-result")
        if e.sources == [one_per_bin_solution_id])
    assert result.best_score < one_per_bin_score
