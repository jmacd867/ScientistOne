import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.ideator import generate_ideas
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

CONSERVATIVE = json.dumps({"ideas": [
    {"title": "FFD", "approach": "sort desc then first-fit", "rationale": "classic"}]})
UNCONVENTIONAL = json.dumps({"ideas": [
    {"title": "Sim-anneal", "approach": "local search over packings", "rationale": "novel"}]})
SCORES = json.dumps({"scores": [
    {"index": 0, "novelty": 2, "feasibility": 5},
    {"index": 1, "novelty": 5, "feasibility": 3}]})


def test_generates_and_ranks(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    bid = store.append("brief-claim", "investigator", {"text": "FFD is strong"})
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([CONSERVATIVE, UNCONVENTIONAL, SCORES]))
    ids = generate_ideas(llm, TASK, "brief text", [bid], store)
    assert len(ids) == 2
    ideas = [store.get(i) for i in ids]
    # ranked by novelty+feasibility: sim-anneal (8) > FFD (7)
    assert ideas[0].payload["title"] == "Sim-anneal"
    assert ideas[0].sources == [bid]


def test_scoring_failure_keeps_generation_order(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(
        [CONSERVATIVE, UNCONVENTIONAL, "junk", "junk", "junk"]))
    ids = generate_ideas(llm, TASK, "brief", [], store)
    assert [store.get(i).payload["title"] for i in ids] == ["FFD", "Sim-anneal"]
