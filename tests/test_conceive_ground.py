from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import DiscoveryResult
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.conceive import conceive
from scientist_one.writer.ground import ground_check

TASK = load_task(Path("tasks/bin_packing"))


def seeded_store(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    return store, sol, ev


def test_conceive_registers_draft_claims(tmp_path):
    store, sol, ev = seeded_store(tmp_path)
    discovery = DiscoveryResult(best_solution_path="s.py", best_solution_id=sol,
                                best_eval_id=ev, best_score=1.08, ablation_ids=[])
    narrative = f"## Results\nFFD reaches a ratio of 1.08. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([narrative]))
    out = conceive(llm, TASK, store, "brief", discovery)
    assert f"{{ev:{ev}}}" in out
    claims = store.by_type("draft-claim")
    assert len(claims) == 1
    assert claims[0].sources == [ev]


def test_ground_passes_clean_narrative(tmp_path):
    store, _, ev = seeded_store(tmp_path)
    narrative = f"FFD reaches a ratio of 1.08. {{ev:{ev}}}"
    assert ground_check(narrative, store, 0.01) == []


def test_ground_flags_unknown_tag_and_mismatch(tmp_path):
    store, _, ev = seeded_store(tmp_path)
    narrative = (f"Score is 3.99. {{ev:{ev}}}\n"      # number not in evidence
                 "Ghost claim. {ev:ev_9999}\n"        # unknown tag
                 "We used 42 instances here.\n")      # numeric, untagged
    kinds = sorted(i.kind for i in ground_check(narrative, store, 0.01))
    assert kinds == ["number-mismatch", "unknown-tag", "untagged-numeric"]
