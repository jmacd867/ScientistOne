import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import DiscoveryResult
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.critic import critic_check
from scientist_one.writer.resolve import resolve
from scientist_one.writer.run import run_writer

TASK = load_task(Path("tasks/bin_packing"))
NO_ISSUES = json.dumps({"issues": []})


def seeded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    sol_file = tmp_path / "s.py"
    sol_file.write_text("def pack(items, capacity): return [[i] for i in items]\n")
    disc = DiscoveryResult(best_solution_path=str(sol_file), best_solution_id=sol,
                           best_eval_id=ev, best_score=1.08, ablation_ids=[])
    return store, ev, disc


def test_critic_parses_and_degrades(tmp_path):
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([json.dumps({"issues": ["overclaim"]})]))
    assert critic_check(llm, "text") == ["overclaim"]
    llm2 = LLMClient(Config(), tmp_path, backend=FakeBackend(["x", "x", "x"]))
    assert critic_check(llm2, "text") == []


def test_resolve_keeps_original_on_empty_reply(tmp_path):
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(["   "]))
    assert resolve(llm, "original", ["issue"]) == "original"


def test_loop_stops_when_clean(tmp_path):
    store, ev, disc = seeded(tmp_path)
    clean = f"The ratio is 1.08. {{ev:{ev}}}"
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([clean, NO_ISSUES]))  # conceive, critic
    result = run_writer(llm, Config(), TASK, store, "brief", disc)
    assert result.narrative == clean
    assert result.remaining_issues == []


def test_loop_resolves_dirty_narrative(tmp_path):
    store, ev, disc = seeded(tmp_path)
    dirty = f"The ratio is 9.99. {{ev:{ev}}}"
    clean = f"The ratio is 1.08. {{ev:{ev}}}"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([
        dirty, NO_ISSUES,   # conceive + critic round 1 (ground finds mismatch)
        clean,              # resolve
        NO_ISSUES,          # critic round 2 (ground now clean)
    ]))
    result = run_writer(llm, Config(), TASK, store, "brief", disc)
    assert result.narrative == clean
    assert result.remaining_issues == []


def test_exhausted_rounds_keeps_critic_issues(tmp_path):
    store, ev, disc = seeded(tmp_path)
    dirty = f"The ratio is 9.99. {{ev:{ev}}}"
    llm = LLMClient(Config(writer={"max_rounds": 1}), tmp_path, backend=FakeBackend([
        dirty,                                    # conceive
        NO_ISSUES,                                # critic round 1 (ground is dirty)
        dirty,                                    # resolve (fails to fix)
        json.dumps({"issues": ["overclaim"]}),    # final critic re-check
    ]))
    result = run_writer(llm, Config(writer={"max_rounds": 1}), TASK, store,
                        "brief", disc)
    assert any("number-mismatch" in i for i in result.remaining_issues)
    assert "overclaim" in result.remaining_issues
