import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.verifier.run import run_verifier

SUPPORTED = json.dumps({"supported": True})
UNSUPPORTED = json.dumps({"supported": False})
CODE = "def pack(items, capacity): return [[i] for i in items]\n"


def seeded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    paper = store.append("paper", "investigator",
                         {"title": "FFD Analysis", "abstract": "FFD is 11/9 OPT."})
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    return store, paper, sol, ev


def test_clean_paper_promoted(tmp_path):
    store, paper_rec, sol, ev = seeded(tmp_path)
    paper = (f"# T\nWe reach ratio 1.08. {{ev:{ev}}}\n"
             f"FFD is known to be near-optimal. {{ev:{paper_rec}}}\n"
             f"Our method sorts items descending. {{ev:{sol}}}\n")
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([SUPPORTED, SUPPORTED]))  # citation, method
    result = run_verifier(llm, Config(), tmp_path, paper, store, [], CODE)
    assert result.promoted is True
    final = Path(result.paper_path).read_text()
    assert "{ev:" not in final                       # tags stripped
    assert result.paper_path.endswith("paper.md")


def test_numeric_violation_refined_then_promoted(tmp_path):
    store, _, _, ev = seeded(tmp_path)
    bad = f"# T\nWe reach ratio 9.99. {{ev:{ev}}}\n"
    fixed = f"# T\nWe reach ratio 1.08. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([fixed]))  # refiner
    result = run_verifier(llm, Config(), tmp_path, bad, store, [], CODE)
    assert result.promoted is True
    assert "1.08" in Path(result.paper_path).read_text()


def test_unfixable_paper_stays_draft(tmp_path):
    store, _, _, ev = seeded(tmp_path)
    bad = f"# T\nWe reach ratio 9.99. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([bad]))  # refiner no-op
    result = run_verifier(llm, Config(), tmp_path, bad, store, [], CODE)
    assert result.promoted is False
    assert result.paper_path.endswith("paper.draft.md")
    assert (Path(tmp_path) / "violations.json").exists()
    assert result.violations[0].reason.startswith("number")
