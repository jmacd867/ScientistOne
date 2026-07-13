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


def test_numeric_claim_on_non_numeric_record_flagged(tmp_path):
    store, _, _, _ = seeded(tmp_path)
    idea = store.append("idea", "discovery", {"title": "FFD"})
    bad = f"# T\nWe achieve a 9.99 improvement. {{ev:{idea}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([bad]))  # refiner no-op
    result = run_verifier(llm, Config(), tmp_path, bad, store, [], CODE)
    assert result.promoted is False
    assert any("not supported by evidence" in v.reason for v in result.violations)


def test_references_section_not_scanned_as_prose(tmp_path):
    """Regression test: the renderer's auto-generated References section
    (e.g. "1. D. Johnson (1974). FFD Analysis. url") must not be flagged as
    an untagged numeric claim — it is deterministic, not model-authored."""
    store, _, _, ev = seeded(tmp_path)
    paper = (f"# T\nWe reach ratio 1.08. {{ev:{ev}}}\n"
             "\n## References\n\n"
             "1. D. Johnson (1974). FFD Analysis. https://s2/ffd\n")
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([]))  # no calls needed
    result = run_verifier(llm, Config(), tmp_path, paper, store, [], CODE)
    assert result.promoted is True
    final = Path(result.paper_path).read_text()
    assert "## References" in final
    assert "1974" in final


def test_citation_numeric_claim_pooled_against_abstract(tmp_path):
    """A number in a citation sentence should be corroborated by numbers in
    the cited paper's abstract, matching ground_check's pooling behavior."""
    store, paper_rec, _, _ = seeded(tmp_path)  # abstract: "FFD is 11/9 OPT."
    paper = f"# T\nFFD achieves a ratio of 11. {{ev:{paper_rec}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([SUPPORTED]))
    result = run_verifier(llm, Config(), tmp_path, paper, store, [], CODE)
    assert result.promoted is True
