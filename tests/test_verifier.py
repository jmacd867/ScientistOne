import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.verifier.run import run_verifier
from scientist_one.writer.compose import compose

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


def test_compose_then_verify_promotes_with_real_references(tmp_path):
    """Integration test: guards the coupling between compose()'s renderer
    output and run_verifier's references-section detection. If either side's
    '## References' heading drifts, this fails instead of the false positive
    silently returning."""
    store, paper_rec, _, ev = seeded(tmp_path)
    task = load_task(Path("tasks/bin_packing"))
    references = [{"title": "FFD Analysis", "authors": ["D. Johnson"], "year": 1974,
                  "url": "https://s2/ffd", "abstract": "", "source": "semantic_scholar",
                  "external_id": "10.1/ffd"}]
    body = f"## Results\nWe reach ratio 1.08. {{ev:{ev}}}\n"
    compose_llm = LLMClient(Config(), tmp_path, backend=FakeBackend([body]))
    paper_md = compose(compose_llm, task, "narrative", references)

    verify_llm = LLMClient(Config(), tmp_path, backend=FakeBackend([]))
    result = run_verifier(verify_llm, Config(), tmp_path, paper_md, store,
                          references, CODE)
    assert result.promoted is True
    final = Path(result.paper_path).read_text()
    assert "FFD Analysis" in final


def test_multi_id_bracket_citation_normalized_and_promoted(tmp_path):
    """Regression test for the production bug: a model citing multiple
    evidence records in one bracket (square OR curly, comma-separated)
    instead of one tag per record. Before normalize_tags, this both failed
    to register as a citation AND caused the IDs' own digits to be
    misread as an unverified numeric claim."""
    store, _, sol, ev = seeded(tmp_path)
    ab1 = store.append("ablation", "discovery",
                       {"component": "Sorting", "score": 1.2,
                        "baseline_score": 1.08, "ok": True, "valid": True},
                       sources=[sol])
    ab2 = store.append("ablation", "discovery",
                       {"component": "Best Fit", "score": 1.3,
                        "baseline_score": 1.08, "ok": True, "valid": True},
                       sources=[sol])
    paper = (f"# T\nAn ablation study was conducted [ev:{ab1}, ev:{ab2}] on "
             "components.\n"
             f"The change resulted in a ratio of 1.2 {{ev:{ab1}, ev:{ab2}}}.\n")
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([]))  # no judge calls needed
    result = run_verifier(llm, Config(), tmp_path, paper, store, [], CODE)
    assert result.promoted is True
