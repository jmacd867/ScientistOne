from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import DiscoveryResult
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.conceive import conceive
from scientist_one.writer.ground import ground_check, normalize_tags

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


def test_normalize_tags_expands_square_bracket_multi_id():
    text = "An ablation was run [ev:ev_0047, ev:ev_0048, ev:ev_0049] on it."
    assert (normalize_tags(text) ==
            "An ablation was run {ev:ev_0047}{ev:ev_0048}{ev:ev_0049} on it.")


def test_normalize_tags_expands_curly_multi_id():
    text = "The ratio changed {ev:ev_0047, ev:ev_0048, ev:ev_0049} here."
    assert (normalize_tags(text) ==
            "The ratio changed {ev:ev_0047}{ev:ev_0048}{ev:ev_0049} here.")


def test_normalize_tags_converts_single_square_bracket():
    assert normalize_tags("See [ev:ev_0016] for details.") == "See {ev:ev_0016} for details."


def test_normalize_tags_idempotent_on_canonical_form():
    text = "Already fine. {ev:ev_0016}"
    assert normalize_tags(text) == text


def test_normalize_tags_does_not_fix_wrong_id():
    # A dropped digit is a real hallucination the caller must still catch —
    # normalize_tags only fixes bracket/multi-ID formatting, not ID correctness.
    text = "Wrong id. {ev:ev_016}"
    assert normalize_tags(text) == text


def test_ground_check_catches_number_hidden_in_malformed_tag(tmp_path):
    # Before normalization, digits inside a malformed multi-ID tag get
    # misread as an unverified numeric claim. Confirm normalize_tags fixes
    # the exact failure mode observed in production: the tag's own IDs
    # (0047, 0048, 0049) no longer masquerade as claim numbers once expanded.
    store, _, ev = seeded_store(tmp_path)
    raw = f"An ablation was run [ev:{ev}, ev:{ev}] here."
    assert ground_check(normalize_tags(raw), store, 0.01) == []


def test_conceive_excludes_invalid_ablations_from_prompt(tmp_path):
    store, sol, ev = seeded_store(tmp_path)
    valid_ab = store.append("ablation", "discovery",
                            {"component": "Best Fit logic", "ok": True,
                             "score": 1.2, "baseline_score": 1.08, "valid": True,
                             "validity_reason": "verified as a genuine modification"},
                            sources=[sol])
    invalid_ab = store.append("ablation", "discovery",
                              {"component": "Sorting", "ok": True, "score": 1.08,
                               "baseline_score": 1.08, "valid": False,
                               "validity_reason": "ablation code is unchanged "
                                                  "from the baseline solution"},
                              sources=[sol])
    discovery = DiscoveryResult(best_solution_path="s.py", best_solution_id=sol,
                                best_eval_id=ev, best_score=1.08,
                                ablation_ids=[valid_ab, invalid_ab])
    narrative = f"## Results\nRatio 1.08. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([narrative]))
    conceive(llm, TASK, store, "brief", discovery)

    # Inspect what was actually sent to the model via the log file, since
    # FakeBackend doesn't expose the prompt directly.
    import json
    log_line = json.loads((tmp_path / "llm_calls.jsonl").read_text().splitlines()[0])
    prompt_sent = log_line["user"]
    assert "Best Fit logic" in prompt_sent
    assert "1 additional ablation attempt(s)" in prompt_sent
    assert "'Sorting' disabled" not in prompt_sent
