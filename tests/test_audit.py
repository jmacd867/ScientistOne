import json
from pathlib import Path

import httpx

from scientist_one.audit.run import run_audit
from scientist_one.config import Config
from scientist_one.llm import FakeBackend, LLMClient

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

NOT_FLAGGED = json.dumps({"flagged": False, "reason": "ok"})
ALIGNED = json.dumps({"aligned": True})


def make_run_dir(tmp_path, paper_score: float):
    sol = tmp_path / "solution.py"
    sol.write_text(FFD_CODE)
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps([{
        "title": "FFD Analysis", "authors": ["D. Johnson"], "year": 1974,
        "abstract": "", "url": "u", "source": "semantic_scholar",
        "external_id": "x"}]))
    (tmp_path / "paper.md").write_text(
        f"# T\nOur method reaches a mean ratio of {paper_score}.\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "task_path": "tasks/bin_packing", "status": "complete",
        "best_solution_path": str(sol), "best_eval_id": "ev_0004",
        "paper_path": str(tmp_path / "paper.md"), "promoted": True,
        "references_path": str(refs)}))
    return tmp_path


def ref_found_client():
    body = {"data": [{"title": "FFD Analysis", "abstract": "", "year": 1974,
                      "url": "u", "externalIds": {}, "authors": []}]}
    def handler(request):
        if "semanticscholar" in request.url.host:
            return httpx.Response(200, json=body)
        return httpx.Response(200, text='<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    return httpx.Client(transport=httpx.MockTransport(handler))


def get_true_score():
    from scientist_one.sandbox import run_evaluation
    from scientist_one.tasks.base import load_task
    import tempfile
    task = load_task(Path("tasks/bin_packing"))
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / "s.py"
        sol.write_text(FFD_CODE)
        return run_evaluation(task, sol, Path(d), 60).score


def test_audit_all_pass(tmp_path):
    score = get_true_score()
    run_dir = make_run_dir(tmp_path, score)
    responses = [json.dumps({"score": score}),        # I1 extraction
                 NOT_FLAGGED, NOT_FLAGGED, NOT_FLAGGED,  # I2 x3
                 ALIGNED, ALIGNED, ALIGNED]              # I4 x3
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    report = run_audit(llm, Config(), run_dir, http_client=ref_found_client())
    assert all(c.passed for c in report.checks)
    assert (run_dir / "audit_report.json").exists()


def test_audit_catches_score_mismatch(tmp_path):
    run_dir = make_run_dir(tmp_path, 0.5)  # paper claims impossible score
    responses = [json.dumps({"score": 0.5}),
                 NOT_FLAGGED, NOT_FLAGGED, NOT_FLAGGED,
                 ALIGNED, ALIGNED, ALIGNED]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    report = run_audit(llm, Config(), run_dir, http_client=ref_found_client())
    i1 = next(c for c in report.checks if c.name == "score-verification")
    assert i1.passed is False
