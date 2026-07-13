import json
import math
from pathlib import Path

import httpx
from pydantic import BaseModel

from ..config import Config
from ..discovery.auditor import audit_solution
from ..investigator.scholarly import search_papers
from ..llm import LLMClient
from ..sandbox import run_evaluation
from ..tasks.base import load_task


class CheckResult(BaseModel):
    name: str
    passed: bool | None
    detail: str


class AuditReport(BaseModel):
    checks: list[CheckResult]


class ReportedScore(BaseModel):
    score: float | None


class Aligned(BaseModel):
    aligned: bool


class RefMatch(BaseModel):
    match: bool


def _i1_score(llm, config, task, paper: str, solution_path: str,
              run_dir: Path) -> CheckResult:
    extracted = llm.chat_json(
        "judging", "Extract the main reported score from the paper. Return JSON.",
        f"Paper:\n{paper}\n\n"
        'Return {"score": float or null if no score is reported}.',
        ReportedScore)
    if extracted is None or extracted.score is None:
        return CheckResult(name="score-verification", passed=None,
                           detail="no score extracted from paper")
    outcome = run_evaluation(task, Path(solution_path), run_dir / "audit_work",
                             task.timeout_s or config.solver.timeout_s)
    if not outcome.ok:
        return CheckResult(name="score-verification", passed=False,
                           detail=f"solution failed re-run: {outcome.log[:200]}")
    ok = math.isclose(extracted.score, outcome.score,
                      rel_tol=config.verifier.numeric_tolerance, abs_tol=1e-9)
    return CheckResult(
        name="score-verification", passed=ok,
        detail=f"paper={extracted.score} rerun={outcome.score}")


def _i2_spec(llm, task, code: str) -> CheckResult:
    verdicts = [audit_solution(llm, task, code).flagged for _ in range(3)]
    flagged = sum(verdicts) >= 2
    return CheckResult(name="spec-violation", passed=not flagged,
                       detail=f"votes flagged={sum(verdicts)}/3")


def _i3_references(llm, references: list[dict],
                   client: httpx.Client) -> CheckResult:
    unresolved = []
    for ref in references:
        found = search_papers(client, [ref["title"]], limit_per_query=5)
        titles = {p.title.strip().casefold() for p in found}
        if ref["title"].strip().casefold() in titles:
            continue
        verdict = llm.chat_json(
            "judging", "Decide whether any candidate is the same paper. Return JSON.",
            f"Reference: {ref['title']} ({ref.get('year')})\n"
            f"Candidates: {[p.title for p in found]}\n"
            'Return {"match": bool}.',
            RefMatch)
        if verdict is None or not verdict.match:
            unresolved.append(ref["title"])
    return CheckResult(name="reference-verification", passed=not unresolved,
                       detail=f"{len(unresolved)} unresolved of "
                              f"{len(references)}: {unresolved[:3]}")


def _i4_alignment(llm, paper: str, code: str) -> CheckResult:
    votes = []
    for _ in range(3):
        verdict = llm.chat_json(
            "judging",
            "Judge whether the paper's method section faithfully describes "
            "the code. Simplification is aligned; a fundamentally different "
            "algorithm is misaligned. Return JSON.",
            f"Paper:\n{paper}\n\nCode:\n```python\n{code}```\n"
            'Return {"aligned": bool}.',
            Aligned)
        votes.append(verdict.aligned if verdict else False)
    aligned = sum(votes) >= 2
    return CheckResult(name="method-code-alignment", passed=aligned,
                       detail=f"votes aligned={sum(votes)}/3")


def run_audit(llm: LLMClient, config: Config, run_dir: Path,
              http_client: httpx.Client | None = None) -> AuditReport:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    task = load_task(Path(manifest["task_path"]))
    paper = Path(manifest["paper_path"]).read_text()
    code = Path(manifest["best_solution_path"]).read_text()
    references = json.loads(Path(manifest["references_path"]).read_text())
    client = http_client or httpx.Client(timeout=30)

    report = AuditReport(checks=[
        _i1_score(llm, config, task, paper, manifest["best_solution_path"], run_dir),
        _i2_spec(llm, task, code),
        _i3_references(llm, references, client),
        _i4_alignment(llm, paper, code),
    ])
    (run_dir / "audit_report.json").write_text(report.model_dump_json(indent=2))
    return report
