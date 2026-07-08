from collections import deque
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..sandbox import run_evaluation
from ..tasks.base import TaskSpec
from .auditor import audit_solution
from .ideator import generate_ideas
from .solver import _extract_code, solve


class _Branch(BaseModel):
    idea_id: str
    feedback: str = ""
    best_score: float | None = None
    best_eval_id: str | None = None
    best_solution_id: str | None = None
    best_solution_path: str | None = None


class DiscoveryResult(BaseModel):
    best_solution_path: str
    best_solution_id: str
    best_eval_id: str
    best_score: float
    ablation_ids: list[str]


class ComponentList(BaseModel):
    components: list[str]


def run_discovery(llm: LLMClient, config: Config, task: TaskSpec,
                  store: EvidenceStore, run_dir: Path, brief_text: str,
                  brief_ids: list[str]) -> DiscoveryResult | None:
    run_dir = Path(run_dir)
    (run_dir / "solutions").mkdir(parents=True, exist_ok=True)
    timeout = task.timeout_s or config.solver.timeout_s

    idea_queue = deque(generate_ideas(llm, task, brief_text, brief_ids, store))
    if not idea_queue:
        return None
    n_branches = min(config.discovery.branches, len(idea_queue))
    branches = [_Branch(idea_id=idea_queue.popleft()) for _ in range(n_branches)]

    for it in range(config.discovery.iterations):
        for bi, br in enumerate(branches):
            idea = store.get(br.idea_id).payload
            code = solve(llm, task, idea, br.feedback or None)
            if code is None:
                br.feedback += "\nprevious reply contained no valid Python code"
                continue
            path = run_dir / "solutions" / f"b{bi}_i{it}.py"
            path.write_text(code)
            sol_id = store.append("solution", "discovery",
                                  {"path": str(path), "iteration": it},
                                  sources=[br.idea_id])
            outcome = run_evaluation(task, path, run_dir / "eval_work", timeout)
            eval_id = store.append("eval-result", "discovery",
                                   {"ok": outcome.ok, "score": outcome.score,
                                    "log": outcome.log}, sources=[sol_id])
            verdict = audit_solution(llm, task, code)
            if verdict.flagged:
                store.append("audit-flag", "discovery",
                             {"reason": verdict.reason}, sources=[sol_id])
            br.feedback = f"score={outcome.score} ok={outcome.ok}\n{outcome.log[-1000:]}"
            if outcome.ok and not verdict.flagged and (
                    br.best_score is None or task.better(outcome.score, br.best_score)):
                br.best_score, br.best_eval_id = outcome.score, eval_id
                br.best_solution_id, br.best_solution_path = sol_id, str(path)
        if it < config.discovery.iterations - 1 and idea_queue:
            scored = [b for b in branches if b.best_score is not None]
            scored.sort(key=lambda b: b.best_score,
                        reverse=(task.metric_direction == "higher"))
            survivors = scored[:config.discovery.survivors]
            refill = [b for b in branches if b not in survivors]
            for i, b in enumerate(refill):
                if idea_queue:
                    refill[i] = _Branch(idea_id=idea_queue.popleft())
            branches = survivors + refill

    best: _Branch | None = None
    for br in branches:
        if br.best_score is not None and (
                best is None or task.better(br.best_score, best.best_score)):
            best = br
    if best is None:
        return None

    ablation_ids = _run_ablations(llm, config, task, store, run_dir, best, timeout)
    return DiscoveryResult(
        best_solution_path=best.best_solution_path,
        best_solution_id=best.best_solution_id,
        best_eval_id=best.best_eval_id,
        best_score=best.best_score,
        ablation_ids=ablation_ids,
    )


def _run_ablations(llm: LLMClient, config: Config, task: TaskSpec,
                   store: EvidenceStore, run_dir: Path, best: _Branch,
                   timeout: int) -> list[str]:
    code = Path(best.best_solution_path).read_text()
    comp = llm.chat_json(
        "judging", "You analyze algorithms. Return JSON.",
        f"List up to 3 named components of this solution that could be "
        f"individually disabled to measure their contribution.\n"
        f"```python\n{code}```\n"
        'Return {"components": [str]}.',
        ComponentList,
    )
    ids: list[str] = []
    for i, component in enumerate((comp.components if comp else [])[:3]):
        reply = llm.chat(
            "reasoning",
            "You are an expert algorithm implementer. Reply with a single "
            "complete Python solution in one ```python code fence.",
            f"Rewrite this solution with the component '{component}' disabled "
            f"or replaced by the trivial alternative, changing nothing else:\n"
            f"```python\n{code}```",
        )
        variant = _extract_code(reply)
        if variant is None:
            continue
        vpath = run_dir / "solutions" / f"ablation_{i}.py"
        vpath.write_text(variant)
        outcome = run_evaluation(task, vpath, run_dir / "eval_work", timeout)
        ids.append(store.append("ablation", "discovery", {
            "component": component, "ok": outcome.ok, "score": outcome.score,
            "baseline_score": best.best_score,
        }, sources=[best.best_solution_id]))
    return ids
