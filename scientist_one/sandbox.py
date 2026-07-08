import json
import subprocess
import sys
import textwrap
from pathlib import Path

from pydantic import BaseModel

from .tasks.base import TaskSpec

_RUNNER = textwrap.dedent("""
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("evaluator", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.evaluate(sys.argv[2], sys.argv[3])
    print("EVAL_RESULT:" + json.dumps(result))
""")


class EvalOutcome(BaseModel):
    ok: bool
    score: float
    log: str


def run_evaluation(task: TaskSpec, solution_path: Path, workdir: Path,
                   timeout_s: int) -> EvalOutcome:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(task.evaluator_path().resolve()),
             str(Path(solution_path).resolve()), str(workdir.resolve())],
            capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return EvalOutcome(ok=False, score=0.0, log=f"timeout after {timeout_s}s")
    for line in proc.stdout.splitlines():
        if line.startswith("EVAL_RESULT:"):
            result = json.loads(line[len("EVAL_RESULT:"):])
            return EvalOutcome(ok=True, score=result["score"], log=result["log"])
    return EvalOutcome(ok=False, score=0.0,
                       log=(proc.stderr or proc.stdout or "no output").strip())
