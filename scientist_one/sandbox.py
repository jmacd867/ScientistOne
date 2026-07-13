import json
import secrets
import subprocess
import sys
import textwrap
from pathlib import Path

from pydantic import BaseModel

from .tasks.base import TaskSpec

_RUNNER = textwrap.dedent("""
    import importlib.util, json, sys
    marker = sys.argv.pop(4)  # remove before evaluator/solution code can run
    spec = importlib.util.spec_from_file_location("evaluator", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.evaluate(sys.argv[2], sys.argv[3])
    print(marker + json.dumps(result))
""")


class EvalOutcome(BaseModel):
    ok: bool
    score: float
    log: str


def run_evaluation(task: TaskSpec, solution_path: Path, workdir: Path,
                   timeout_s: int) -> EvalOutcome:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    # A per-run random marker (not a fixed string like "EVAL_RESULT:") so
    # solution code can't spoof the result line by guessing what to print.
    marker = f"EVAL_RESULT_{secrets.token_hex(16)}:"
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(task.evaluator_path().resolve()),
             str(Path(solution_path).resolve()), str(workdir.resolve()), marker],
            capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return EvalOutcome(ok=False, score=0.0, log=f"timeout after {timeout_s}s")
    # Take the LAST matching line: the evaluator's own print is the final
    # thing the runner script does, so this is robust even if earlier
    # solution-code output happened to collide with the marker.
    result_line = None
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            result_line = line[len(marker):]
    if result_line is not None:
        try:
            result = json.loads(result_line)
            return EvalOutcome(ok=True, score=float(result["score"]),
                               log=str(result["log"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return EvalOutcome(ok=False, score=0.0,
                               log=f"malformed evaluator output: {result_line[:500]}")
    return EvalOutcome(ok=False, score=0.0,
                       log=(proc.stderr or proc.stdout or "no output").strip())
