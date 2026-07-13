from pathlib import Path

from scientist_one.sandbox import run_evaluation
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))


def test_good_solution(tmp_path):
    out = run_evaluation(TASK, TASK.starter_path(), tmp_path, timeout_s=30)
    assert out.ok is True
    assert out.score >= 1.0


def test_crashing_solution(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def pack(items, capacity):\n    raise RuntimeError('boom')\n")
    out = run_evaluation(TASK, bad, tmp_path, timeout_s=30)
    assert out.ok is False
    assert "boom" in out.log


def test_hanging_solution_times_out(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text(
        "import time\ndef pack(items, capacity):\n    time.sleep(60)\n    return []\n")
    out = run_evaluation(TASK, slow, tmp_path, timeout_s=2)
    assert out.ok is False
    assert "timeout" in out.log.lower()


def _make_task(tmp_path, evaluator_code: str, metric_direction: str = "higher"):
    task_dir = tmp_path / "fake_task"
    task_dir.mkdir()
    (task_dir / "task.yaml").write_text(
        f"name: fake_task\ndescription: test\nmetric_direction: {metric_direction}\n")
    (task_dir / "starter.py").write_text("def pack(items, capacity): return []\n")
    (task_dir / "evaluator.py").write_text(evaluator_code)
    return load_task(task_dir)


def test_solution_stdout_cannot_spoof_the_result_line(tmp_path):
    """Solution code that prints a fake old-style 'EVAL_RESULT:' line (the
    pre-fix fixed marker) must not be able to override the real score,
    since the marker is now randomized per run and unknown to the solution."""
    task = _make_task(tmp_path,
        "def evaluate(solution_path, workdir):\n"
        "    import importlib.util\n"
        "    spec = importlib.util.spec_from_file_location('sol', solution_path)\n"
        "    mod = importlib.util.module_from_spec(spec)\n"
        "    spec.loader.exec_module(mod)\n"
        "    mod.pack([], 1.0)\n"
        "    return {'score': 42.0, 'log': 'real result'}\n")
    solution = tmp_path / "spoofer.py"
    solution.write_text(
        "def pack(items, capacity):\n"
        "    print('EVAL_RESULT:{\"score\": 999.0, \"log\": \"fake\"}')\n"
        "    return []\n"
    )
    out = run_evaluation(task, solution, tmp_path / "work", timeout_s=30)
    assert out.ok is True
    assert out.score == 42.0
    assert out.log == "real result"


def test_malformed_evaluator_output_degrades_instead_of_crashing(tmp_path):
    """An evaluator (or a solution corrupting its return value) that yields
    a non-numeric score must produce ok=False, not raise out of the sandbox."""
    task = _make_task(tmp_path,
        "def evaluate(solution_path, workdir):\n"
        "    return {'score': 'not-a-number', 'log': 'oops'}\n")
    solution = tmp_path / "sol.py"
    solution.write_text("def pack(items, capacity): return []\n")
    out = run_evaluation(task, solution, tmp_path / "work", timeout_s=30)
    assert out.ok is False
    assert "malformed" in out.log.lower()
