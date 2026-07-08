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
