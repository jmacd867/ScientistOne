import importlib.util
import json
from pathlib import Path

from scientist_one.tasks.base import load_task

TASK_DIR = Path("tasks/bin_packing")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluator", TASK_DIR / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_loads():
    task = load_task(TASK_DIR)
    assert task.metric_direction == "lower"


def test_starter_first_fit_scores(tmp_path):
    # starter.py ships a naive first-fit so the pipeline has a working baseline
    result = load_evaluator().evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    assert 1.0 <= result["score"] < 2.0
    assert "instances" in result["log"]


def test_invalid_packing_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def pack(items, capacity):\n    return [items]\n")  # overflows one bin
    import pytest
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))
