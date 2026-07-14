import importlib.util
from pathlib import Path

import pytest

from scientist_one.tasks.base import load_task

pytestmark = pytest.mark.gpu

TASK_DIR = Path("tasks/memory_bandwidth")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluator", TASK_DIR / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_loads():
    task = load_task(TASK_DIR)
    assert task.metric_direction == "higher"
    assert task.allowed_imports == ["torch"]


def test_starter_triad_scores(tmp_path):
    result = load_evaluator().evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    assert result["score"] > 0
    assert "GB/s" in result["log"]


def test_wrong_shape_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return (a + scalar * b)[:-1]\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))


def test_wrong_dtype_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return (a + scalar * b).to(torch.float16)\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))


def test_incorrect_result_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return a - scalar * b\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))
