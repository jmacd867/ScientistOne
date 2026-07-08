import pytest
from scientist_one.tasks.base import TaskSpec, load_task


def make_task_dir(tmp_path, yaml_text):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "task.yaml").write_text(yaml_text)
    (d / "starter.py").write_text("def pack(items, capacity): ...\n")
    (d / "evaluator.py").write_text("def evaluate(solution_path, workdir): ...\n")
    return d


def test_load_task(tmp_path):
    d = make_task_dir(tmp_path,
        "name: demo\ndescription: a demo\nmetric_direction: lower\n"
        "seed_queries: [bin packing]\n")
    task = load_task(d)
    assert task.name == "demo"
    assert task.metric_direction == "lower"
    assert task.starter_path().exists()
    assert task.better(1.0, 2.0) is True   # lower is better
    assert task.better(2.0, 1.0) is False


def test_missing_file_raises(tmp_path):
    d = make_task_dir(tmp_path, "name: x\ndescription: y\nmetric_direction: higher\n")
    (d / "evaluator.py").unlink()
    with pytest.raises(FileNotFoundError):
        load_task(d)
