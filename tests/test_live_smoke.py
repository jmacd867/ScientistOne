"""Live smoke test: needs Ollama running with the configured models.

Run explicitly with:  pytest -m live tests/test_live_smoke.py -v
Excluded from default runs by pyproject's `-m 'not live'` addopts.
"""
from pathlib import Path

import pytest

from scientist_one.config import Config
from scientist_one.pipeline import run_pipeline


@pytest.mark.live
def test_tiny_live_run(tmp_path):
    config = Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                    investigator={"max_papers": 3},
                    writer={"max_rounds": 2})
    manifest = run_pipeline(config, Path("tasks/bin_packing"), tmp_path)
    # A live run may legitimately end not-promoted with a weak local model;
    # what must hold: it terminates, evaluates real code, and leaves artifacts.
    assert manifest["status"] in ("complete", "not-promoted", "discovery-failed")
    assert (tmp_path / "evidence.jsonl").exists()
    if manifest["status"] != "discovery-failed":
        assert Path(manifest["best_solution_path"]).exists()
        assert Path(manifest["paper_path"]).exists()


@pytest.mark.live
@pytest.mark.gpu
def test_tiny_live_run_memory_bandwidth(tmp_path):
    config = Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                    investigator={"max_papers": 3},
                    writer={"max_rounds": 2})
    manifest = run_pipeline(config, Path("tasks/memory_bandwidth"), tmp_path)
    assert manifest["status"] in ("complete", "not-promoted", "discovery-failed")
    assert (tmp_path / "evidence.jsonl").exists()
    if manifest["status"] != "discovery-failed":
        assert Path(manifest["best_solution_path"]).exists()
        assert Path(manifest["paper_path"]).exists()
