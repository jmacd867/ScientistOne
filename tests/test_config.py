# tests/test_config.py
from pathlib import Path
from scientist_one.config import Config, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.models.reasoning == "gemma4:26b"
    assert cfg.models.judging == "gemma4:12b"
    assert cfg.discovery.branches == 3
    assert cfg.solver.timeout_s == 60
    assert cfg.llm.timeout_s == 300
    assert cfg.llm.max_output_tokens == 4096
    assert cfg.llm.repeat_penalty == 1.3
    assert cfg.llm.frequency_penalty == 0.4
    assert cfg.llm.presence_penalty == 0.4
    assert cfg.llm.think is False
    assert cfg.verifier.max_refine_rounds == 3


def test_loads_overrides(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("discovery: {branches: 1, iterations: 2, survivors: 1}\n")
    cfg = load_config(p)
    assert cfg.discovery.branches == 1
    assert cfg.models.reasoning == "gemma4:26b"  # untouched defaults survive
