import json
from pathlib import Path

import httpx

from scientist_one.config import Config
from scientist_one.llm import FakeBackend
from scientist_one.pipeline import run_pipeline

TASK_PATH = Path("tasks/bin_packing")

FFD_CODE = ("def pack(items, capacity):\n"
            "    bins = []\n"
            "    for item in sorted(items, reverse=True):\n"
            "        for b in bins:\n"
            "            if sum(b) + item <= capacity:\n"
            "                b.append(item)\n"
            "                break\n"
            "        else:\n"
            "            bins.append([item])\n"
            "    return bins\n")

BRIEF = json.dumps({"framing": "Pack items into few bins.",
                    "claims": [{"text": "FFD is a strong baseline.",
                                "paper_indexes": []}],
                    "baselines": "first fit"})
IDEAS_A = json.dumps({"ideas": [{"title": "FFD", "approach": "sort desc",
                                 "rationale": "classic"}]})
IDEAS_B = json.dumps({"ideas": []})
SCORES = json.dumps({"scores": [{"index": 0, "novelty": 3, "feasibility": 5}]})
NOT_FLAGGED = json.dumps({"flagged": False, "reason": "ok"})
NO_COMPONENTS = json.dumps({"components": []})
# ev_0001 brief-claim, ev_0002 idea, ev_0003 solution, ev_0004 eval-result
NARRATIVE = "The heuristic packs items efficiently. {ev:ev_0004}"
NO_ISSUES = json.dumps({"issues": []})

RESPONSES = [
    BRIEF,                          # investigator brief
    IDEAS_A, IDEAS_B, SCORES,       # ideator
    f"```python\n{FFD_CODE}```",    # solve
    NOT_FLAGGED,                    # spec audit
    NO_COMPONENTS,                  # ablations
    NARRATIVE,                      # conceive
    NO_ISSUES,                      # critic
    NARRATIVE,                      # compose body
]


def no_network():
    return httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))


def small_config():
    return Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                  solver={"timeout_s": 60})


def test_pipeline_end_to_end(tmp_path):
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend(list(RESPONSES)),
                            http_client=no_network())
    assert manifest["status"] == "complete"
    assert manifest["promoted"] is True
    paper = Path(manifest["paper_path"]).read_text()
    assert "{ev:" not in paper
    assert (tmp_path / "investigator.json").exists()
    assert (tmp_path / "discovery.json").exists()
    assert (tmp_path / "writer.json").exists()


def test_pipeline_resumes_without_llm(tmp_path):
    run_pipeline(small_config(), TASK_PATH, tmp_path,
                 backend=FakeBackend(list(RESPONSES)), http_client=no_network())
    # Empty backend: any LLM call would raise IndexError — resume must not call
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend([]), http_client=no_network())
    assert manifest["status"] == "complete"


def test_pipeline_discovery_failure(tmp_path):
    responses = [BRIEF, IDEAS_A, IDEAS_B, SCORES, "no code ((("]
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend(responses),
                            http_client=no_network())
    assert manifest["status"] == "discovery-failed"
    assert manifest["paper_path"] is None
