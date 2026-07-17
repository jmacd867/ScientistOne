import json
from pathlib import Path

import httpx

from scientist_one.config import Config
from scientist_one.evidence import EvidenceStore
from scientist_one.investigator.run import run_investigator
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

S2_BODY = {"data": [
    {"title": "FFD Analysis", "abstract": "FFD uses at most 11/9 OPT bins.",
     "year": 1974, "url": "https://s2/ffd", "externalIds": {"DOI": "10.1/ffd"},
     "authors": [{"name": "D. Johnson"}]},
    {"title": "Irrelevant Paper", "abstract": "About databases.", "year": 2000,
     "url": "https://s2/x", "externalIds": {}, "authors": []},
]}

RELEVANCE = json.dumps({"scores": [{"index": 0, "relevance": 5},
                                   {"index": 1, "relevance": 1}]})
BRIEF = json.dumps({
    "framing": "1D bin packing seeks minimal bins.",
    "claims": [{"text": "FFD achieves 11/9 OPT.", "paper_indexes": [0]}],
    "baselines": "First-fit ratio around 1.7.",
})


def mock_client():
    def handler(request):
        if "semanticscholar" in request.url.host:
            return httpx.Response(200, json=S2_BODY)
        return httpx.Response(200, text='<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_investigator_produces_grounded_brief(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    cfg = Config(investigator={"max_papers": 1})
    llm = LLMClient(cfg, tmp_path, backend=FakeBackend([RELEVANCE, BRIEF]))
    result = run_investigator(llm, cfg, TASK, store, tmp_path,
                              http_client=mock_client())
    refs = json.loads(Path(result.references_path).read_text())
    assert [r["title"] for r in refs] == ["FFD Analysis"]  # top-1 by relevance
    assert "FFD achieves 11/9 OPT." in result.brief_text
    claim = store.get(result.brief_ids[0])
    assert claim.type == "brief-claim"
    paper_rec = store.get(claim.sources[0])
    assert paper_rec.payload["title"] == "FFD Analysis"


def test_no_papers_still_briefs(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    cfg = Config()
    empty = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))
    llm = LLMClient(cfg, tmp_path, backend=FakeBackend([json.dumps(
        {"framing": "f", "claims": [{"text": "c", "paper_indexes": []}],
         "baselines": "b"})]))
    result = run_investigator(llm, cfg, TASK, store, tmp_path, http_client=empty)
    assert json.loads(Path(result.references_path).read_text()) == []
    assert store.get(result.brief_ids[0]).sources == []
