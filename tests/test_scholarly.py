import httpx

from scientist_one.investigator.scholarly import search_papers

S2_BODY = {"data": [{
    "title": "First Fit Decreasing", "abstract": "We analyze FFD.",
    "year": 1974, "url": "https://s2/ffd", "externalIds": {"DOI": "10.1/ffd"},
    "authors": [{"name": "D. Johnson"}],
}]}

ARXIV_BODY = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>Online Bin Packing Revisited</title>
    <summary>A new online heuristic.</summary>
    <published>2020-01-01T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
</feed>"""


def mock_client(s2_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if "semanticscholar" in request.url.host:
            return httpx.Response(s2_status, json=S2_BODY)
        return httpx.Response(200, text=ARXIV_BODY)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_papers_merges_and_dedupes():
    papers = search_papers(mock_client(), ["bin packing"], limit_per_query=5)
    titles = {p.title for p in papers}
    assert "First Fit Decreasing" in titles
    assert "Online Bin Packing Revisited" in titles
    sources = {p.source for p in papers}
    assert sources == {"semantic_scholar", "arxiv"}


def test_api_failure_degrades():
    papers = search_papers(mock_client(s2_status=500), ["bin packing"], 5)
    assert [p.source for p in papers] == ["arxiv"]  # S2 failed, arXiv survived
