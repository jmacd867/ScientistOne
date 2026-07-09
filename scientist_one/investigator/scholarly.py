import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


class PaperMeta(BaseModel):
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    source: str
    external_id: str


def search_semantic_scholar(client: httpx.Client, query: str,
                            limit: int) -> list[PaperMeta]:
    resp = client.get(S2_URL, params={
        "query": query, "limit": limit,
        "fields": "title,abstract,authors,year,url,externalIds"})
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("data", []):
        ext = item.get("externalIds") or {}
        papers.append(PaperMeta(
            title=item.get("title") or "",
            authors=[a["name"] for a in item.get("authors") or []],
            year=item.get("year"),
            abstract=item.get("abstract") or "",
            url=item.get("url") or "",
            source="semantic_scholar",
            external_id=ext.get("DOI") or ext.get("ArXiv") or "",
        ))
    return papers


def search_arxiv(client: httpx.Client, query: str, limit: int) -> list[PaperMeta]:
    resp = client.get(ARXIV_URL, params={
        "search_query": f"all:{query}", "max_results": limit})
    resp.raise_for_status()
    papers = []
    for entry in ET.fromstring(resp.text).findall(f"{_ATOM}entry"):
        published = entry.findtext(f"{_ATOM}published") or ""
        papers.append(PaperMeta(
            title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()),
            authors=[(a.findtext(f"{_ATOM}name") or "")
                     for a in entry.findall(f"{_ATOM}author")],
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
            url=entry.findtext(f"{_ATOM}id") or "",
            source="arxiv",
            external_id=(entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1],
        ))
    return papers


def search_papers(client: httpx.Client, queries: list[str],
                  limit_per_query: int) -> list[PaperMeta]:
    papers: list[PaperMeta] = []
    for query in queries:
        for fn in (search_semantic_scholar, search_arxiv):
            try:
                papers.extend(fn(client, query, limit_per_query))
            except (httpx.HTTPError, ET.ParseError):
                continue
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p.title.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
