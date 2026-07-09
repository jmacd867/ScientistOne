import json
from pathlib import Path

import httpx
from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .scholarly import PaperMeta, search_papers


class InvestigatorResult(BaseModel):
    brief_text: str
    brief_ids: list[str]
    references_path: str


class RelevanceScore(BaseModel):
    index: int
    relevance: int


class RelevanceScores(BaseModel):
    scores: list[RelevanceScore]


class BriefClaim(BaseModel):
    text: str
    paper_indexes: list[int]


class Brief(BaseModel):
    framing: str
    claims: list[BriefClaim]
    baselines: str


def _rank_papers(llm: LLMClient, task: TaskSpec, papers: list[PaperMeta],
                 max_papers: int) -> list[PaperMeta]:
    if len(papers) <= max_papers:
        return papers
    listing = "\n".join(f"{i}: {p.title} — {p.abstract[:200]}"
                        for i, p in enumerate(papers))
    scored = llm.chat_json(
        "judging", "You rate paper relevance 1-5. Return JSON.",
        f"Task: {task.description}\n\nPapers:\n{listing}\n\n"
        'Return {"scores": [{"index", "relevance"}]}.',
        RelevanceScores,
    )
    if scored is None:
        return papers[:max_papers]
    ranks = {s.index: s.relevance for s in scored.scores}
    order = sorted(range(len(papers)), key=lambda i: -ranks.get(i, 0))
    return [papers[i] for i in order[:max_papers]]


def run_investigator(llm: LLMClient, config: Config, task: TaskSpec,
                     store: EvidenceStore, run_dir: Path,
                     http_client: httpx.Client | None = None) -> InvestigatorResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    client = http_client or httpx.Client(timeout=30)
    papers = search_papers(client, task.seed_queries or [task.description],
                           limit_per_query=config.investigator.max_papers)
    papers = _rank_papers(llm, task, papers, config.investigator.max_papers)

    paper_ids = [store.append("paper", "investigator", p.model_dump())
                 for p in papers]
    references_path = run_dir / "references.json"
    references_path.write_text(json.dumps([p.model_dump() for p in papers], indent=2))

    listing = "\n".join(f"{i}: {p.title} ({p.year}) — {p.abstract[:300]}"
                        for i, p in enumerate(papers)) or "(no papers retrieved)"
    brief = llm.chat_json(
        "reasoning",
        "You write research briefs grounded ONLY in the provided papers. "
        "Every claim must cite paper indexes; use an empty list only when no "
        "paper supports it. Return JSON.",
        f"Task: {task.description}\n\nRetrieved papers:\n{listing}\n\n"
        'Return {"framing": str, "claims": [{"text", "paper_indexes"}], '
        '"baselines": str}.',
        Brief,
    ) or Brief(framing=task.description,
               claims=[BriefClaim(text="No literature grounding available.",
                                  paper_indexes=[])],
               baselines="unknown")

    brief_ids = []
    lines = [f"# Research brief: {task.name}", "", brief.framing, "", "## Findings"]
    for claim in brief.claims:
        sources = [paper_ids[i] for i in claim.paper_indexes
                   if 0 <= i < len(paper_ids)]
        cid = store.append("brief-claim", "investigator",
                           {"text": claim.text}, sources=sources)
        brief_ids.append(cid)
        lines.append(f"- {claim.text} {{ev:{cid}}}")
    lines += ["", "## Baselines", brief.baselines]
    brief_text = "\n".join(lines)
    (run_dir / "brief.md").write_text(brief_text)
    return InvestigatorResult(brief_text=brief_text, brief_ids=brief_ids,
                              references_path=str(references_path))
