from pydantic import BaseModel

from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec


class Idea(BaseModel):
    title: str
    approach: str
    rationale: str


class IdeaList(BaseModel):
    ideas: list[Idea]


class IdeaScore(BaseModel):
    index: int
    novelty: int
    feasibility: int


class IdeaScores(BaseModel):
    scores: list[IdeaScore]


_SYSTEM = ("You are a research ideator. Ground every idea in the research brief. "
           "Return JSON matching the requested schema.")


def _generate(llm: LLMClient, task: TaskSpec, brief_text: str, mode: str) -> list[Idea]:
    prompt = (
        f"Task: {task.description}\n\nResearch brief:\n{brief_text}\n\n"
        f"Propose 2 {mode} algorithmic approaches. "
        '{"ideas": [{"title", "approach", "rationale"}]}'
    )
    result = llm.chat_json("reasoning", _SYSTEM, prompt, IdeaList)
    return result.ideas if result else []


def generate_ideas(llm: LLMClient, task: TaskSpec, brief_text: str,
                   brief_ids: list[str], store: EvidenceStore) -> list[str]:
    ideas = (_generate(llm, task, brief_text, "conservative, well-established")
             + _generate(llm, task, brief_text, "unconventional, creative"))
    listing = "\n".join(f"{i}: {x.title} — {x.approach}" for i, x in enumerate(ideas))
    scored = llm.chat_json(
        "judging", _SYSTEM,
        f"Score each idea 1-5 for novelty and feasibility for this task: "
        f"{task.description}\n\n{listing}\n\n"
        '{"scores": [{"index", "novelty", "feasibility"}]}',
        IdeaScores,
    )
    ranks = {s.index: s for s in scored.scores} if scored else {}
    order = sorted(
        range(len(ideas)),
        key=lambda i: -(ranks[i].novelty + ranks[i].feasibility) if i in ranks else 0,
    )
    ids = []
    for i in order:
        idea, s = ideas[i], ranks.get(i)
        ids.append(store.append("idea", "discovery", {
            "title": idea.title, "approach": idea.approach, "rationale": idea.rationale,
            "novelty": s.novelty if s else None,
            "feasibility": s.feasibility if s else None,
        }, sources=brief_ids))
    return ids
