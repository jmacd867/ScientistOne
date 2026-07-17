from pydantic import BaseModel

from ..llm import LLMClient


class CriticIssues(BaseModel):
    issues: list[str]


def critic_check(llm: LLMClient, narrative: str) -> list[str]:
    result = llm.chat_json(
        "reasoning",
        "You are a rigorous research critic. Find what deterministic checks "
        "cannot: internal contradictions, overclaims, gap between evidence "
        "and conclusion, missing comparisons, unfair baselines. Return JSON.",
        f"Narrative:\n{narrative}\n\n"
        'Return {"issues": [str]} — empty list if the narrative is sound.',
        CriticIssues,
    )
    return result.issues if result else []
