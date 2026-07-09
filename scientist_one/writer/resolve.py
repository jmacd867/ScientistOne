from ..llm import LLMClient


def resolve(llm: LLMClient, narrative: str, issues: list[str]) -> str:
    issue_list = "\n".join(f"- {i}" for i in issues)
    reply = llm.chat(
        "reasoning",
        "You revise research narratives. Fix every listed issue by "
        "correcting, calibrating, or DELETING unsupported claims. Keep all "
        "valid {ev:...} evidence tags. Reply with the full revised markdown "
        "narrative only.",
        f"Narrative:\n{narrative}\n\nIssues to fix:\n{issue_list}",
    )
    return reply if reply.strip() else narrative
