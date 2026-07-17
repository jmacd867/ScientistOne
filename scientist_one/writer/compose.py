from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .render import MarkdownRenderer, Renderer


def compose(llm: LLMClient, task: TaskSpec, narrative: str,
            references: list[dict], renderer: Renderer | None = None) -> str:
    renderer = renderer or MarkdownRenderer()
    body = llm.chat(
        "reasoning",
        "You compose research papers in markdown with sections Introduction, "
        "Method, Results, Conclusion. HARD RULES: keep every {ev:...} "
        "evidence tag attached to its sentence; never add facts, numbers, or "
        "claims that are not in the narrative; write prose around the "
        "established facts only.",
        f"Task: {task.description}\n\nGrounded narrative:\n{narrative}\n\n"
        "Compose the paper body (no title line, no references section).",
    )
    if not body.strip():
        body = narrative
    title = f"Automated Discovery for {task.name.replace('_', ' ').title()}"
    return renderer.render(title, body, references)
