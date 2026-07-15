import ast
import re

from ..llm import LLMClient
from ..tasks.base import TaskSpec


def _system(task: TaskSpec) -> str:
    allowed = "only the standard library"
    if task.allowed_imports:
        allowed = "the standard library plus: " + ", ".join(task.allowed_imports)
    return ("You are an expert algorithm implementer. Reply with a single "
            "complete Python solution in one ```python code fence. The code "
            "must define exactly the interface in the starter template, use "
            f"{allowed}, and be deterministic.")


def _extract_code(reply: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    candidate = match.group(1) if match else reply
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def solve(llm: LLMClient, task: TaskSpec, idea: dict, feedback: str | None) -> str | None:
    starter = task.starter_path().read_text()
    prompt = (
        f"Task: {task.description}\n\nStarter template:\n```python\n{starter}```\n\n"
        f"Approach to implement: {idea['title']} — {idea['approach']}\n"
        f"Rationale: {idea['rationale']}\n"
    )
    if feedback:
        prompt += f"\nFeedback from previous attempt:\n{feedback}\n\nImprove on it."
    return _extract_code(llm.chat("reasoning", _system(task), prompt))
