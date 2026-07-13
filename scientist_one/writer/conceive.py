from pathlib import Path

from ..discovery.pee import DiscoveryResult
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .ground import TAG_RE, normalize_tags, sentences

_SYSTEM = (
    "You write research narratives in markdown. HARD RULE: every factual "
    "sentence must end with an evidence tag like {ev:ev_0042} naming one of "
    "the evidence records provided. Never invent numbers or tags; only state "
    "what the evidence supports."
)


def conceive(llm: LLMClient, task: TaskSpec, store: EvidenceStore,
             brief_text: str, discovery: DiscoveryResult) -> str:
    eval_rec = store.get(discovery.best_eval_id)
    ablations = [store.get(a) for a in discovery.ablation_ids]
    # Ablations that never actually changed the algorithm's behavior (see
    # pee.py's _check_ablation_validity) are excluded from the narrative —
    # their "no score change" isn't a real finding, it's a no-op re-run of
    # the same program, and presenting it as evidence would be misleading.
    valid_ablations = [a for a in ablations if a.payload.get("valid", True)]
    excluded = len(ablations) - len(valid_ablations)
    ablation_lines = [
        f"- {a.id}: component '{a.payload['component']}' disabled -> "
        f"score {a.payload['score']} (baseline {a.payload['baseline_score']})"
        for a in valid_ablations
    ]
    if excluded:
        ablation_lines.append(
            f"- Note: {excluded} additional ablation attempt(s) did not "
            "produce a behaviorally different implementation of the "
            "component and are excluded from this analysis.")
    ablation_text = "\n".join(ablation_lines) or "(none)"
    try:
        code = Path(discovery.best_solution_path).read_text()
    except OSError:
        code = "(solution source unavailable)"
    prompt = (
        f"Task: {task.description}\n\n"
        f"Research brief (with its evidence tags):\n{brief_text}\n\n"
        f"Best solution evidence record {discovery.best_eval_id}: "
        f"score={eval_rec.payload['score']}\nEvaluator log:\n"
        f"{eval_rec.payload['log']}\n\nAblations:\n{ablation_text}\n\n"
        f"Solution code:\n```python\n{code}```\n\n"
        "Write the research narrative: problem, method, results, ablation "
        "analysis. Tag every factual sentence."
    )
    narrative = normalize_tags(llm.chat("reasoning", _SYSTEM, prompt))
    known = {r.id for r in store.all()}
    for sent in sentences(narrative):
        tags = [t for t in TAG_RE.findall(sent) if t in known]
        if tags:
            store.append("draft-claim", "writer",
                         {"text": TAG_RE.sub("", sent).strip()}, sources=tags)
    return narrative
