from pydantic import BaseModel

from ..config import Config
from ..discovery.pee import DiscoveryResult
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .conceive import conceive
from .critic import critic_check
from .ground import ground_check
from .resolve import resolve


class WriterResult(BaseModel):
    narrative: str
    remaining_issues: list[str]


def run_writer(llm: LLMClient, config: Config, task: TaskSpec,
               store: EvidenceStore, brief_text: str,
               discovery: DiscoveryResult) -> WriterResult:
    narrative = conceive(llm, task, store, brief_text, discovery)
    issues: list[str] = []
    for round_no in range(config.writer.max_rounds):
        ground = [f"[{i.kind}] {i.detail}"
                  for i in ground_check(narrative, store,
                                        config.verifier.numeric_tolerance)]
        critic = critic_check(llm, narrative)
        issues = ground + critic
        if not issues:
            break
        narrative = resolve(llm, narrative, issues)
    else:
        issues = [f"[{i.kind}] {i.detail}"
                  for i in ground_check(narrative, store,
                                        config.verifier.numeric_tolerance)]
    return WriterResult(narrative=narrative, remaining_issues=issues)
