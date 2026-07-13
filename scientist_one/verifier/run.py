import json
import math
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..writer.ground import TAG_RE, numbers_in_payload, numbers_in_text, sentences


class Violation(BaseModel):
    claim: str
    reason: str


class VerifierResult(BaseModel):
    promoted: bool
    paper_path: str
    violations: list[Violation]


class Entailment(BaseModel):
    supported: bool


def _verify(llm: LLMClient, config: Config, paper_md: str, store: EvidenceStore,
            solution_code: str) -> list[Violation]:
    known = {r.id for r in store.all()}
    violations: list[Violation] = []
    for sent in sentences(paper_md):
        tags = TAG_RE.findall(sent)
        clean = TAG_RE.sub("", sent).strip()
        nums = numbers_in_text(sent)
        if not tags:
            if nums:
                violations.append(Violation(
                    claim=clean, reason="untagged numeric claim (compose drift)"))
            continue
        for tag in tags:
            if tag not in known:
                violations.append(Violation(claim=clean,
                                            reason=f"unknown evidence {tag}"))
                continue
            rec = store.get(tag)
            if rec.type in ("eval-result", "ablation"):
                available = numbers_in_payload(rec.payload)
                for num in nums:
                    if not any(math.isclose(num, a,
                               rel_tol=config.verifier.numeric_tolerance,
                               abs_tol=1e-9) for a in available):
                        violations.append(Violation(
                            claim=clean,
                            reason=f"number {num} not in evidence {tag}"))
            elif rec.type == "paper":
                verdict = llm.chat_json(
                    "judging",
                    "Judge whether the abstract supports the claim. Return JSON.",
                    f"Abstract: {rec.payload.get('abstract', '')}\n"
                    f"Claim: {clean}\n"
                    'Return {"supported": bool}.',
                    Entailment)
                if verdict is None:
                    violations.append(Violation(claim=clean,
                                                reason=f"unverifiable citation {tag}"))
                elif not verdict.supported:
                    violations.append(Violation(
                        claim=clean, reason=f"citation {tag} does not support claim"))
            elif rec.type == "solution":
                verdict = llm.chat_json(
                    "judging",
                    "Judge whether the code actually implements what the "
                    "claim describes. Simplification is fine; a different "
                    "algorithm is not. Return JSON.",
                    f"Code:\n```python\n{solution_code}```\nClaim: {clean}\n"
                    'Return {"supported": bool}.',
                    Entailment)
                if verdict is None:
                    violations.append(Violation(claim=clean,
                                                reason=f"unverifiable method {tag}"))
                elif not verdict.supported:
                    violations.append(Violation(
                        claim=clean, reason=f"method claim not matched by code"))
    return violations


def run_verifier(llm: LLMClient, config: Config, run_dir: Path, paper_md: str,
                 store: EvidenceStore, references: list[dict],
                 solution_code: str) -> VerifierResult:
    run_dir = Path(run_dir)
    violations = _verify(llm, config, paper_md, store, solution_code)
    if violations:
        listing = "\n".join(f"- {v.claim}: {v.reason}" for v in violations)
        paper_md = llm.chat(
            "reasoning",
            "You repair research papers. Rewrite each flagged sentence to "
            "match its evidence, or DELETE it if it cannot be supported. "
            "Keep all valid {ev:...} tags. Reply with the full markdown.",
            f"Paper:\n{paper_md}\n\nFlagged claims:\n{listing}",
        ) or paper_md
        violations = _verify(llm, config, paper_md, store, solution_code)
    if not violations:
        final = TAG_RE.sub("", paper_md)
        path = run_dir / "paper.md"
        path.write_text(final)
        return VerifierResult(promoted=True, paper_path=str(path), violations=[])
    (run_dir / "violations.json").write_text(
        json.dumps([v.model_dump() for v in violations], indent=2))
    path = run_dir / "paper.draft.md"
    path.write_text(paper_md)
    return VerifierResult(promoted=False, paper_path=str(path),
                          violations=violations)
