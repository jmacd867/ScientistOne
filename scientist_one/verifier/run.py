import json
import math
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..writer.ground import (TAG_RE, normalize_tags, numbers_in_payload,
                             numbers_in_text, sentences)


class Violation(BaseModel):
    claim: str
    reason: str


class VerifierResult(BaseModel):
    promoted: bool
    paper_path: str
    violations: list[Violation]


class Entailment(BaseModel):
    supported: bool


_REFERENCES_MARKER = "\n## References"


def _split_references(paper_md: str) -> tuple[str, str]:
    """Split off the renderer's auto-generated References section.

    That section is built deterministically from references.json (not the
    model), so it is trustworthy by construction and must not be re-scanned
    as prose — its "(1974)" style years would otherwise read as untagged
    numeric claims.

    Splits at the LAST occurrence of the marker: compose is instructed not
    to emit its own References heading, but its output isn't trusted, so if
    it disobeys and includes one mid-body, that fake section still gets
    scanned as prose — only the renderer's trailing section is exempt.
    """
    idx = paper_md.rfind(_REFERENCES_MARKER)
    if idx == -1:
        return paper_md, ""
    return paper_md[:idx], paper_md[idx:]


def _verify(llm: LLMClient, config: Config, body_md: str, store: EvidenceStore,
            solution_code: str) -> list[Violation]:
    known = {r.id for r in store.all()}
    violations: list[Violation] = []
    for sent in sentences(body_md):
        tags = TAG_RE.findall(sent)
        clean = TAG_RE.sub("", sent).strip()
        nums = numbers_in_text(sent)
        if not tags:
            if nums:
                violations.append(Violation(
                    claim=clean, reason="untagged numeric claim (compose drift)"))
            continue
        known_tags = []
        for tag in tags:
            if tag not in known:
                violations.append(Violation(claim=clean,
                                            reason=f"unknown evidence {tag}"))
                continue
            known_tags.append(tag)
        if nums:
            # Pool numbers across every tagged record's payload (matching
            # ground_check), so a citation sentence can be corroborated by
            # numbers in a paper's abstract, not only by eval-result data.
            available = [n for tag in known_tags
                         for n in numbers_in_payload(store.get(tag).payload)]
            for num in nums:
                if not any(math.isclose(num, a,
                           rel_tol=config.verifier.numeric_tolerance,
                           abs_tol=1e-9) for a in available):
                    violations.append(Violation(
                        claim=clean,
                        reason=f"number {num} not supported by evidence "
                               f"{known_tags}"))
        for tag in known_tags:
            rec = store.get(tag)
            if rec.type == "paper":
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
                        claim=clean, reason="method claim not matched by code"))
    return violations


def run_verifier(llm: LLMClient, config: Config, run_dir: Path, paper_md: str,
                 store: EvidenceStore, references: list[dict],
                 solution_code: str) -> VerifierResult:
    run_dir = Path(run_dir)
    paper_md = normalize_tags(paper_md)
    body, references_section = _split_references(paper_md)
    violations = _verify(llm, config, body, store, solution_code)
    if violations:
        listing = "\n".join(f"- {v.claim}: {v.reason}" for v in violations)
        body = normalize_tags(llm.chat(
            "reasoning",
            "You repair research papers. Rewrite each flagged sentence to "
            "match its evidence, or DELETE it if it cannot be supported. "
            "Keep all valid {ev:...} tags. Reply with the full markdown.",
            f"Paper:\n{body}\n\nFlagged claims:\n{listing}",
        ) or body)
        violations = _verify(llm, config, body, store, solution_code)
    paper_md = body + references_section
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
