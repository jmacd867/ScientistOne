import math
import re

from pydantic import BaseModel

from ..evidence import EvidenceStore

TAG_RE = re.compile(r"\{ev:(ev_\d+)\}")
_NUM_RE = re.compile(r"\d+\.\d+|\d{2,}")
# Sentence boundary: after .!? unless a {ev:...} tag follows (tags trail their
# sentence), and after a closing tag brace.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\{ev:)|(?<=\})\s+")


class GroundIssue(BaseModel):
    kind: str
    detail: str


def sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.extend(s.strip() for s in _SPLIT_RE.split(line) if s.strip())
    return out


def numbers_in_text(s: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(TAG_RE.sub("", s))]


def numbers_in_payload(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [float(m) for m in _NUM_RE.findall(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers_in_payload(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in numbers_in_payload(v)]
    return []


def ground_check(narrative: str, store: EvidenceStore,
                 tolerance: float) -> list[GroundIssue]:
    issues: list[GroundIssue] = []
    for sent in sentences(narrative):
        tags = TAG_RE.findall(sent)
        known, unknown = [], []
        for tag in tags:
            (unknown, known)[tag in {r.id for r in store.all()}].append(tag)
        for tag in unknown:
            issues.append(GroundIssue(kind="unknown-tag",
                                      detail=f"{tag} in: {sent}"))
        nums = numbers_in_text(sent)
        if nums and not tags:
            issues.append(GroundIssue(kind="untagged-numeric", detail=sent))
            continue
        if nums and known:
            available = [n for tag in known
                         for n in numbers_in_payload(store.get(tag).payload)]
            for num in nums:
                if not any(math.isclose(num, a, rel_tol=tolerance, abs_tol=1e-9)
                           for a in available):
                    issues.append(GroundIssue(
                        kind="number-mismatch",
                        detail=f"{num} not in evidence for: {sent}"))
    return issues
