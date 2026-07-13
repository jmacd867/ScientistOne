import math
import re

from pydantic import BaseModel

from ..evidence import EvidenceStore

TAG_RE = re.compile(r"\{ev:(ev_\d+)\}")
_NUM_RE = re.compile(r"\d+\.\d+|\d{2,}")
# Sentence boundary: after .!? unless a {ev:...} tag follows (tags trail their
# sentence), and after a closing tag brace.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\{ev:)|(?<=\})\s+")
# Matches one-or-more evidence IDs bundled in a single bracket pair, in
# either bracket style: "[ev:ev_0047, ev:ev_0048]" or "{ev:ev_0047}".
_MULTI_TAG_RE = re.compile(r"[\{\[]\s*ev:\s*(ev_\d+(?:\s*,\s*ev:\s*ev_\d+)*)\s*[\}\]]")


def normalize_tags(text: str) -> str:
    """Rewrite evidence-tag citations into canonical single-ID {ev:ev_XXXX} tags.

    Models sometimes cite multiple records in one bracket pair (e.g.
    "[ev:ev_0047, ev:ev_0048]" or "{ev:ev_0047, ev:ev_0048}") or use square
    brackets instead of curly braces. TAG_RE only recognizes exactly
    "{ev:ev_XXXX}" (one ID), so any deviation silently fails to register as
    a tag — and the digits inside the malformed bracket then get misread as
    an unverified numeric claim. This expands every such variant into one
    canonical tag per ID before any tag-based logic runs. Already-canonical
    text passes through unchanged (idempotent).

    Does NOT fix a wrong ID (e.g. a dropped digit like "ev_016" for
    "ev_0016") — that's a real hallucination the caller should still catch
    as an unknown-evidence violation, not something to paper over here.
    """
    def _expand(match: re.Match) -> str:
        ids = re.findall(r"ev_\d+", match.group(1))
        return "".join(f"{{ev:{i}}}" for i in ids)
    return _MULTI_TAG_RE.sub(_expand, text)


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
