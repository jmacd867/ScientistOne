import math
import re

from pydantic import BaseModel

from ..evidence import EvidenceStore

TAG_RE = re.compile(r"\{ev:(ev_\d+)\}")
# Comma-grouped integer (optionally with a decimal tail) first, so a number
# like "32,000,000" is read as one value instead of the comma splitting it
# into three digit runs ("32", "000", "000") that each spuriously fail an
# evidence-support check for a claim that was actually correct.
_NUM_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{2,}")
# Sentence boundary: after .!? unless a {ev:...} tag follows (tags trail their
# sentence), and after a closing tag brace.
_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?!\{ev:)|(?<=\})\s+")
# Matches one-or-more evidence IDs bundled in a single bracket pair, in
# either bracket style: "[ev:ev_0047, ev:ev_0048]" or "{ev:ev_0047}".
_MULTI_TAG_RE = re.compile(r"[\{\[]\s*ev:\s*(ev_\d+(?:\s*,\s*ev:\s*ev_\d+)*)\s*[\}\]]")
# Loose citation shape: any bracket pair starting with "ev:", regardless of
# what's inside — used to catch garbled IDs (e.g. "{ev:ev_004CA}") that
# don't match TAG_RE/_MULTI_TAG_RE's strict ev_\d+ requirement at all, so
# they don't silently leak their embedded digits into a numeric-claim check.
_LOOSE_TAG_RE = re.compile(r"[\{\[]\s*ev:\s*([^{}\[\]]*?)\s*[\}\]]")
# Citation missing the "ev:" prefix entirely — e.g. "{ev_0025}", optionally
# wrapped in backticks or dollar signs (LaTeX math mode), with the brace and
# underscore themselves sometimes backslash-escaped: "$\{ev\_0025\}$". The
# "ev:" requirement in every regex above means this shape is otherwise
# invisible as a tag, so its digits leak into the numeric-claim check as an
# apparently unsupported number.
_BARE_ID_RE = re.compile(r"[`$]*\\?\{\\?ev\\?_(\d+)\\?\}[`$]*")


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

    Also fixes citations missing the "ev:" prefix entirely (e.g. "{ev_0025}"
    or LaTeX-escaped "$\\{ev\\_0025\\}$") — a distinct malformation from the
    bracket/multi-ID cases above, since those all still contain "ev:".

    Does NOT fix a wrong ID (e.g. a dropped digit like "ev_016" for
    "ev_0016") — that's a real hallucination the caller should still catch
    as an unknown-evidence violation, not something to paper over here.
    """
    def _expand(match: re.Match) -> str:
        ids = re.findall(r"ev_\d+", match.group(1))
        return "".join(f"{{ev:{i}}}" for i in ids)
    text = _MULTI_TAG_RE.sub(_expand, text)
    return _BARE_ID_RE.sub(lambda m: f"{{ev:ev_{m.group(1)}}}", text)


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


def malformed_tags(sent: str) -> list[str]:
    """Citation-shaped brackets that don't reduce to a canonical {ev:ev_XXXX}
    tag even after normalize_tags — e.g. a hallucinated ID with letters in
    it ("{ev:ev_004CA}"). Returns each such occurrence verbatim, so it can be
    surfaced as its own violation instead of its embedded digits leaking
    into a numeric-claim check as an unrelated "fact"."""
    out = []
    for m in _LOOSE_TAG_RE.finditer(sent):
        if not re.fullmatch(r"ev_\d+", m.group(1).strip()):
            out.append(m.group(0))
    return out


def near_miss_hint(tag: str, known_ids: set[str]) -> str:
    """If inserting a single '0' right after 'ev_' turns this tag into a
    real ID, say so — models have been observed reliably dropping exactly
    one leading zero when copying 4-digit zero-padded evidence IDs (e.g.
    writing "ev_046" for "ev_0046"). Giving the refiner this hint directly,
    rather than just "unknown", makes the very next attempt far more likely
    to actually fix it instead of guessing again."""
    if not tag.startswith("ev_"):
        return ""
    candidate = f"ev_0{tag[len('ev_'):]}"
    return f" (did you mean {candidate}?)" if candidate in known_ids else ""


def _parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def numbers_in_text(s: str) -> list[float]:
    s = TAG_RE.sub("", s)
    s = _LOOSE_TAG_RE.sub("", s)
    return [_parse_num(m) for m in _NUM_RE.findall(s)]


def numbers_in_payload(obj) -> list[float]:
    if isinstance(obj, bool):
        return []
    if isinstance(obj, (int, float)):
        return [float(obj)]
    if isinstance(obj, str):
        return [_parse_num(m) for m in _NUM_RE.findall(obj)]
    if isinstance(obj, dict):
        return [n for v in obj.values() for n in numbers_in_payload(v)]
    if isinstance(obj, list):
        return [n for v in obj for n in numbers_in_payload(v)]
    return []


def ground_check(narrative: str, store: EvidenceStore,
                 tolerance: float) -> list[GroundIssue]:
    issues: list[GroundIssue] = []
    known_ids = {r.id for r in store.all()}
    for sent in sentences(narrative):
        tags = TAG_RE.findall(sent)
        known, unknown = [], []
        for tag in tags:
            (unknown, known)[tag in known_ids].append(tag)
        for tag in unknown:
            hint = near_miss_hint(tag, known_ids)
            issues.append(GroundIssue(kind="unknown-tag",
                                      detail=f"{tag}{hint} in: {sent}"))
        for garbage in malformed_tags(sent):
            issues.append(GroundIssue(kind="malformed-tag",
                                      detail=f"{garbage} in: {sent}"))
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
