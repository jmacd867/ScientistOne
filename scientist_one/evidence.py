import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel


class EvidenceRecord(BaseModel):
    id: str
    type: str
    stage: str
    payload: dict
    sources: list[str] = []
    created_at: str


class EvidenceStore:
    """Append-only JSONL evidence chain with provenance-ID integrity."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._records: dict[str, EvidenceRecord] = {}
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = EvidenceRecord.model_validate_json(line)
                    self._records[rec.id] = rec

    def append(self, type: str, stage: str, payload: dict,
               sources: list[str] | None = None) -> str:
        sources = sources or []
        for sid in sources:
            if sid not in self._records:
                raise ValueError(f"unknown source id: {sid}")
        rid = f"ev_{len(self._records) + 1:04d}"
        rec = EvidenceRecord(
            id=rid, type=type, stage=stage, payload=payload, sources=sources,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(rec.model_dump_json() + "\n")
        self._records[rid] = rec
        return rid

    def get(self, record_id: str) -> EvidenceRecord:
        return self._records[record_id]

    def by_type(self, type: str) -> list[EvidenceRecord]:
        return [r for r in self._records.values() if r.type == type]

    def all(self) -> list[EvidenceRecord]:
        return list(self._records.values())
