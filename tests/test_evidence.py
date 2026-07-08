import pytest
from scientist_one.evidence import EvidenceStore


def test_append_and_get(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    rid = store.append("paper", "investigator", {"title": "FFD"})
    assert rid == "ev_0001"
    rec = store.get(rid)
    assert rec.payload["title"] == "FFD"
    assert rec.sources == []


def test_sources_must_exist(tmp_path):
    store = EvidenceStore(tmp_path / "evidence.jsonl")
    with pytest.raises(ValueError):
        store.append("brief-claim", "investigator", {}, sources=["ev_0099"])


def test_reload_from_disk(tmp_path):
    path = tmp_path / "evidence.jsonl"
    s1 = EvidenceStore(path)
    rid = s1.append("paper", "investigator", {"t": 1})
    s2 = EvidenceStore(path)
    assert s2.get(rid).payload == {"t": 1}
    assert s2.append("idea", "discovery", {}, sources=[rid]) == "ev_0002"


def test_by_type(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    store.append("paper", "investigator", {})
    store.append("idea", "discovery", {})
    assert [r.type for r in store.by_type("idea")] == ["idea"]
