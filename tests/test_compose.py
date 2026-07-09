from pathlib import Path

from scientist_one.config import Config
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.compose import compose
from scientist_one.writer.render import MarkdownRenderer

TASK = load_task(Path("tasks/bin_packing"))
REFS = [{"title": "FFD Analysis", "authors": ["D. Johnson"], "year": 1974,
         "url": "https://s2/ffd", "abstract": "", "source": "semantic_scholar",
         "external_id": "10.1/ffd"}]


def test_compose_renders_with_references(tmp_path):
    body = "## Results\nRatio 1.08. {ev:ev_0003}"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([body]))
    paper = compose(llm, TASK, "narrative", REFS)
    assert paper.startswith("# ")
    assert "{ev:ev_0003}" in paper           # tags preserved for the verifier
    assert "FFD Analysis" in paper           # references section present
    assert "D. Johnson" in paper


def test_compose_falls_back_on_empty_reply(tmp_path):
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(["  "]))
    paper = compose(llm, TASK, "the narrative", REFS)
    assert "the narrative" in paper


def test_markdown_renderer_no_references():
    out = MarkdownRenderer().render("Title", "body", [])
    assert "## References" not in out
