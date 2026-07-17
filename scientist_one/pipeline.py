import json
from pathlib import Path

import httpx

from .config import Config
from .discovery.pee import DiscoveryResult, run_discovery
from .evidence import EvidenceStore
from .investigator.run import InvestigatorResult, run_investigator
from .llm import Backend, LLMClient
from .tasks.base import load_task
from .verifier.run import run_verifier
from .writer.compose import compose
from .writer.run import run_writer


def _stage(run_dir: Path, name: str, fn):
    """Run fn() unless a completion marker exists; persist result as JSON."""
    marker = run_dir / f"{name}.json"
    if marker.exists():
        return json.loads(marker.read_text())
    result = fn()
    marker.write_text(json.dumps(result))
    return result


def run_pipeline(config: Config, task_path: Path, run_dir: Path,
                 backend: Backend | None = None,
                 http_client: httpx.Client | None = None) -> dict:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    task = load_task(Path(task_path))
    store = EvidenceStore(run_dir / "evidence.jsonl")
    llm = LLMClient(config, run_dir, backend=backend)

    inv = _stage(run_dir, "investigator", lambda: run_investigator(
        llm, config, task, store, run_dir, http_client=http_client).model_dump())

    def _discovery():
        result = run_discovery(llm, config, task, store, run_dir,
                               inv["brief_text"], inv["brief_ids"])
        return result.model_dump() if result else None

    disc = _stage(run_dir, "discovery", _discovery)

    manifest = {"task_path": str(task_path), "references_path":
                inv["references_path"], "best_solution_path": None,
                "best_eval_id": None, "paper_path": None, "promoted": False}
    if disc is None:
        manifest["status"] = "discovery-failed"
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        return manifest

    def _writer():
        discovery = DiscoveryResult.model_validate(disc)
        writer = run_writer(llm, config, task, store, inv["brief_text"], discovery)
        references = json.loads(Path(inv["references_path"]).read_text())
        paper_md = compose(llm, task, writer.narrative, references)
        code = Path(discovery.best_solution_path).read_text()
        verdict = run_verifier(llm, config, run_dir, paper_md, store,
                               references, code)
        return {"remaining_issues": writer.remaining_issues,
                "verifier": verdict.model_dump()}

    wr = _stage(run_dir, "writer", _writer)

    manifest.update({
        "best_solution_path": disc["best_solution_path"],
        "best_eval_id": disc["best_eval_id"],
        "paper_path": wr["verifier"]["paper_path"],
        "promoted": wr["verifier"]["promoted"],
        "status": "complete" if wr["verifier"]["promoted"] else "not-promoted",
    })
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
