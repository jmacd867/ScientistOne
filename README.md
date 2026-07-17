# ScientistOne Mini

A local mini-replica of the ScientistOne autonomous-research pipeline
(arXiv:2605.26340) running on Ollama models. Pipeline: Problem Investigator
(real Semantic Scholar/arXiv retrieval) → Discovery (ideation + parallel
explore-exploit over sandboxed solutions) → Paper Writer (conceive → ground →
critic → resolve → compose) → Claim Verifier, with an append-only evidence
chain connecting every claim to its grounding source, plus a post-hoc CoE
Integrity Audit.

## Setup

    docker start ollama          # or however you run Ollama
    ollama pull gemma4:26b gemma4:12b
    pip install -e ".[dev]"

## Usage

    scientist-one run --task tasks/bin_packing        # full run
    scientist-one status runs/<id>                    # stage progress
    scientist-one run --task tasks/bin_packing --run-dir runs/<id>   # resume
    scientist-one audit runs/<id>                     # CoE integrity audit

Outputs land in `runs/<timestamp>/`: `paper.md` (verified) or
`paper.draft.md` + `violations.json`, `evidence.jsonl`, `brief.md`,
`references.json`, `solutions/`, `audit_report.json`.

## Adding your own research task

Create a directory with:

    my_task/
    ├── task.yaml        # name, description, metric_direction, seed_queries
    ├── starter.py       # the solution interface the solver must implement
    ├── evaluator.py     # def evaluate(solution_path, workdir) -> {"score", "log"}
    └── data/            # optional fixtures

The evaluator must be deterministic. Then:

    scientist-one run --task path/to/my_task

## Tests

    pytest                       # fast, no Ollama needed (fake LLM backend)
    pytest -m live               # tiny end-to-end run against real Ollama

## Configuration

Edit `config.yaml`: model per role (reasoning/judging), branch counts,
iterations, paper limits, timeouts, numeric tolerance.
