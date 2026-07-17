# ScientistOne Mini — Design

**Date:** 2026-07-07
**Status:** Approved (pending implementation plan)
**Source paper:** "ScientistOne: Towards Human-Level Autonomous Research via Chain-of-Evidence" (arXiv:2605.26340, Google Cloud AI Research, May 2026)

## Goal

A faithful mini-replica of the ScientistOne multi-agent research workflow: all the
architectural pieces of the paper (Problem Investigator, Ideator + Parallel
Explore-Exploit discovery, five-phase Paper Writer, Claim Verifier, CoE Integrity
Audit) at a scale that runs on local hardware, with a pluggable task interface so
future research problems can be dropped in without touching the pipeline.

**Non-goals:** matching the paper's scale (100 PDFs, ADRS benchmark, frontier
models), LaTeX/PDF output in v1, any cloud/API dependency for inference.

## Decisions made

| Decision | Choice |
|---|---|
| Scope | Faithful mini-replica — structure matches the paper, scale doesn't |
| Task domain | Pluggable Task/Evaluator interface + built-in bin-packing demo task |
| Runtime | Python package, local inference via Ollama (no agent framework) |
| Models | gemma4:26b for reasoning-heavy roles, gemma4:12b for high-volume judging; per-role assignment in config |
| Output | Markdown paper via a `Renderer` interface; LaTeX renderer addable later |
| Orchestration | Staged pipeline (Approach A) — one module per paper component, explicit control flow |
| Literature | Real retrieval: Semantic Scholar + arXiv free APIs, ~10–20 papers |

## Core principle: the evidence chain

Chain-of-Evidence (CoE) is implemented as a first-class data structure, not prompt
discipline. Each run gets a workspace `runs/<timestamp>/` containing an append-only
`evidence.jsonl`. Each record:

```json
{"id": "ev_0042", "type": "eval-result", "stage": "discovery",
 "payload": {...}, "sources": ["ev_0031", "ev_0007"], "created_at": "..."}
```

Record types: `paper`, `brief-claim`, `idea`, `solution`, `eval-result`,
`audit-flag`, `ablation`, `draft-claim`. The store enforces at append time that
every ID in `sources` already exists. Every stage reads upstream records by ID and
writes new ones, so any claim in the final paper traces through recorded IDs to a
grounding source (an evaluator log line, retrieved paper metadata, an ablation
result).

## Architecture

```
scientist_one/
├── llm.py              # Ollama client wrapper: chat, JSON-schema-constrained output,
│                       #   retries, per-call logging, model registry, fake backend hook
├── evidence.py         # Evidence chain store (JSONL, provenance IDs, integrity checks)
├── tasks/
│   ├── base.py         # Task + Evaluator protocols, task-directory loader
│   └── bin_packing/    # Demo task
├── investigator/       # Stage 1
├── discovery/          # Stage 2: ideator.py, pee.py (orchestrator), solver.py, auditor.py
├── writer/             # Stage 3a: conceive.py, ground.py, critic.py, resolve.py, compose.py
│   └── render/         # Renderer interface + markdown renderer
├── verifier/           # Stage 3b: claim extraction, per-type checks, refiner
├── audit/              # CoE Integrity Audit: I1–I4 checks + report
└── cli.py              # scientist-one run | resume | audit | status
```

Dependencies: `ollama`, `httpx`, `pydantic`, `pyyaml`, `pytest` (dev). No agent
frameworks.

### Stage 1 — Problem Investigator (`investigator/`)

Input: task definition (+ optional seed paper IDs / search queries from `task.yaml`).

1. Query Semantic Scholar + arXiv APIs; expand one hop through citations of seeds.
2. LLM-score relevance (12b), keep top N (default 15).
3. Fetch abstracts (full-text PDFs best-effort, not required).
4. Produce a **research brief** (26b): problem framing, candidate approaches from
   the literature, baseline expectations — every statement tagged with the `paper`
   record it derives from.

Output: brief (`brief-claim` records) + `references.json` with API-verified
metadata. Zero references from model memory: a reference exists only if an API
returned it.

### Stage 2 — Discovery (`discovery/`)

1. **Ideator** (26b): two generation prompts — conservative and unconventional —
   each producing candidate approaches grounded in the brief; scored on novelty +
   feasibility; top proposals seed **B branches** (default 3).
2. **PEE orchestrator**: per branch, loop up to I iterations (default 4):
   - **Solve** (26b): write/revise Python solution code against the task's
     `starter.py` template, given prior scores and logs as feedback.
   - **Evaluate**: run the solution in a sandboxed subprocess (timeout-bounded,
     isolated from the orchestrator process) against the task's `evaluator.py`;
     capture score + log. Crashes score 0 with traceback captured — a data
     point, not a pipeline failure. **v1 does not deny network access** — see
     Robustness below.
   - **Audit** (12b): check solution for specification violations (hardcoded
     answers, evaluator gaming); flagged solutions are excluded from selection.
   After each round, top-K branches (default 2) survive; pruned slots refill with
   fresh ideation informed by the surviving branches' feedback.
3. **Best-run selection + ablations**: pick the best-scoring non-flagged solution
   (respecting the task's metric direction — higher- or lower-is-better);
   for each named component in the solution, disable it and re-evaluate
   (`ablation` records).

All scores, logs, flags, and ablations land in the evidence chain.

### Stage 3a — Paper Writer (`writer/`)

Five phases mirroring the paper:

1. **Conceive** (26b): build a markdown research narrative from the brief,
   experimental logs, verified scores, solution code, and ablations. Every factual
   sentence carries an inline evidence tag `{ev:ev_0042}` binding it to an
   evidence record (`draft-claim` records).
2. **Ground** (deterministic, no LLM): validate every tag — reported score matches
   the best-run `eval-result`, referenced records exist, baselines trace to brief
   entries or are marked ESTIMATED.
3. **Critic** (26b): audit what deterministic checks can't — internal
   contradictions, overclaims, missing comparisons, baseline fairness. Returns
   PASS or a list of issues.
4. **Resolve** (26b): rewrite against Ground flags + Critic issues jointly,
   dropping unsupported claims and calibrating overclaims. Ground→Critic→Resolve
   loops until clean or plateau (max 3 rounds).
5. **Compose**: render the grounded representation section-by-section through the
   `Renderer` interface. v1 renderer: markdown (`paper.md` + references section).
   Renderer receives verified numbers and named baselines alongside the narrative
   so it writes prose around established facts.

### Stage 3b — Claim Verifier (`verifier/`)

Composition can introduce drift, so the composed draft is re-verified:

1. **Extract** claims from the rendered draft (12b, JSON-constrained).
2. **Dispatch by type**: numerical → match against evaluator-log records within
   tolerance (default 1%); citation → match against `references.json` (LLM
   abstract-entailment check for content consistency); methodological → compare
   against the actual solution code (12b judge).
3. **Flag** unsourced or contradicted claims.
4. **Refiner** (26b): rewrite flagged sentences to match their evidence, remove
   claims that cannot be supported, strip inline evidence tags from the final
   output. Only a draft with no remaining blocking violations is promoted to
   `paper.md`; otherwise the run ends with `paper.draft.md` + a violations report.

### CoE Integrity Audit (`audit/`)

`scientist-one audit runs/<id>` — post-hoc, usable on the system's own outputs:

- **I1 Score Verification**: extract reported score from the paper (LLM), re-run
  the submitted solution on the task evaluator, compare within tolerance.
- **I2 Specification Violation**: LLM judges (12b) inspect solution code against
  evaluator + task spec; majority vote of 3.
- **I3 Reference Verification**: resolve each reference via Semantic Scholar /
  arXiv / Crossref; LLM disambiguation of near-misses; unmatched → hallucinated.
- **I4 Method–Code Alignment**: LLM judges read method section vs. solution code;
  majority vote of 3; simplification = aligned, different algorithm = misaligned.

Output: a report card (per-check pass/fail + details) written to the run dir.

## Task plugin interface

```
tasks/<name>/
├── task.yaml        # name, description, metric direction, seed queries/paper IDs,
│                    #   optional per-task overrides (timeout, tolerance)
├── evaluator.py     # def evaluate(solution_path, workdir) -> {"score": float, "log": str}
├── starter.py       # solution template defining the required function interface
└── data/            # optional fixtures
```

Hard requirement: deterministic evaluation (audit's score re-verification depends
on it). Tasks can live anywhere on disk (`scientist-one run --task <path>`).

**Demo task — 1D bin packing heuristic discovery.** `starter.py` defines
`pack(items, capacity) -> list[list[float]]`; evaluator runs fixed instance sets,
scores by bins used relative to the L1 lower bound (lower is better). Deterministic,
fast, and has real literature for Stage 1 (first-fit-decreasing, best-fit, etc.).

## Configuration

Single `config.yaml`:

```yaml
models:
  reasoning: gemma4:26b     # ideator, solver, conceive, critic, resolve, refiner
  judging: gemma4:12b       # relevance scoring, audits, claim extraction, judges
ollama_host: http://localhost:11434
discovery: {branches: 3, iterations: 4, survivors: 2}
investigator: {max_papers: 15}
writer: {max_rounds: 3}
verifier: {numeric_tolerance: 0.01}
solver: {timeout_s: 60}
```

## Robustness

- **Sandboxed execution**: solver code runs in a subprocess with a timeout,
  never in the orchestrator process. **Known limitation:** v1 does not deny
  network access — full network isolation isn't portably achievable without
  containers (a Linux-only `unshare -n` wrapper is a possible future
  hardening step, but adds a real dependency for a single-user local tool).
  Solver code is LLM-generated, not adversarial-user-supplied, so this is an
  accepted risk for this project's scope, not a promise this doc no longer
  makes.
- **LLM reliability**: all calls via `llm.py` — JSON-schema-constrained output
  (Ollama `format` parameter) where structure is needed, up to 2 retries on
  malformed output, every call logged (prompt, response, model, duration) to the
  run workspace. After retries exhausted, the stage degrades explicitly (branch
  marked failed, claim marked unverifiable) rather than crashing.
- **Resumability**: each stage writes a completion marker; `--resume` skips
  completed stages.
- **Network**: scholarly API calls retry with backoff; Stage 1 can proceed with
  fewer papers if some fetches fail (recorded as such).

## Testing

- **Unit (pytest, no LLM)**: evidence store integrity (append-time source-ID
  enforcement), Ground's deterministic checks, claim extraction/dispatch,
  subprocess sandboxing (timeout, crash capture), task loading/validation.
- **Fake-LLM pipeline tests**: `llm.py` accepts a scripted fake backend; the full
  pipeline runs in seconds with canned responses, verifying orchestration order,
  evidence flow, resume logic, and failure degradation.
- **Live smoke test** (opt-in, `pytest -m live`): 1 branch × 1 iteration against
  real Ollama to catch API drift.

## Error handling summary

| Failure | Behavior |
|---|---|
| Solver code crashes/hangs | Score 0, traceback captured as evidence, loop continues |
| LLM output malformed after retries | Stage-level explicit degradation (failed branch / unverifiable claim) |
| Scholarly API down | Proceed with fewer papers; recorded in evidence |
| Claim verification finds blocking violations after refinement | Run ends with `paper.draft.md` + violations report, not a false "clean" paper |
| Run interrupted | `--resume` continues from last completed stage |

## Milestones (for the implementation plan)

1. Skeleton: package layout, `llm.py` (+ fake backend), `evidence.py`, config, CLI stub — with unit tests.
2. Task interface + bin-packing demo task + sandboxed evaluator runner.
3. Stage 2 Discovery (Ideator, PEE loop, ablations) — the heart of the system; testable end-to-end with fake LLM.
4. Stage 1 Investigator (scholarly APIs, brief).
5. Stage 3 Writer + Claim Verifier + markdown renderer.
6. CoE Integrity Audit + `audit` CLI.
7. Live end-to-end run on bin packing; tune prompts for gemma4 models.
