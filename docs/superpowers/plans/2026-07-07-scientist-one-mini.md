# ScientistOne Mini Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local, faithful mini-replica of the ScientistOne autonomous-research pipeline (arXiv:2605.26340): Problem Investigator → Discovery (Ideator + Parallel Explore-Exploit) → Paper Writer + Claim Verifier, plus a post-hoc CoE Integrity Audit, running on Ollama models with a pluggable task interface.

**Architecture:** Plain Python package, one module per paper component, explicit staged control flow (no agent framework). An append-only JSONL evidence chain (`evidence.jsonl` per run) is the spine: every stage reads upstream records by ID and appends new ones, so every claim in the final paper traces to a grounding source. Solution code executes in sandboxed subprocesses; all LLM calls go through one wrapper with a fake backend for tests.

**Tech Stack:** Python 3.11+, `ollama` (client), `httpx`, `pydantic` v2, `pyyaml`, `pytest`. Models: gemma4:26b (reasoning), gemma4:12b (judging), configurable.

## Global Constraints

- Python 3.11+; dependencies limited to: `ollama`, `httpx`, `pydantic>=2`, `pyyaml` (dev: `pytest`). No agent frameworks.
- Evidence store enforces at append time that every `sources` ID already exists.
- Solver/evaluator code never runs in the orchestrator process — always a subprocess with timeout.
- Every LLM call is logged to the run workspace; JSON-constrained calls retry up to 2 times on malformed output, then the stage degrades explicitly (no crash).
- Task metric direction (`higher`/`lower` is better) must be respected in all comparisons.
- References may only come from scholarly API responses, never from model memory.
- All tests except those marked `-m live` must run without Ollama.
- Spec: `docs/superpowers/specs/2026-07-07-scientist-one-mini-design.md`.

---

### Task 1: Package skeleton and config

**Files:**
- Create: `pyproject.toml`, `config.yaml`, `scientist_one/__init__.py`, `scientist_one/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `scientist_one.config.Config` (pydantic model) with fields `models: ModelConfig(reasoning: str, judging: str)`, `ollama_host: str`, `discovery: DiscoveryConfig(branches: int, iterations: int, survivors: int)`, `investigator: InvestigatorConfig(max_papers: int)`, `writer: WriterConfig(max_rounds: int)`, `verifier: VerifierConfig(numeric_tolerance: float)`, `solver: SolverConfig(timeout_s: int)`; and `load_config(path: Path | None) -> Config` (missing file → all defaults).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from pathlib import Path
from scientist_one.config import Config, load_config


def test_defaults_when_no_file(tmp_path):
    cfg = load_config(tmp_path / "missing.yaml")
    assert cfg.models.reasoning == "gemma4:26b"
    assert cfg.models.judging == "gemma4:12b"
    assert cfg.discovery.branches == 3
    assert cfg.solver.timeout_s == 60


def test_loads_overrides(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text("discovery: {branches: 1, iterations: 2, survivors: 1}\n")
    cfg = load_config(p)
    assert cfg.discovery.branches == 1
    assert cfg.models.reasoning == "gemma4:26b"  # untouched defaults survive
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scientist_one'`

- [ ] **Step 3: Write the package skeleton and config**

```toml
# pyproject.toml
[project]
name = "scientist-one"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["ollama", "httpx", "pydantic>=2", "pyyaml"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
scientist-one = "scientist_one.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["scientist_one*"]

[tool.pytest.ini_options]
markers = ["live: requires a running Ollama server"]
addopts = "-m 'not live'"
```

```python
# scientist_one/__init__.py
```

```python
# scientist_one/config.py
from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    reasoning: str = "gemma4:26b"
    judging: str = "gemma4:12b"


class DiscoveryConfig(BaseModel):
    branches: int = 3
    iterations: int = 4
    survivors: int = 2


class InvestigatorConfig(BaseModel):
    max_papers: int = 15


class WriterConfig(BaseModel):
    max_rounds: int = 3


class VerifierConfig(BaseModel):
    numeric_tolerance: float = 0.01


class SolverConfig(BaseModel):
    timeout_s: int = 60


class Config(BaseModel):
    models: ModelConfig = ModelConfig()
    ollama_host: str = "http://localhost:11434"
    discovery: DiscoveryConfig = DiscoveryConfig()
    investigator: InvestigatorConfig = InvestigatorConfig()
    writer: WriterConfig = WriterConfig()
    verifier: VerifierConfig = VerifierConfig()
    solver: SolverConfig = SolverConfig()


def load_config(path: Path | None = None) -> Config:
    if path is None or not Path(path).exists():
        return Config()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(data)
```

```yaml
# config.yaml
models:
  reasoning: gemma4:26b
  judging: gemma4:12b
ollama_host: http://localhost:11434
discovery: {branches: 3, iterations: 4, survivors: 2}
investigator: {max_papers: 15}
writer: {max_rounds: 3}
verifier: {numeric_tolerance: 0.01}
solver: {timeout_s: 60}
```

- [ ] **Step 4: Install editable and run test to verify it passes**

Run: `pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml config.yaml scientist_one/ tests/
git commit -m "feat: package skeleton and typed config loading"
```

---

### Task 2: Evidence chain store

**Files:**
- Create: `scientist_one/evidence.py`
- Test: `tests/test_evidence.py`

**Interfaces:**
- Produces: `EvidenceRecord` (pydantic: `id: str`, `type: str`, `stage: str`, `payload: dict`, `sources: list[str]`, `created_at: str`) and `EvidenceStore(path: Path)` with methods `append(type: str, stage: str, payload: dict, sources: list[str] | None = None) -> str` (returns new id `ev_0001`-style, raises `ValueError` on unknown source id), `get(record_id: str) -> EvidenceRecord` (raises `KeyError` if missing), `by_type(type: str) -> list[EvidenceRecord]`, `all() -> list[EvidenceRecord]`. Store reloads existing records from the JSONL file on construction (resume support).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evidence.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evidence.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `scientist_one.evidence`

- [ ] **Step 3: Implement the store**

```python
# scientist_one/evidence.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evidence.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/evidence.py tests/test_evidence.py
git commit -m "feat: append-only evidence chain store with provenance integrity"
```

---

### Task 3: LLM wrapper with fake backend

**Files:**
- Create: `scientist_one/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `Config` (Task 1).
- Produces: `LLMClient(config: Config, log_dir: Path, backend: Backend | None = None)` with `chat(role: str, system: str, user: str) -> str` and `chat_json(role: str, system: str, user: str, schema: type[BaseModel]) -> BaseModel | None` (role is `"reasoning"` or `"judging"`; `chat_json` retries up to 2 extra times on unparseable output, returns `None` after exhaustion — callers degrade explicitly). Also `FakeBackend(responses: list[str])` (pops responses in order; raises `IndexError` when exhausted) and `LLMError`. `Backend` is any callable `(model: str, system: str, user: str, format: dict | None) -> str`. Every call appends a JSON line to `log_dir / "llm_calls.jsonl"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import json
from pydantic import BaseModel
from scientist_one.config import Config
from scientist_one.llm import FakeBackend, LLMClient


class Verdict(BaseModel):
    flagged: bool
    reason: str


def make_client(tmp_path, responses):
    return LLMClient(Config(), tmp_path, backend=FakeBackend(responses))


def test_chat_returns_text_and_logs(tmp_path):
    client = make_client(tmp_path, ["hello"])
    assert client.chat("reasoning", "sys", "usr") == "hello"
    lines = (tmp_path / "llm_calls.jsonl").read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["model"] == "gemma4:26b"
    assert entry["response"] == "hello"


def test_chat_json_parses_model(tmp_path):
    client = make_client(tmp_path, ['{"flagged": true, "reason": "hardcoded"}'])
    v = client.chat_json("judging", "sys", "usr", Verdict)
    assert v.flagged is True


def test_chat_json_retries_then_none(tmp_path):
    client = make_client(tmp_path, ["not json", "still not", "nope"])
    assert client.chat_json("judging", "sys", "usr", Verdict) is None


def test_chat_json_recovers_on_retry(tmp_path):
    client = make_client(tmp_path, ["bad", '{"flagged": false, "reason": "ok"}'])
    v = client.chat_json("judging", "sys", "usr", Verdict)
    assert v.reason == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL with `ImportError` for `scientist_one.llm`

- [ ] **Step 3: Implement the wrapper**

```python
# scientist_one/llm.py
import json
import re
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from .config import Config

Backend = Callable[[str, str, str, dict | None], str]


class LLMError(Exception):
    pass


class FakeBackend:
    """Scripted backend for tests: returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, model, system, user, format):
        return self.responses.pop(0)


def _ollama_backend(host: str) -> Backend:
    import ollama

    client = ollama.Client(host=host)

    def call(model: str, system: str, user: str, format: dict | None) -> str:
        resp = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=format,
        )
        return resp["message"]["content"]

    return call


class LLMClient:
    def __init__(self, config: Config, log_dir: Path, backend: Backend | None = None):
        self.config = config
        self.log_path = Path(log_dir) / "llm_calls.jsonl"
        self.backend = backend or _ollama_backend(config.ollama_host)

    def _model(self, role: str) -> str:
        return {"reasoning": self.config.models.reasoning,
                "judging": self.config.models.judging}[role]

    def _call(self, role: str, system: str, user: str, format: dict | None) -> str:
        model = self._model(role)
        start = time.monotonic()
        response = self.backend(model, system, user, format)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({
                "model": model, "system": system, "user": user,
                "response": response, "duration_s": round(time.monotonic() - start, 2),
            }) + "\n")
        return response

    def chat(self, role: str, system: str, user: str) -> str:
        return self._call(role, system, user, None)

    def chat_json(self, role: str, system: str, user: str,
                  schema: type[BaseModel]) -> BaseModel | None:
        fmt = schema.model_json_schema()
        for _ in range(3):
            raw = self._call(role, system, user, fmt)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError):
                continue
        return None


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of possibly-chatty model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return match.group(0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/llm.py tests/test_llm.py
git commit -m "feat: LLM wrapper with JSON-constrained retries and fake backend"
```

---

### Task 4: Task plugin interface and loader

**Files:**
- Create: `scientist_one/tasks/__init__.py`, `scientist_one/tasks/base.py`
- Test: `tests/test_tasks.py`

**Interfaces:**
- Produces: `TaskSpec` (pydantic: `name: str`, `description: str`, `metric_direction: Literal["higher", "lower"]`, `seed_queries: list[str] = []`, `timeout_s: int | None = None`, `path: Path`) with methods `starter_path() -> Path`, `evaluator_path() -> Path`, and `better(a: float, b: float) -> bool` (True if `a` beats `b` per metric direction); and `load_task(path: Path) -> TaskSpec` (raises `FileNotFoundError` if `task.yaml`, `starter.py`, or `evaluator.py` is missing).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tasks.py
import pytest
from scientist_one.tasks.base import TaskSpec, load_task


def make_task_dir(tmp_path, yaml_text):
    d = tmp_path / "demo"
    d.mkdir()
    (d / "task.yaml").write_text(yaml_text)
    (d / "starter.py").write_text("def pack(items, capacity): ...\n")
    (d / "evaluator.py").write_text("def evaluate(solution_path, workdir): ...\n")
    return d


def test_load_task(tmp_path):
    d = make_task_dir(tmp_path,
        "name: demo\ndescription: a demo\nmetric_direction: lower\n"
        "seed_queries: [bin packing]\n")
    task = load_task(d)
    assert task.name == "demo"
    assert task.metric_direction == "lower"
    assert task.starter_path().exists()
    assert task.better(1.0, 2.0) is True   # lower is better
    assert task.better(2.0, 1.0) is False


def test_missing_file_raises(tmp_path):
    d = make_task_dir(tmp_path, "name: x\ndescription: y\nmetric_direction: higher\n")
    (d / "evaluator.py").unlink()
    with pytest.raises(FileNotFoundError):
        load_task(d)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tasks.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/tasks/__init__.py
```

```python
# scientist_one/tasks/base.py
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class TaskSpec(BaseModel):
    name: str
    description: str
    metric_direction: Literal["higher", "lower"]
    seed_queries: list[str] = []
    timeout_s: int | None = None
    path: Path

    def starter_path(self) -> Path:
        return self.path / "starter.py"

    def evaluator_path(self) -> Path:
        return self.path / "evaluator.py"

    def better(self, a: float, b: float) -> bool:
        return a > b if self.metric_direction == "higher" else a < b


def load_task(path: Path) -> TaskSpec:
    path = Path(path)
    for required in ("task.yaml", "starter.py", "evaluator.py"):
        if not (path / required).exists():
            raise FileNotFoundError(f"task is missing {required}: {path / required}")
    data = yaml.safe_load((path / "task.yaml").read_text())
    return TaskSpec(path=path, **data)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tasks.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/tasks/ tests/test_tasks.py
git commit -m "feat: pluggable task interface and loader"
```

---

### Task 5: Bin-packing demo task

**Files:**
- Create: `tasks/bin_packing/task.yaml`, `tasks/bin_packing/starter.py`, `tasks/bin_packing/evaluator.py`, `tasks/bin_packing/data/instances.json`
- Test: `tests/test_bin_packing.py`

**Interfaces:**
- Consumes: `load_task` (Task 4).
- Produces: a loadable task directory at repo-root `tasks/bin_packing/`. The evaluator module exposes `evaluate(solution_path: str, workdir: str) -> dict` returning `{"score": float, "log": str}` where score = mean(bins_used / L1_lower_bound) over fixed instances (lower is better; 1.0 is optimal). Invalid packings (item lost, duplicated, or bin overflow) raise `ValueError` inside evaluate.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_bin_packing.py
import importlib.util
import json
from pathlib import Path

from scientist_one.tasks.base import load_task

TASK_DIR = Path("tasks/bin_packing")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluator", TASK_DIR / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_loads():
    task = load_task(TASK_DIR)
    assert task.metric_direction == "lower"


def test_starter_first_fit_scores(tmp_path):
    # starter.py ships a naive first-fit so the pipeline has a working baseline
    result = load_evaluator().evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    assert 1.0 <= result["score"] < 2.0
    assert "instances" in result["log"]


def test_invalid_packing_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def pack(items, capacity):\n    return [items]\n")  # overflows one bin
    import pytest
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_bin_packing.py -v`
Expected: FAIL (task.yaml missing → `FileNotFoundError`)

- [ ] **Step 3: Create the task files**

```yaml
# tasks/bin_packing/task.yaml
name: bin_packing
description: >
  Discover a heuristic for 1D bin packing. Implement pack(items, capacity)
  returning a list of bins (each a list of item sizes) using as few bins as
  possible. Score is mean(bins_used / L1 lower bound) over fixed instance
  sets; lower is better, 1.0 is optimal.
metric_direction: lower
seed_queries:
  - bin packing heuristic first fit decreasing
  - online bin packing approximation algorithm
```

```python
# tasks/bin_packing/starter.py
def pack(items: list[float], capacity: float) -> list[list[float]]:
    """Pack items into bins of the given capacity. Return a list of bins.

    Baseline: naive first-fit. Improve on this.
    """
    bins: list[list[float]] = []
    for item in items:
        for b in bins:
            if sum(b) + item <= capacity:
                b.append(item)
                break
        else:
            bins.append([item])
    return bins
```

```python
# tasks/bin_packing/evaluator.py
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pack


def evaluate(solution_path: str, workdir: str) -> dict:
    pack = _load_solution(solution_path)
    instances = json.loads(
        (Path(__file__).parent / "data" / "instances.json").read_text())
    ratios, lines = [], []
    for inst in instances:
        items, capacity = inst["items"], inst["capacity"]
        bins = pack(list(items), capacity)
        flat = [x for b in bins for x in b]
        if Counter(flat) != Counter(items):
            raise ValueError(f"{inst['name']}: packed items differ from input")
        for b in bins:
            if sum(b) > capacity + 1e-9:
                raise ValueError(f"{inst['name']}: bin overflow {sum(b)} > {capacity}")
        lower_bound = max(1, math.ceil(sum(items) / capacity))
        ratio = len(bins) / lower_bound
        ratios.append(ratio)
        lines.append(f"{inst['name']}: bins={len(bins)} lb={lower_bound} ratio={ratio:.4f}")
    score = sum(ratios) / len(ratios)
    lines.append(f"instances={len(instances)} mean_ratio={score:.4f}")
    return {"score": round(score, 4), "log": "\n".join(lines)}
```

- [ ] **Step 4: Generate the fixed instance data**

Run this once to create deterministic instances (seeded), then commit the JSON:

```bash
python - <<'EOF'
import json, random
random.seed(42)
instances = []
for i, (n, cap) in enumerate([(50, 1.0), (100, 1.0), (200, 1.0), (100, 10.0), (150, 5.0)]):
    items = [round(random.uniform(0.05, 0.7) * cap, 4) for _ in range(n)]
    instances.append({"name": f"inst_{i}", "capacity": cap, "items": items})
from pathlib import Path
Path("tasks/bin_packing/data").mkdir(parents=True, exist_ok=True)
Path("tasks/bin_packing/data/instances.json").write_text(json.dumps(instances))
print("wrote", len(instances), "instances")
EOF
```

Expected output: `wrote 5 instances`

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_bin_packing.py -v`
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add tasks/ tests/test_bin_packing.py
git commit -m "feat: bin-packing demo task with deterministic evaluator"
```

---

### Task 6: Sandboxed evaluation runner

**Files:**
- Create: `scientist_one/sandbox.py`
- Test: `tests/test_sandbox.py`

**Interfaces:**
- Consumes: `TaskSpec` (Task 4), `SolverConfig.timeout_s` (Task 1).
- Produces: `run_evaluation(task: TaskSpec, solution_path: Path, workdir: Path, timeout_s: int) -> EvalOutcome` where `EvalOutcome` is a pydantic model with `ok: bool`, `score: float` (0.0 on failure — callers treat failures as worst via `ok`, not the score value), `log: str`. Runs the task evaluator in a `python` subprocess (never in-process) with the timeout; crash/timeout/invalid-packing → `ok=False` with the traceback or "timeout" in `log`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sandbox.py
from pathlib import Path

from scientist_one.sandbox import run_evaluation
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))


def test_good_solution(tmp_path):
    out = run_evaluation(TASK, TASK.starter_path(), tmp_path, timeout_s=30)
    assert out.ok is True
    assert out.score >= 1.0


def test_crashing_solution(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("def pack(items, capacity):\n    raise RuntimeError('boom')\n")
    out = run_evaluation(TASK, bad, tmp_path, timeout_s=30)
    assert out.ok is False
    assert "boom" in out.log


def test_hanging_solution_times_out(tmp_path):
    slow = tmp_path / "slow.py"
    slow.write_text(
        "import time\ndef pack(items, capacity):\n    time.sleep(60)\n    return []\n")
    out = run_evaluation(TASK, slow, tmp_path, timeout_s=2)
    assert out.ok is False
    assert "timeout" in out.log.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sandbox.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/sandbox.py
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from pydantic import BaseModel

from .tasks.base import TaskSpec

_RUNNER = textwrap.dedent("""
    import importlib.util, json, sys
    spec = importlib.util.spec_from_file_location("evaluator", sys.argv[1])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    result = mod.evaluate(sys.argv[2], sys.argv[3])
    print("EVAL_RESULT:" + json.dumps(result))
""")


class EvalOutcome(BaseModel):
    ok: bool
    score: float
    log: str


def run_evaluation(task: TaskSpec, solution_path: Path, workdir: Path,
                   timeout_s: int) -> EvalOutcome:
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(task.evaluator_path()),
             str(solution_path), str(workdir)],
            capture_output=True, text=True, timeout=timeout_s, cwd=workdir,
        )
    except subprocess.TimeoutExpired:
        return EvalOutcome(ok=False, score=0.0, log=f"timeout after {timeout_s}s")
    for line in proc.stdout.splitlines():
        if line.startswith("EVAL_RESULT:"):
            result = json.loads(line[len("EVAL_RESULT:"):])
            return EvalOutcome(ok=True, score=result["score"], log=result["log"])
    return EvalOutcome(ok=False, score=0.0,
                       log=(proc.stderr or proc.stdout or "no output").strip())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sandbox.py -v`
Expected: 3 PASSED (the timeout test takes ~2s)

- [ ] **Step 5: Commit**

```bash
git add scientist_one/sandbox.py tests/test_sandbox.py
git commit -m "feat: sandboxed subprocess evaluation with timeout and crash capture"
```

---

### Task 7: Ideator

**Files:**
- Create: `scientist_one/discovery/__init__.py`, `scientist_one/discovery/ideator.py`
- Test: `tests/test_ideator.py`

**Interfaces:**
- Consumes: `LLMClient` (Task 3), `TaskSpec` (Task 4), `EvidenceStore` (Task 2).
- Produces: `generate_ideas(llm: LLMClient, task: TaskSpec, brief_text: str, brief_ids: list[str], store: EvidenceStore) -> list[str]` — returns `idea` evidence-record IDs ranked best-first (novelty+feasibility scored by the judging model; unscored ideas keep generation order at the end). Each `idea` payload: `{"title", "approach", "rationale", "novelty", "feasibility"}`; `sources` = `brief_ids`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ideator.py
import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.ideator import generate_ideas
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

CONSERVATIVE = json.dumps({"ideas": [
    {"title": "FFD", "approach": "sort desc then first-fit", "rationale": "classic"}]})
UNCONVENTIONAL = json.dumps({"ideas": [
    {"title": "Sim-anneal", "approach": "local search over packings", "rationale": "novel"}]})
SCORES = json.dumps({"scores": [
    {"index": 0, "novelty": 2, "feasibility": 5},
    {"index": 1, "novelty": 5, "feasibility": 3}]})


def test_generates_and_ranks(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    bid = store.append("brief-claim", "investigator", {"text": "FFD is strong"})
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([CONSERVATIVE, UNCONVENTIONAL, SCORES]))
    ids = generate_ideas(llm, TASK, "brief text", [bid], store)
    assert len(ids) == 2
    ideas = [store.get(i) for i in ids]
    # ranked by novelty+feasibility: sim-anneal (8) > FFD (7)
    assert ideas[0].payload["title"] == "Sim-anneal"
    assert ideas[0].sources == [bid]


def test_scoring_failure_keeps_generation_order(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(
        [CONSERVATIVE, UNCONVENTIONAL, "junk", "junk", "junk"]))
    ids = generate_ideas(llm, TASK, "brief", [], store)
    assert [store.get(i).payload["title"] for i in ids] == ["FFD", "Sim-anneal"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ideator.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/discovery/__init__.py
```

```python
# scientist_one/discovery/ideator.py
from pydantic import BaseModel

from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec


class Idea(BaseModel):
    title: str
    approach: str
    rationale: str


class IdeaList(BaseModel):
    ideas: list[Idea]


class IdeaScore(BaseModel):
    index: int
    novelty: int
    feasibility: int


class IdeaScores(BaseModel):
    scores: list[IdeaScore]


_SYSTEM = ("You are a research ideator. Ground every idea in the research brief. "
           "Return JSON matching the requested schema.")


def _generate(llm: LLMClient, task: TaskSpec, brief_text: str, mode: str) -> list[Idea]:
    prompt = (
        f"Task: {task.description}\n\nResearch brief:\n{brief_text}\n\n"
        f"Propose 2 {mode} algorithmic approaches. "
        '{"ideas": [{"title", "approach", "rationale"}]}'
    )
    result = llm.chat_json("reasoning", _SYSTEM, prompt, IdeaList)
    return result.ideas if result else []


def generate_ideas(llm: LLMClient, task: TaskSpec, brief_text: str,
                   brief_ids: list[str], store: EvidenceStore) -> list[str]:
    ideas = (_generate(llm, task, brief_text, "conservative, well-established")
             + _generate(llm, task, brief_text, "unconventional, creative"))
    listing = "\n".join(f"{i}: {x.title} — {x.approach}" for i, x in enumerate(ideas))
    scored = llm.chat_json(
        "judging", _SYSTEM,
        f"Score each idea 1-5 for novelty and feasibility for this task: "
        f"{task.description}\n\n{listing}\n\n"
        '{"scores": [{"index", "novelty", "feasibility"}]}',
        IdeaScores,
    )
    ranks = {s.index: s for s in scored.scores} if scored else {}
    order = sorted(
        range(len(ideas)),
        key=lambda i: -(ranks[i].novelty + ranks[i].feasibility) if i in ranks else 0,
    )
    ids = []
    for i in order:
        idea, s = ideas[i], ranks.get(i)
        ids.append(store.append("idea", "discovery", {
            "title": idea.title, "approach": idea.approach, "rationale": idea.rationale,
            "novelty": s.novelty if s else None,
            "feasibility": s.feasibility if s else None,
        }, sources=brief_ids))
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ideator.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/discovery/ tests/test_ideator.py
git commit -m "feat: ideator with conservative+unconventional generation and ranking"
```

---

### Task 8: Solver and spec-violation Auditor

**Files:**
- Create: `scientist_one/discovery/solver.py`, `scientist_one/discovery/auditor.py`
- Test: `tests/test_solver_auditor.py`

**Interfaces:**
- Consumes: `LLMClient` (Task 3), `TaskSpec` (Task 4).
- Produces: `solve(llm: LLMClient, task: TaskSpec, idea: dict, feedback: str | None) -> str | None` (returns Python source extracted from the reply's ```python fence, or the whole reply if it already compiles; `None` if nothing compiles — the branch fails explicitly). `audit_solution(llm: LLMClient, task: TaskSpec, code: str) -> AuditVerdict` where `AuditVerdict` is pydantic `{flagged: bool, reason: str}`; judge failure → `flagged=False, reason="audit unavailable"`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_solver_auditor.py
import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.auditor import audit_solution
from scientist_one.discovery.solver import solve
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))
IDEA = {"title": "FFD", "approach": "sort desc", "rationale": "classic"}
CODE = "def pack(items, capacity):\n    return [[i] for i in items]\n"


def make_llm(tmp_path, responses):
    return LLMClient(Config(), tmp_path, backend=FakeBackend(responses))


def test_solve_extracts_fenced_code(tmp_path):
    llm = make_llm(tmp_path, [f"Here you go:\n```python\n{CODE}```\ndone"])
    assert solve(llm, TASK, IDEA, None) == CODE


def test_solve_returns_none_on_garbage(tmp_path):
    llm = make_llm(tmp_path, ["I cannot write code today ((("])
    assert solve(llm, TASK, IDEA, feedback="prior score 1.4") is None


def test_audit_flagged(tmp_path):
    llm = make_llm(tmp_path, [json.dumps({"flagged": True, "reason": "hardcodes answers"})])
    v = audit_solution(llm, TASK, CODE)
    assert v.flagged and "hardcodes" in v.reason


def test_audit_degrades_unflagged(tmp_path):
    llm = make_llm(tmp_path, ["junk", "junk", "junk"])
    v = audit_solution(llm, TASK, CODE)
    assert v.flagged is False
    assert v.reason == "audit unavailable"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_solver_auditor.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/discovery/solver.py
import ast
import re

from ..llm import LLMClient
from ..tasks.base import TaskSpec

_SYSTEM = ("You are an expert algorithm implementer. Reply with a single "
           "complete Python solution in one ```python code fence. The code "
           "must define exactly the interface in the starter template, use "
           "only the standard library, and be deterministic.")


def _extract_code(reply: str) -> str | None:
    match = re.search(r"```(?:python)?\n(.*?)```", reply, re.DOTALL)
    candidate = match.group(1) if match else reply
    try:
        ast.parse(candidate)
    except SyntaxError:
        return None
    return candidate


def solve(llm: LLMClient, task: TaskSpec, idea: dict, feedback: str | None) -> str | None:
    starter = task.starter_path().read_text()
    prompt = (
        f"Task: {task.description}\n\nStarter template:\n```python\n{starter}```\n\n"
        f"Approach to implement: {idea['title']} — {idea['approach']}\n"
        f"Rationale: {idea['rationale']}\n"
    )
    if feedback:
        prompt += f"\nFeedback from previous attempt:\n{feedback}\n\nImprove on it."
    return _extract_code(llm.chat("reasoning", _SYSTEM, prompt))
```

```python
# scientist_one/discovery/auditor.py
from pydantic import BaseModel

from ..llm import LLMClient
from ..tasks.base import TaskSpec


class AuditVerdict(BaseModel):
    flagged: bool
    reason: str


_SYSTEM = ("You are a research-integrity auditor. Flag solutions that violate "
           "the task specification: hardcoded answers for known inputs, "
           "reading evaluator internals, gaming the metric instead of solving "
           "the problem. Legitimate heuristics are not violations. Return JSON.")


def audit_solution(llm: LLMClient, task: TaskSpec, code: str) -> AuditVerdict:
    verdict = llm.chat_json(
        "judging", _SYSTEM,
        f"Task spec: {task.description}\n\nSolution code:\n```python\n{code}```\n\n"
        'Return {"flagged": bool, "reason": str}.',
        AuditVerdict,
    )
    return verdict or AuditVerdict(flagged=False, reason="audit unavailable")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_solver_auditor.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/discovery/solver.py scientist_one/discovery/auditor.py tests/test_solver_auditor.py
git commit -m "feat: solver with code extraction and spec-violation auditor"
```

---

### Task 9: PEE orchestrator with ablations

**Files:**
- Create: `scientist_one/discovery/pee.py`
- Test: `tests/test_pee.py`

**Interfaces:**
- Consumes: `generate_ideas` (Task 7), `solve`, `_extract_code`, `audit_solution` (Task 8), `run_evaluation` (Task 6), `EvidenceStore` (Task 2), `Config` (Task 1), `TaskSpec` (Task 4).
- Produces: `run_discovery(llm: LLMClient, config: Config, task: TaskSpec, store: EvidenceStore, run_dir: Path, brief_text: str, brief_ids: list[str]) -> DiscoveryResult | None` where `DiscoveryResult` is pydantic `{best_solution_path: str, best_solution_id: str, best_eval_id: str, best_score: float, ablation_ids: list[str]}`. Returns `None` when no branch produced a valid, unflagged, ok-evaluated solution. Solutions written to `run_dir/solutions/b{branch}_i{iter}.py`. Evidence records: `solution` (sources=[idea_id]), `eval-result` (sources=[solution_id]), `audit-flag` (sources=[solution_id]), `ablation` (sources=[best_solution_id]).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pee.py
import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import run_discovery
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

FFD_CODE = ("def pack(items, capacity):\n"
            "    bins = []\n"
            "    for item in sorted(items, reverse=True):\n"
            "        for b in bins:\n"
            "            if sum(b) + item <= capacity:\n"
            "                b.append(item)\n"
            "                break\n"
            "        else:\n"
            "            bins.append([item])\n"
            "    return bins\n")
FF_NOSORT = FFD_CODE.replace("sorted(items, reverse=True)", "items")

IDEAS_A = json.dumps({"ideas": [{"title": "FFD", "approach": "sort desc + first fit",
                                 "rationale": "classic"}]})
IDEAS_B = json.dumps({"ideas": []})
SCORES = json.dumps({"scores": [{"index": 0, "novelty": 3, "feasibility": 5}]})
NOT_FLAGGED = json.dumps({"flagged": False, "reason": "legitimate heuristic"})
COMPONENTS = json.dumps({"components": ["descending sort"]})


def small_config():
    return Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                  solver={"timeout_s": 30})


def test_full_discovery_run(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [
        IDEAS_A, IDEAS_B, SCORES,                      # ideator
        f"```python\n{FFD_CODE}```",                   # solve b0 i0
        NOT_FLAGGED,                                   # audit b0 i0
        COMPONENTS,                                    # ablation component list
        f"```python\n{FF_NOSORT}```",                  # ablation variant code
    ]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    result = run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", [])
    assert result is not None
    assert Path(result.best_solution_path).read_text() == FFD_CODE
    assert store.get(result.best_eval_id).payload["ok"] is True
    assert len(result.ablation_ids) == 1
    abl = store.get(result.ablation_ids[0])
    assert abl.payload["component"] == "descending sort"
    assert abl.sources == [result.best_solution_id]


def test_no_valid_solution_returns_none(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [IDEAS_A, IDEAS_B, SCORES, "no code here ((("]  # solver garbage
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    assert run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", []) is None


def test_flagged_solution_excluded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    responses = [IDEAS_A, IDEAS_B, SCORES,
                 f"```python\n{FFD_CODE}```",
                 json.dumps({"flagged": True, "reason": "gaming"})]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    assert run_discovery(llm, small_config(), TASK, store, tmp_path, "brief", []) is None
    assert len(store.by_type("audit-flag")) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pee.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/discovery/pee.py
from collections import deque
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..sandbox import run_evaluation
from ..tasks.base import TaskSpec
from .auditor import audit_solution
from .ideator import generate_ideas
from .solver import _extract_code, solve


class _Branch(BaseModel):
    idea_id: str
    feedback: str = ""
    best_score: float | None = None
    best_eval_id: str | None = None
    best_solution_id: str | None = None
    best_solution_path: str | None = None


class DiscoveryResult(BaseModel):
    best_solution_path: str
    best_solution_id: str
    best_eval_id: str
    best_score: float
    ablation_ids: list[str]


class ComponentList(BaseModel):
    components: list[str]


def run_discovery(llm: LLMClient, config: Config, task: TaskSpec,
                  store: EvidenceStore, run_dir: Path, brief_text: str,
                  brief_ids: list[str]) -> DiscoveryResult | None:
    run_dir = Path(run_dir)
    (run_dir / "solutions").mkdir(parents=True, exist_ok=True)
    timeout = task.timeout_s or config.solver.timeout_s

    idea_queue = deque(generate_ideas(llm, task, brief_text, brief_ids, store))
    if not idea_queue:
        return None
    branches = [_Branch(idea_id=idea_queue.popleft())
                for _ in range(min(config.discovery.branches, len(idea_queue) + 1))]

    for it in range(config.discovery.iterations):
        for bi, br in enumerate(branches):
            idea = store.get(br.idea_id).payload
            code = solve(llm, task, idea, br.feedback or None)
            if code is None:
                br.feedback += "\nprevious reply contained no valid Python code"
                continue
            path = run_dir / "solutions" / f"b{bi}_i{it}.py"
            path.write_text(code)
            sol_id = store.append("solution", "discovery",
                                  {"path": str(path), "iteration": it},
                                  sources=[br.idea_id])
            outcome = run_evaluation(task, path, run_dir / "eval_work", timeout)
            eval_id = store.append("eval-result", "discovery",
                                   {"ok": outcome.ok, "score": outcome.score,
                                    "log": outcome.log}, sources=[sol_id])
            verdict = audit_solution(llm, task, code)
            if verdict.flagged:
                store.append("audit-flag", "discovery",
                             {"reason": verdict.reason}, sources=[sol_id])
            br.feedback = f"score={outcome.score} ok={outcome.ok}\n{outcome.log[-1000:]}"
            if outcome.ok and not verdict.flagged and (
                    br.best_score is None or task.better(outcome.score, br.best_score)):
                br.best_score, br.best_eval_id = outcome.score, eval_id
                br.best_solution_id, br.best_solution_path = sol_id, str(path)
        if it < config.discovery.iterations - 1 and idea_queue:
            scored = [b for b in branches if b.best_score is not None]
            scored.sort(key=lambda b: b.best_score,
                        reverse=(task.metric_direction == "higher"))
            survivors = scored[:config.discovery.survivors]
            refill = [b for b in branches if b not in survivors]
            for i, b in enumerate(refill):
                if idea_queue:
                    refill[i] = _Branch(idea_id=idea_queue.popleft())
            branches = survivors + refill

    best: _Branch | None = None
    for br in branches:
        if br.best_score is not None and (
                best is None or task.better(br.best_score, best.best_score)):
            best = br
    if best is None:
        return None

    ablation_ids = _run_ablations(llm, config, task, store, run_dir, best, timeout)
    return DiscoveryResult(
        best_solution_path=best.best_solution_path,
        best_solution_id=best.best_solution_id,
        best_eval_id=best.best_eval_id,
        best_score=best.best_score,
        ablation_ids=ablation_ids,
    )


def _run_ablations(llm: LLMClient, config: Config, task: TaskSpec,
                   store: EvidenceStore, run_dir: Path, best: _Branch,
                   timeout: int) -> list[str]:
    code = Path(best.best_solution_path).read_text()
    comp = llm.chat_json(
        "judging",
        "You analyze algorithms. Return JSON.",
        f"List up to 3 named components of this solution that could be "
        f"individually disabled to measure their contribution.\n"
        f"```python\n{code}```\n"
        'Return {"components": [str]}.',
        ComponentList,
    )
    ids: list[str] = []
    for i, component in enumerate((comp.components if comp else [])[:3]):
        reply = llm.chat(
            "reasoning",
            "You are an expert algorithm implementer. Reply with a single "
            "complete Python solution in one ```python code fence.",
            f"Rewrite this solution with the component '{component}' disabled "
            f"or replaced by the trivial alternative, changing nothing else:\n"
            f"```python\n{code}```",
        )
        variant = _extract_code(reply)
        if variant is None:
            continue
        vpath = run_dir / "solutions" / f"ablation_{i}.py"
        vpath.write_text(variant)
        outcome = run_evaluation(task, vpath, run_dir / "eval_work", timeout)
        ids.append(store.append("ablation", "discovery", {
            "component": component, "ok": outcome.ok, "score": outcome.score,
            "baseline_score": best.best_score,
        }, sources=[best.best_solution_id]))
    return ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pee.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/discovery/pee.py tests/test_pee.py
git commit -m "feat: parallel explore-exploit orchestrator with pruning and ablations"
```

---

### Task 10: Scholarly API clients

**Files:**
- Create: `scientist_one/investigator/__init__.py`, `scientist_one/investigator/scholarly.py`
- Test: `tests/test_scholarly.py`

**Interfaces:**
- Produces: `PaperMeta` (pydantic: `title: str`, `authors: list[str]`, `year: int | None`, `abstract: str`, `url: str`, `source: str`, `external_id: str`) and `search_papers(client: httpx.Client, queries: list[str], limit_per_query: int) -> list[PaperMeta]` — queries Semantic Scholar then arXiv per query, deduplicates by lowercase title, swallows per-request network/HTTP errors (a failed query contributes zero papers, never raises). Also exposes `search_semantic_scholar(client, query, limit)` and `search_arxiv(client, query, limit)` (used individually by the audit's reference verification in Task 16).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scholarly.py
import httpx

from scientist_one.investigator.scholarly import search_papers

S2_BODY = {"data": [{
    "title": "First Fit Decreasing", "abstract": "We analyze FFD.",
    "year": 1974, "url": "https://s2/ffd", "externalIds": {"DOI": "10.1/ffd"},
    "authors": [{"name": "D. Johnson"}],
}]}

ARXIV_BODY = """<?xml version="1.0"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2001.00001v1</id>
    <title>Online Bin Packing Revisited</title>
    <summary>A new online heuristic.</summary>
    <published>2020-01-01T00:00:00Z</published>
    <author><name>A. Author</name></author>
  </entry>
</feed>"""


def mock_client(s2_status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if "semanticscholar" in request.url.host:
            return httpx.Response(s2_status, json=S2_BODY)
        return httpx.Response(200, text=ARXIV_BODY)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_papers_merges_and_dedupes():
    papers = search_papers(mock_client(), ["bin packing"], limit_per_query=5)
    titles = {p.title for p in papers}
    assert "First Fit Decreasing" in titles
    assert "Online Bin Packing Revisited" in titles
    sources = {p.source for p in papers}
    assert sources == {"semantic_scholar", "arxiv"}


def test_api_failure_degrades():
    papers = search_papers(mock_client(s2_status=500), ["bin packing"], 5)
    assert [p.source for p in papers] == ["arxiv"]  # S2 failed, arXiv survived
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scholarly.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/investigator/__init__.py
```

```python
# scientist_one/investigator/scholarly.py
import xml.etree.ElementTree as ET

import httpx
from pydantic import BaseModel

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
ARXIV_URL = "https://export.arxiv.org/api/query"
_ATOM = "{http://www.w3.org/2005/Atom}"


class PaperMeta(BaseModel):
    title: str
    authors: list[str]
    year: int | None
    abstract: str
    url: str
    source: str
    external_id: str


def search_semantic_scholar(client: httpx.Client, query: str,
                            limit: int) -> list[PaperMeta]:
    resp = client.get(S2_URL, params={
        "query": query, "limit": limit,
        "fields": "title,abstract,authors,year,url,externalIds"})
    resp.raise_for_status()
    papers = []
    for item in resp.json().get("data", []):
        ext = item.get("externalIds") or {}
        papers.append(PaperMeta(
            title=item.get("title") or "",
            authors=[a["name"] for a in item.get("authors") or []],
            year=item.get("year"),
            abstract=item.get("abstract") or "",
            url=item.get("url") or "",
            source="semantic_scholar",
            external_id=ext.get("DOI") or ext.get("ArXiv") or "",
        ))
    return papers


def search_arxiv(client: httpx.Client, query: str, limit: int) -> list[PaperMeta]:
    resp = client.get(ARXIV_URL, params={
        "search_query": f"all:{query}", "max_results": limit})
    resp.raise_for_status()
    papers = []
    for entry in ET.fromstring(resp.text).findall(f"{_ATOM}entry"):
        published = entry.findtext(f"{_ATOM}published") or ""
        papers.append(PaperMeta(
            title=" ".join((entry.findtext(f"{_ATOM}title") or "").split()),
            authors=[(a.findtext(f"{_ATOM}name") or "")
                     for a in entry.findall(f"{_ATOM}author")],
            year=int(published[:4]) if published[:4].isdigit() else None,
            abstract=" ".join((entry.findtext(f"{_ATOM}summary") or "").split()),
            url=entry.findtext(f"{_ATOM}id") or "",
            source="arxiv",
            external_id=(entry.findtext(f"{_ATOM}id") or "").rsplit("/", 1)[-1],
        ))
    return papers


def search_papers(client: httpx.Client, queries: list[str],
                  limit_per_query: int) -> list[PaperMeta]:
    papers: list[PaperMeta] = []
    for query in queries:
        for fn in (search_semantic_scholar, search_arxiv):
            try:
                papers.extend(fn(client, query, limit_per_query))
            except (httpx.HTTPError, ET.ParseError):
                continue
    seen: set[str] = set()
    unique = []
    for p in papers:
        key = p.title.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(p)
    return unique
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scholarly.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/investigator/ tests/test_scholarly.py
git commit -m "feat: Semantic Scholar and arXiv clients with graceful degradation"
```

---

### Task 11: Problem Investigator (relevance filter + research brief)

**Files:**
- Create: `scientist_one/investigator/run.py`
- Test: `tests/test_investigator.py`

**Interfaces:**
- Consumes: `search_papers`, `PaperMeta` (Task 10), `LLMClient` (Task 3), `EvidenceStore` (Task 2), `Config` (Task 1), `TaskSpec` (Task 4).
- Produces: `run_investigator(llm: LLMClient, config: Config, task: TaskSpec, store: EvidenceStore, run_dir: Path, http_client: httpx.Client | None = None) -> InvestigatorResult` where `InvestigatorResult` is pydantic `{brief_text: str, brief_ids: list[str], references_path: str}`. Writes `run_dir/references.json` (list of `PaperMeta` dumps — API-sourced only) and `run_dir/brief.md`. Evidence: one `paper` record per kept paper; one `brief-claim` record per brief claim with `sources` = the paper records it cites (may be empty). Relevance-scoring failure keeps the first `max_papers` papers; zero papers still produces a brief (claims unsourced, noted in brief).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_investigator.py
import json
from pathlib import Path

import httpx

from scientist_one.config import Config
from scientist_one.evidence import EvidenceStore
from scientist_one.investigator.run import run_investigator
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task

TASK = load_task(Path("tasks/bin_packing"))

S2_BODY = {"data": [
    {"title": "FFD Analysis", "abstract": "FFD uses at most 11/9 OPT bins.",
     "year": 1974, "url": "https://s2/ffd", "externalIds": {"DOI": "10.1/ffd"},
     "authors": [{"name": "D. Johnson"}]},
    {"title": "Irrelevant Paper", "abstract": "About databases.", "year": 2000,
     "url": "https://s2/x", "externalIds": {}, "authors": []},
]}

RELEVANCE = json.dumps({"scores": [{"index": 0, "relevance": 5},
                                   {"index": 1, "relevance": 1}]})
BRIEF = json.dumps({
    "framing": "1D bin packing seeks minimal bins.",
    "claims": [{"text": "FFD achieves 11/9 OPT.", "paper_indexes": [0]}],
    "baselines": "First-fit ratio around 1.7.",
})


def mock_client():
    def handler(request):
        if "semanticscholar" in request.url.host:
            return httpx.Response(200, json=S2_BODY)
        return httpx.Response(200, text='<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_investigator_produces_grounded_brief(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    cfg = Config(investigator={"max_papers": 1})
    llm = LLMClient(cfg, tmp_path, backend=FakeBackend([RELEVANCE, BRIEF]))
    result = run_investigator(llm, cfg, TASK, store, tmp_path,
                              http_client=mock_client())
    refs = json.loads(Path(result.references_path).read_text())
    assert [r["title"] for r in refs] == ["FFD Analysis"]  # top-1 by relevance
    assert "FFD achieves 11/9 OPT." in result.brief_text
    claim = store.get(result.brief_ids[0])
    assert claim.type == "brief-claim"
    paper_rec = store.get(claim.sources[0])
    assert paper_rec.payload["title"] == "FFD Analysis"


def test_no_papers_still_briefs(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    cfg = Config()
    empty = httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))
    llm = LLMClient(cfg, tmp_path, backend=FakeBackend([json.dumps(
        {"framing": "f", "claims": [{"text": "c", "paper_indexes": []}],
         "baselines": "b"})]))
    result = run_investigator(llm, cfg, TASK, store, tmp_path, http_client=empty)
    assert json.loads(Path(result.references_path).read_text()) == []
    assert store.get(result.brief_ids[0]).sources == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_investigator.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/investigator/run.py
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .scholarly import PaperMeta, search_papers


class InvestigatorResult(BaseModel):
    brief_text: str
    brief_ids: list[str]
    references_path: str


class RelevanceScore(BaseModel):
    index: int
    relevance: int


class RelevanceScores(BaseModel):
    scores: list[RelevanceScore]


class BriefClaim(BaseModel):
    text: str
    paper_indexes: list[int]


class Brief(BaseModel):
    framing: str
    claims: list[BriefClaim]
    baselines: str


def _rank_papers(llm: LLMClient, task: TaskSpec, papers: list[PaperMeta],
                 max_papers: int) -> list[PaperMeta]:
    if len(papers) <= max_papers:
        return papers
    listing = "\n".join(f"{i}: {p.title} — {p.abstract[:200]}"
                        for i, p in enumerate(papers))
    scored = llm.chat_json(
        "judging", "You rate paper relevance 1-5. Return JSON.",
        f"Task: {task.description}\n\nPapers:\n{listing}\n\n"
        'Return {"scores": [{"index", "relevance"}]}.',
        RelevanceScores,
    )
    if scored is None:
        return papers[:max_papers]
    ranks = {s.index: s.relevance for s in scored.scores}
    order = sorted(range(len(papers)), key=lambda i: -ranks.get(i, 0))
    return [papers[i] for i in order[:max_papers]]


def run_investigator(llm: LLMClient, config: Config, task: TaskSpec,
                     store: EvidenceStore, run_dir: Path,
                     http_client: httpx.Client | None = None) -> InvestigatorResult:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    client = http_client or httpx.Client(timeout=30)
    papers = search_papers(client, task.seed_queries or [task.description],
                           limit_per_query=config.investigator.max_papers)
    papers = _rank_papers(llm, task, papers, config.investigator.max_papers)

    paper_ids = [store.append("paper", "investigator", p.model_dump())
                 for p in papers]
    references_path = run_dir / "references.json"
    references_path.write_text(json.dumps([p.model_dump() for p in papers], indent=2))

    listing = "\n".join(f"{i}: {p.title} ({p.year}) — {p.abstract[:300]}"
                        for i, p in enumerate(papers)) or "(no papers retrieved)"
    brief = llm.chat_json(
        "reasoning",
        "You write research briefs grounded ONLY in the provided papers. "
        "Every claim must cite paper indexes; use an empty list only when no "
        "paper supports it. Return JSON.",
        f"Task: {task.description}\n\nRetrieved papers:\n{listing}\n\n"
        'Return {"framing": str, "claims": [{"text", "paper_indexes"}], '
        '"baselines": str}.',
        Brief,
    ) or Brief(framing=task.description,
               claims=[BriefClaim(text="No literature grounding available.",
                                  paper_indexes=[])],
               baselines="unknown")

    brief_ids = []
    lines = [f"# Research brief: {task.name}", "", brief.framing, "", "## Findings"]
    for claim in brief.claims:
        sources = [paper_ids[i] for i in claim.paper_indexes if i < len(paper_ids)]
        cid = store.append("brief-claim", "investigator",
                           {"text": claim.text}, sources=sources)
        brief_ids.append(cid)
        lines.append(f"- {claim.text} {{ev:{cid}}}")
    lines += ["", "## Baselines", brief.baselines]
    brief_text = "\n".join(lines)
    (run_dir / "brief.md").write_text(brief_text)
    return InvestigatorResult(brief_text=brief_text, brief_ids=brief_ids,
                              references_path=str(references_path))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_investigator.py -v`
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/investigator/run.py tests/test_investigator.py
git commit -m "feat: problem investigator with relevance ranking and grounded brief"
```

---

### Task 12: Writer — Conceive and Ground

**Files:**
- Create: `scientist_one/writer/__init__.py`, `scientist_one/writer/conceive.py`, `scientist_one/writer/ground.py`
- Test: `tests/test_conceive_ground.py`

**Interfaces:**
- Consumes: `LLMClient` (Task 3), `EvidenceStore` (Task 2), `DiscoveryResult` (Task 9), `TaskSpec` (Task 4).
- Produces: `conceive(llm: LLMClient, task: TaskSpec, store: EvidenceStore, brief_text: str, discovery: DiscoveryResult) -> str` — a markdown narrative where factual sentences end with `{ev:ev_XXXX}` tags; appends one `draft-claim` record per tagged sentence (sources = the tagged, existing evidence IDs). And `ground_check(narrative: str, store: EvidenceStore, tolerance: float) -> list[GroundIssue]` where `GroundIssue` is pydantic `{kind: str, detail: str}` with kinds `unknown-tag`, `untagged-numeric`, `number-mismatch` — fully deterministic, no LLM. Shared helpers in `ground.py`: `TAG_RE` (regex for `{ev:...}` tags), `sentences(text: str) -> list[str]`, `numbers_in_text(s: str) -> list[float]`, `numbers_in_payload(obj) -> list[float]` (recursive, includes numbers inside strings such as eval logs).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_conceive_ground.py
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import DiscoveryResult
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.conceive import conceive
from scientist_one.writer.ground import ground_check

TASK = load_task(Path("tasks/bin_packing"))


def seeded_store(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    return store, sol, ev


def test_conceive_registers_draft_claims(tmp_path):
    store, sol, ev = seeded_store(tmp_path)
    discovery = DiscoveryResult(best_solution_path="s.py", best_solution_id=sol,
                                best_eval_id=ev, best_score=1.08, ablation_ids=[])
    narrative = f"## Results\nFFD reaches a ratio of 1.08. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([narrative]))
    out = conceive(llm, TASK, store, "brief", discovery)
    assert f"{{ev:{ev}}}" in out
    claims = store.by_type("draft-claim")
    assert len(claims) == 1
    assert claims[0].sources == [ev]


def test_ground_passes_clean_narrative(tmp_path):
    store, _, ev = seeded_store(tmp_path)
    narrative = f"FFD reaches a ratio of 1.08. {{ev:{ev}}}"
    assert ground_check(narrative, store, 0.01) == []


def test_ground_flags_unknown_tag_and_mismatch(tmp_path):
    store, _, ev = seeded_store(tmp_path)
    narrative = (f"Score is 3.99. {{ev:{ev}}}\n"      # number not in evidence
                 "Ghost claim. {ev:ev_9999}\n"        # unknown tag
                 "We used 42 instances here.\n")      # numeric, untagged
    kinds = sorted(i.kind for i in ground_check(narrative, store, 0.01))
    assert kinds == ["number-mismatch", "unknown-tag", "untagged-numeric"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_conceive_ground.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/writer/__init__.py
```

```python
# scientist_one/writer/ground.py
import math
import re

from pydantic import BaseModel

from ..evidence import EvidenceStore

TAG_RE = re.compile(r"\{ev:(ev_\d+)\}")
_NUM_RE = re.compile(r"\d+\.\d+|\d{2,}")


class GroundIssue(BaseModel):
    kind: str
    detail: str


def sentences(text: str) -> list[str]:
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.extend(s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip())
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
```

```python
# scientist_one/writer/conceive.py
from pathlib import Path

from ..discovery.pee import DiscoveryResult
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .ground import TAG_RE, sentences

_SYSTEM = (
    "You write research narratives in markdown. HARD RULE: every factual "
    "sentence must end with an evidence tag like {ev:ev_0042} naming one of "
    "the evidence records provided. Never invent numbers or tags; only state "
    "what the evidence supports."
)


def conceive(llm: LLMClient, task: TaskSpec, store: EvidenceStore,
             brief_text: str, discovery: DiscoveryResult) -> str:
    eval_rec = store.get(discovery.best_eval_id)
    ablations = [store.get(a) for a in discovery.ablation_ids]
    ablation_text = "\n".join(
        f"- {a.id}: component '{a.payload['component']}' disabled -> "
        f"score {a.payload['score']} (baseline {a.payload['baseline_score']})"
        for a in ablations) or "(none)"
    code = Path(discovery.best_solution_path).read_text()
    prompt = (
        f"Task: {task.description}\n\n"
        f"Research brief (with its evidence tags):\n{brief_text}\n\n"
        f"Best solution evidence record {discovery.best_eval_id}: "
        f"score={eval_rec.payload['score']}\nEvaluator log:\n"
        f"{eval_rec.payload['log']}\n\nAblations:\n{ablation_text}\n\n"
        f"Solution code:\n```python\n{code}```\n\n"
        "Write the research narrative: problem, method, results, ablation "
        "analysis. Tag every factual sentence."
    )
    narrative = llm.chat("reasoning", _SYSTEM, prompt)
    known = {r.id for r in store.all()}
    for sent in sentences(narrative):
        tags = [t for t in TAG_RE.findall(sent) if t in known]
        if tags:
            store.append("draft-claim", "writer",
                         {"text": TAG_RE.sub("", sent).strip()}, sources=tags)
    return narrative
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_conceive_ground.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/writer/ tests/test_conceive_ground.py
git commit -m "feat: conceive with evidence tags and deterministic ground checks"
```

---

### Task 13: Writer — Critic, Resolve, and the writing loop

**Files:**
- Create: `scientist_one/writer/critic.py`, `scientist_one/writer/resolve.py`, `scientist_one/writer/run.py`
- Test: `tests/test_writer_loop.py`

**Interfaces:**
- Consumes: `conceive`, `ground_check` (Task 12), `LLMClient` (Task 3), `Config` (Task 1), `DiscoveryResult` (Task 9).
- Produces: `critic_check(llm: LLMClient, narrative: str) -> list[str]` (judge failure → `[]`, i.e. critic skipped, never blocks); `resolve(llm: LLMClient, narrative: str, issues: list[str]) -> str` (rewritten narrative; unparseable/empty reply → original narrative unchanged); `run_writer(llm: LLMClient, config: Config, task: TaskSpec, store: EvidenceStore, brief_text: str, discovery: DiscoveryResult) -> WriterResult` where `WriterResult` is pydantic `{narrative: str, remaining_issues: list[str]}` — loops Ground+Critic → Resolve up to `config.writer.max_rounds`, stops early when both are clean.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_writer_loop.py
import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.discovery.pee import DiscoveryResult
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.tasks.base import load_task
from scientist_one.writer.critic import critic_check
from scientist_one.writer.resolve import resolve
from scientist_one.writer.run import run_writer

TASK = load_task(Path("tasks/bin_packing"))
NO_ISSUES = json.dumps({"issues": []})


def seeded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    sol_file = tmp_path / "s.py"
    sol_file.write_text("def pack(items, capacity): return [[i] for i in items]\n")
    disc = DiscoveryResult(best_solution_path=str(sol_file), best_solution_id=sol,
                           best_eval_id=ev, best_score=1.08, ablation_ids=[])
    return store, ev, disc


def test_critic_parses_and_degrades(tmp_path):
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([json.dumps({"issues": ["overclaim"]})]))
    assert critic_check(llm, "text") == ["overclaim"]
    llm2 = LLMClient(Config(), tmp_path, backend=FakeBackend(["x", "x", "x"]))
    assert critic_check(llm2, "text") == []


def test_resolve_keeps_original_on_empty_reply(tmp_path):
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(["   "]))
    assert resolve(llm, "original", ["issue"]) == "original"


def test_loop_stops_when_clean(tmp_path):
    store, ev, disc = seeded(tmp_path)
    clean = f"The ratio is 1.08. {{ev:{ev}}}"
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([clean, NO_ISSUES]))  # conceive, critic
    result = run_writer(llm, Config(), TASK, store, "brief", disc)
    assert result.narrative == clean
    assert result.remaining_issues == []


def test_loop_resolves_dirty_narrative(tmp_path):
    store, ev, disc = seeded(tmp_path)
    dirty = f"The ratio is 9.99. {{ev:{ev}}}"
    clean = f"The ratio is 1.08. {{ev:{ev}}}"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([
        dirty, NO_ISSUES,   # conceive + critic round 1 (ground finds mismatch)
        clean,              # resolve
        NO_ISSUES,          # critic round 2 (ground now clean)
    ]))
    result = run_writer(llm, Config(), TASK, store, "brief", disc)
    assert result.narrative == clean
    assert result.remaining_issues == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_writer_loop.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/writer/critic.py
from pydantic import BaseModel

from ..llm import LLMClient


class CriticIssues(BaseModel):
    issues: list[str]


def critic_check(llm: LLMClient, narrative: str) -> list[str]:
    result = llm.chat_json(
        "reasoning",
        "You are a rigorous research critic. Find what deterministic checks "
        "cannot: internal contradictions, overclaims, gap between evidence "
        "and conclusion, missing comparisons, unfair baselines. Return JSON.",
        f"Narrative:\n{narrative}\n\n"
        'Return {"issues": [str]} — empty list if the narrative is sound.',
        CriticIssues,
    )
    return result.issues if result else []
```

```python
# scientist_one/writer/resolve.py
from ..llm import LLMClient


def resolve(llm: LLMClient, narrative: str, issues: list[str]) -> str:
    issue_list = "\n".join(f"- {i}" for i in issues)
    reply = llm.chat(
        "reasoning",
        "You revise research narratives. Fix every listed issue by "
        "correcting, calibrating, or DELETING unsupported claims. Keep all "
        "valid {ev:...} evidence tags. Reply with the full revised markdown "
        "narrative only.",
        f"Narrative:\n{narrative}\n\nIssues to fix:\n{issue_list}",
    )
    return reply if reply.strip() else narrative
```

```python
# scientist_one/writer/run.py
from pydantic import BaseModel

from ..config import Config
from ..discovery.pee import DiscoveryResult
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .conceive import conceive
from .critic import critic_check
from .ground import ground_check
from .resolve import resolve


class WriterResult(BaseModel):
    narrative: str
    remaining_issues: list[str]


def run_writer(llm: LLMClient, config: Config, task: TaskSpec,
               store: EvidenceStore, brief_text: str,
               discovery: DiscoveryResult) -> WriterResult:
    narrative = conceive(llm, task, store, brief_text, discovery)
    issues: list[str] = []
    for round_no in range(config.writer.max_rounds):
        ground = [f"[{i.kind}] {i.detail}"
                  for i in ground_check(narrative, store,
                                        config.verifier.numeric_tolerance)]
        critic = critic_check(llm, narrative)
        issues = ground + critic
        if not issues:
            break
        narrative = resolve(llm, narrative, issues)
    else:
        issues = [f"[{i.kind}] {i.detail}"
                  for i in ground_check(narrative, store,
                                        config.verifier.numeric_tolerance)]
    return WriterResult(narrative=narrative, remaining_issues=issues)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_writer_loop.py -v`
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/writer/critic.py scientist_one/writer/resolve.py scientist_one/writer/run.py tests/test_writer_loop.py
git commit -m "feat: ground-critic-resolve writing loop with explicit degradation"
```

---

### Task 14: Compose and the Renderer interface

**Files:**
- Create: `scientist_one/writer/render.py`, `scientist_one/writer/compose.py`
- Test: `tests/test_compose.py`

**Interfaces:**
- Consumes: `LLMClient` (Task 3), `TaskSpec` (Task 4).
- Produces: `Renderer` (Protocol with `render(self, title: str, body_md: str, references: list[dict]) -> str`), `MarkdownRenderer` (deterministic: `# title`, body, `## References` numbered from the reference dicts' `title`/`authors`/`year`/`url`), and `compose(llm: LLMClient, task: TaskSpec, narrative: str, references: list[dict], renderer: Renderer | None = None) -> str` — one reasoning call expands the narrative into paper sections (Introduction/Method/Results/Conclusion) **preserving `{ev:...}` tags**; an empty/whitespace reply falls back to the narrative unchanged; result goes through the renderer (default `MarkdownRenderer`). The composed output KEEPS evidence tags — the Claim Verifier (Task 15) needs them; tags are stripped only at promotion.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_compose.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_compose.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/writer/render.py
from typing import Protocol


class Renderer(Protocol):
    def render(self, title: str, body_md: str, references: list[dict]) -> str: ...


class MarkdownRenderer:
    def render(self, title: str, body_md: str, references: list[dict]) -> str:
        parts = [f"# {title}", "", body_md.strip()]
        if references:
            parts += ["", "## References", ""]
            for i, ref in enumerate(references, 1):
                authors = ", ".join(ref.get("authors") or []) or "Unknown"
                parts.append(f"{i}. {authors} ({ref.get('year')}). "
                             f"{ref['title']}. {ref.get('url', '')}")
        return "\n".join(parts) + "\n"
```

```python
# scientist_one/writer/compose.py
from ..llm import LLMClient
from ..tasks.base import TaskSpec
from .render import MarkdownRenderer, Renderer


def compose(llm: LLMClient, task: TaskSpec, narrative: str,
            references: list[dict], renderer: Renderer | None = None) -> str:
    renderer = renderer or MarkdownRenderer()
    body = llm.chat(
        "reasoning",
        "You compose research papers in markdown with sections Introduction, "
        "Method, Results, Conclusion. HARD RULES: keep every {ev:...} "
        "evidence tag attached to its sentence; never add facts, numbers, or "
        "claims that are not in the narrative; write prose around the "
        "established facts only.",
        f"Task: {task.description}\n\nGrounded narrative:\n{narrative}\n\n"
        "Compose the paper body (no title line, no references section).",
    )
    if not body.strip():
        body = narrative
    title = f"Automated Discovery for {task.name.replace('_', ' ').title()}"
    return renderer.render(title, body, references)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_compose.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/writer/render.py scientist_one/writer/compose.py tests/test_compose.py
git commit -m "feat: compose phase with pluggable renderer (markdown v1)"
```

---

### Task 15: Claim Verifier and Refiner

**Files:**
- Create: `scientist_one/verifier/__init__.py`, `scientist_one/verifier/run.py`
- Test: `tests/test_verifier.py`

**Interfaces:**
- Consumes: `EvidenceStore` (Task 2), `LLMClient` (Task 3), `Config` (Task 1), ground helpers `TAG_RE`, `sentences`, `numbers_in_text`, `numbers_in_payload` (Task 12).
- Produces: `Violation` (pydantic `{claim: str, reason: str}`) and `run_verifier(llm: LLMClient, config: Config, run_dir: Path, paper_md: str, store: EvidenceStore, references: list[dict], solution_code: str) -> VerifierResult` where `VerifierResult` is pydantic `{promoted: bool, paper_path: str, violations: list[Violation]}`. Behavior: extract claims deterministically from `{ev:...}` tags in the composed paper; dispatch by referenced record type — `eval-result`/`ablation` → numeric check (numbers within `config.verifier.numeric_tolerance`), `paper` → citation entailment judge (12b, `{"supported": bool}`; judge failure = violation "unverifiable"), `solution` → method–code judge (same shape). If violations: one Refiner pass (reasoning model rewrites/removes flagged sentences, keeps tags), then re-verify. Clean → strip all tags, write `run_dir/paper.md`, `promoted=True`. Still dirty → write `run_dir/paper.draft.md` + `run_dir/violations.json`, `promoted=False`. Untagged numeric sentences in the paper body are violations too (compose drift).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_verifier.py
import json
from pathlib import Path

from scientist_one.config import Config
from scientist_one.evidence import EvidenceStore
from scientist_one.llm import FakeBackend, LLMClient
from scientist_one.verifier.run import run_verifier

SUPPORTED = json.dumps({"supported": True})
UNSUPPORTED = json.dumps({"supported": False})
CODE = "def pack(items, capacity): return [[i] for i in items]\n"


def seeded(tmp_path):
    store = EvidenceStore(tmp_path / "e.jsonl")
    paper = store.append("paper", "investigator",
                         {"title": "FFD Analysis", "abstract": "FFD is 11/9 OPT."})
    idea = store.append("idea", "discovery", {"title": "FFD"})
    sol = store.append("solution", "discovery", {"path": "s.py"}, sources=[idea])
    ev = store.append("eval-result", "discovery",
                      {"ok": True, "score": 1.08, "log": "mean_ratio=1.0800"},
                      sources=[sol])
    return store, paper, sol, ev


def test_clean_paper_promoted(tmp_path):
    store, paper_rec, sol, ev = seeded(tmp_path)
    paper = (f"# T\nWe reach ratio 1.08. {{ev:{ev}}}\n"
             f"FFD is known to be near-optimal. {{ev:{paper_rec}}}\n"
             f"Our method sorts items descending. {{ev:{sol}}}\n")
    llm = LLMClient(Config(), tmp_path,
                    backend=FakeBackend([SUPPORTED, SUPPORTED]))  # citation, method
    result = run_verifier(llm, Config(), tmp_path, paper, store, [], CODE)
    assert result.promoted is True
    final = Path(result.paper_path).read_text()
    assert "{ev:" not in final                       # tags stripped
    assert result.paper_path.endswith("paper.md")


def test_numeric_violation_refined_then_promoted(tmp_path):
    store, _, _, ev = seeded(tmp_path)
    bad = f"# T\nWe reach ratio 9.99. {{ev:{ev}}}\n"
    fixed = f"# T\nWe reach ratio 1.08. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([fixed]))  # refiner
    result = run_verifier(llm, Config(), tmp_path, bad, store, [], CODE)
    assert result.promoted is True
    assert "1.08" in Path(result.paper_path).read_text()


def test_unfixable_paper_stays_draft(tmp_path):
    store, _, _, ev = seeded(tmp_path)
    bad = f"# T\nWe reach ratio 9.99. {{ev:{ev}}}\n"
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend([bad]))  # refiner no-op
    result = run_verifier(llm, Config(), tmp_path, bad, store, [], CODE)
    assert result.promoted is False
    assert result.paper_path.endswith("paper.draft.md")
    assert (Path(tmp_path) / "violations.json").exists()
    assert result.violations[0].reason.startswith("number")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_verifier.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/verifier/__init__.py
```

```python
# scientist_one/verifier/run.py
import json
import math
from pathlib import Path

from pydantic import BaseModel

from ..config import Config
from ..evidence import EvidenceStore
from ..llm import LLMClient
from ..writer.ground import TAG_RE, numbers_in_payload, numbers_in_text, sentences


class Violation(BaseModel):
    claim: str
    reason: str


class VerifierResult(BaseModel):
    promoted: bool
    paper_path: str
    violations: list[Violation]


class Entailment(BaseModel):
    supported: bool


def _verify(llm: LLMClient, config: Config, paper_md: str, store: EvidenceStore,
            solution_code: str) -> list[Violation]:
    known = {r.id for r in store.all()}
    violations: list[Violation] = []
    for sent in sentences(paper_md):
        tags = TAG_RE.findall(sent)
        clean = TAG_RE.sub("", sent).strip()
        nums = numbers_in_text(sent)
        if not tags:
            if nums:
                violations.append(Violation(
                    claim=clean, reason="untagged numeric claim (compose drift)"))
            continue
        for tag in tags:
            if tag not in known:
                violations.append(Violation(claim=clean,
                                            reason=f"unknown evidence {tag}"))
                continue
            rec = store.get(tag)
            if rec.type in ("eval-result", "ablation"):
                available = numbers_in_payload(rec.payload)
                for num in nums:
                    if not any(math.isclose(num, a,
                               rel_tol=config.verifier.numeric_tolerance,
                               abs_tol=1e-9) for a in available):
                        violations.append(Violation(
                            claim=clean,
                            reason=f"number {num} not in evidence {tag}"))
            elif rec.type == "paper":
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
                        claim=clean, reason=f"method claim not matched by code"))
    return violations


def run_verifier(llm: LLMClient, config: Config, run_dir: Path, paper_md: str,
                 store: EvidenceStore, references: list[dict],
                 solution_code: str) -> VerifierResult:
    run_dir = Path(run_dir)
    violations = _verify(llm, config, paper_md, store, solution_code)
    if violations:
        listing = "\n".join(f"- {v.claim}: {v.reason}" for v in violations)
        paper_md = llm.chat(
            "reasoning",
            "You repair research papers. Rewrite each flagged sentence to "
            "match its evidence, or DELETE it if it cannot be supported. "
            "Keep all valid {ev:...} tags. Reply with the full markdown.",
            f"Paper:\n{paper_md}\n\nFlagged claims:\n{listing}",
        ) or paper_md
        violations = _verify(llm, config, paper_md, store, solution_code)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_verifier.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add scientist_one/verifier/ tests/test_verifier.py
git commit -m "feat: claim verifier with per-type dispatch and refiner gate"
```

---

### Task 16: Pipeline orchestration and CLI

**Files:**
- Create: `scientist_one/pipeline.py`, `scientist_one/cli.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 1–15.
- Produces: `run_pipeline(config: Config, task_path: Path, run_dir: Path, backend: Backend | None = None, http_client: httpx.Client | None = None) -> dict` — runs investigator → discovery → writer(+verifier), writing a stage marker `run_dir/<stage>.json` after each stage; a rerun on the same `run_dir` skips stages whose marker exists (resume). Returns the manifest dict, also written to `run_dir/manifest.json`: `{"task_path", "status" ("complete" | "discovery-failed" | "not-promoted"), "best_solution_path" | None, "best_eval_id" | None, "paper_path" | None, "promoted": bool, "references_path"}`. CLI (`scientist_one/cli.py`, entry point `main()`): `scientist-one run --task PATH [--config PATH] [--run-dir PATH]` (default run dir `runs/<UTC timestamp>`; rerunning with `--run-dir` of an existing run resumes it) and `scientist-one status RUN_DIR` (prints which stage markers exist). The `audit` subcommand is added in Task 17.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pipeline.py
import json
from pathlib import Path

import httpx

from scientist_one.config import Config
from scientist_one.llm import FakeBackend
from scientist_one.pipeline import run_pipeline

TASK_PATH = Path("tasks/bin_packing")

FFD_CODE = ("def pack(items, capacity):\n"
            "    bins = []\n"
            "    for item in sorted(items, reverse=True):\n"
            "        for b in bins:\n"
            "            if sum(b) + item <= capacity:\n"
            "                b.append(item)\n"
            "                break\n"
            "        else:\n"
            "            bins.append([item])\n"
            "    return bins\n")

BRIEF = json.dumps({"framing": "Pack items into few bins.",
                    "claims": [{"text": "FFD is a strong baseline.",
                                "paper_indexes": []}],
                    "baselines": "first fit"})
IDEAS_A = json.dumps({"ideas": [{"title": "FFD", "approach": "sort desc",
                                 "rationale": "classic"}]})
IDEAS_B = json.dumps({"ideas": []})
SCORES = json.dumps({"scores": [{"index": 0, "novelty": 3, "feasibility": 5}]})
NOT_FLAGGED = json.dumps({"flagged": False, "reason": "ok"})
NO_COMPONENTS = json.dumps({"components": []})
# ev_0001 brief-claim, ev_0002 idea, ev_0003 solution, ev_0004 eval-result
NARRATIVE = "The heuristic packs items efficiently. {ev:ev_0004}"
NO_ISSUES = json.dumps({"issues": []})

RESPONSES = [
    BRIEF,                          # investigator brief
    IDEAS_A, IDEAS_B, SCORES,       # ideator
    f"```python\n{FFD_CODE}```",    # solve
    NOT_FLAGGED,                    # spec audit
    NO_COMPONENTS,                  # ablations
    NARRATIVE,                      # conceive
    NO_ISSUES,                      # critic
    NARRATIVE,                      # compose body
]


def no_network():
    return httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(500)))


def small_config():
    return Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                  solver={"timeout_s": 60})


def test_pipeline_end_to_end(tmp_path):
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend(list(RESPONSES)),
                            http_client=no_network())
    assert manifest["status"] == "complete"
    assert manifest["promoted"] is True
    paper = Path(manifest["paper_path"]).read_text()
    assert "{ev:" not in paper
    assert (tmp_path / "investigator.json").exists()
    assert (tmp_path / "discovery.json").exists()
    assert (tmp_path / "writer.json").exists()


def test_pipeline_resumes_without_llm(tmp_path):
    run_pipeline(small_config(), TASK_PATH, tmp_path,
                 backend=FakeBackend(list(RESPONSES)), http_client=no_network())
    # Empty backend: any LLM call would raise IndexError — resume must not call
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend([]), http_client=no_network())
    assert manifest["status"] == "complete"


def test_pipeline_discovery_failure(tmp_path):
    responses = [BRIEF, IDEAS_A, IDEAS_B, SCORES, "no code ((("]
    manifest = run_pipeline(small_config(), TASK_PATH, tmp_path,
                            backend=FakeBackend(responses),
                            http_client=no_network())
    assert manifest["status"] == "discovery-failed"
    assert manifest["paper_path"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/pipeline.py
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
```

```python
# scientist_one/cli.py
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .config import load_config
from .pipeline import run_pipeline

STAGES = ("investigator", "discovery", "writer")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="scientist-one")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="run the full pipeline on a task")
    run_p.add_argument("--task", required=True, type=Path)
    run_p.add_argument("--config", type=Path, default=Path("config.yaml"))
    run_p.add_argument("--run-dir", type=Path, default=None,
                       help="existing run dir resumes; default runs/<timestamp>")

    status_p = sub.add_parser("status", help="show stage completion for a run")
    status_p.add_argument("run_dir", type=Path)

    args = parser.parse_args(argv)
    if args.command == "run":
        run_dir = args.run_dir or Path("runs") / datetime.now(
            timezone.utc).strftime("%Y%m%d-%H%M%S")
        manifest = run_pipeline(load_config(args.config), args.task, run_dir)
        print(json.dumps(manifest, indent=2))
        print(f"\nrun dir: {run_dir}")
    elif args.command == "status":
        for stage in STAGES:
            done = (args.run_dir / f"{stage}.json").exists()
            print(f"{stage:14} {'done' if done else 'pending'}")
        manifest = args.run_dir / "manifest.json"
        if manifest.exists():
            print(json.loads(manifest.read_text())["status"])


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_pipeline.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Smoke-check the CLI plumbing (no Ollama needed)**

Run: `scientist-one status /tmp 2>/dev/null || python -m scientist_one.cli status /tmp`
Expected: three lines ending in `pending`

- [ ] **Step 6: Commit**

```bash
git add scientist_one/pipeline.py scientist_one/cli.py tests/test_pipeline.py
git commit -m "feat: staged pipeline with resume markers and CLI"
```

---

### Task 17: CoE Integrity Audit

**Files:**
- Create: `scientist_one/audit/__init__.py`, `scientist_one/audit/run.py`
- Modify: `scientist_one/cli.py` (add `audit` subcommand)
- Test: `tests/test_audit.py`

**Interfaces:**
- Consumes: manifest layout (Task 16), `run_evaluation` (Task 6), `audit_solution` (Task 8), `search_papers` (Task 10), `LLMClient` (Task 3), `load_task` (Task 4).
- Produces: `CheckResult` (pydantic `{name: str, passed: bool | None, detail: str}` — `None` = check not applicable/unverifiable) and `run_audit(llm: LLMClient, config: Config, run_dir: Path, http_client: httpx.Client | None = None) -> AuditReport` (pydantic `{checks: list[CheckResult]}`), written to `run_dir/audit_report.json`. Checks: **I1 score-verification** (LLM extracts the reported score from the paper; solution re-run via sandbox; compare within `numeric_tolerance`), **I2 spec-violation** (3× `audit_solution`, majority vote), **I3 reference-verification** (each `references.json` entry re-resolved by title via scholarly APIs; exact case-folded title match = verified, else LLM disambiguation `{"match": bool}`; any unresolved → failed with count), **I4 method–code alignment** (3× judge `{"aligned": bool}`, majority). CLI gains `scientist-one audit RUN_DIR [--config PATH]` printing one line per check.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_audit.py
import json
from pathlib import Path

import httpx

from scientist_one.audit.run import run_audit
from scientist_one.config import Config
from scientist_one.llm import FakeBackend, LLMClient

FFD_CODE = ("def pack(items, capacity):\n"
            "    bins = []\n"
            "    for item in sorted(items, reverse=True):\n"
            "        for b in bins:\n"
            "            if sum(b) + item <= capacity:\n"
            "                b.append(item)\n"
            "                break\n"
            "        else:\n"
            "            bins.append([item])\n"
            "    return bins\n")

NOT_FLAGGED = json.dumps({"flagged": False, "reason": "ok"})
ALIGNED = json.dumps({"aligned": True})


def make_run_dir(tmp_path, paper_score: float):
    sol = tmp_path / "solution.py"
    sol.write_text(FFD_CODE)
    refs = tmp_path / "references.json"
    refs.write_text(json.dumps([{
        "title": "FFD Analysis", "authors": ["D. Johnson"], "year": 1974,
        "abstract": "", "url": "u", "source": "semantic_scholar",
        "external_id": "x"}]))
    (tmp_path / "paper.md").write_text(
        f"# T\nOur method reaches a mean ratio of {paper_score}.\n")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "task_path": "tasks/bin_packing", "status": "complete",
        "best_solution_path": str(sol), "best_eval_id": "ev_0004",
        "paper_path": str(tmp_path / "paper.md"), "promoted": True,
        "references_path": str(refs)}))
    return tmp_path


def ref_found_client():
    body = {"data": [{"title": "FFD Analysis", "abstract": "", "year": 1974,
                      "url": "u", "externalIds": {}, "authors": []}]}
    def handler(request):
        if "semanticscholar" in request.url.host:
            return httpx.Response(200, json=body)
        return httpx.Response(200, text='<?xml version="1.0"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
    return httpx.Client(transport=httpx.MockTransport(handler))


def get_true_score():
    from scientist_one.sandbox import run_evaluation
    from scientist_one.tasks.base import load_task
    import tempfile
    task = load_task(Path("tasks/bin_packing"))
    with tempfile.TemporaryDirectory() as d:
        sol = Path(d) / "s.py"
        sol.write_text(FFD_CODE)
        return run_evaluation(task, sol, Path(d), 60).score


def test_audit_all_pass(tmp_path):
    score = get_true_score()
    run_dir = make_run_dir(tmp_path, score)
    responses = [json.dumps({"score": score}),        # I1 extraction
                 NOT_FLAGGED, NOT_FLAGGED, NOT_FLAGGED,  # I2 x3
                 ALIGNED, ALIGNED, ALIGNED]              # I4 x3
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    report = run_audit(llm, Config(), run_dir, http_client=ref_found_client())
    assert all(c.passed for c in report.checks)
    assert (run_dir / "audit_report.json").exists()


def test_audit_catches_score_mismatch(tmp_path):
    run_dir = make_run_dir(tmp_path, 0.5)  # paper claims impossible score
    responses = [json.dumps({"score": 0.5}),
                 NOT_FLAGGED, NOT_FLAGGED, NOT_FLAGGED,
                 ALIGNED, ALIGNED, ALIGNED]
    llm = LLMClient(Config(), tmp_path, backend=FakeBackend(responses))
    report = run_audit(llm, Config(), run_dir, http_client=ref_found_client())
    i1 = next(c for c in report.checks if c.name == "score-verification")
    assert i1.passed is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_audit.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement**

```python
# scientist_one/audit/__init__.py
```

```python
# scientist_one/audit/run.py
import json
import math
from pathlib import Path

import httpx
from pydantic import BaseModel

from ..config import Config
from ..discovery.auditor import audit_solution
from ..investigator.scholarly import search_papers
from ..llm import LLMClient
from ..sandbox import run_evaluation
from ..tasks.base import load_task


class CheckResult(BaseModel):
    name: str
    passed: bool | None
    detail: str


class AuditReport(BaseModel):
    checks: list[CheckResult]


class ReportedScore(BaseModel):
    score: float | None


class Aligned(BaseModel):
    aligned: bool


class RefMatch(BaseModel):
    match: bool


def _i1_score(llm, config, task, paper: str, solution_path: str,
              run_dir: Path) -> CheckResult:
    extracted = llm.chat_json(
        "judging", "Extract the main reported score from the paper. Return JSON.",
        f"Paper:\n{paper}\n\n"
        'Return {"score": float or null if no score is reported}.',
        ReportedScore)
    if extracted is None or extracted.score is None:
        return CheckResult(name="score-verification", passed=None,
                           detail="no score extracted from paper")
    outcome = run_evaluation(task, Path(solution_path), run_dir / "audit_work",
                             task.timeout_s or config.solver.timeout_s)
    if not outcome.ok:
        return CheckResult(name="score-verification", passed=False,
                           detail=f"solution failed re-run: {outcome.log[:200]}")
    ok = math.isclose(extracted.score, outcome.score,
                      rel_tol=config.verifier.numeric_tolerance, abs_tol=1e-9)
    return CheckResult(
        name="score-verification", passed=ok,
        detail=f"paper={extracted.score} rerun={outcome.score}")


def _i2_spec(llm, task, code: str) -> CheckResult:
    verdicts = [audit_solution(llm, task, code).flagged for _ in range(3)]
    flagged = sum(verdicts) >= 2
    return CheckResult(name="spec-violation", passed=not flagged,
                       detail=f"votes flagged={sum(verdicts)}/3")


def _i3_references(llm, references: list[dict],
                   client: httpx.Client) -> CheckResult:
    unresolved = []
    for ref in references:
        found = search_papers(client, [ref["title"]], limit_per_query=5)
        titles = {p.title.strip().casefold() for p in found}
        if ref["title"].strip().casefold() in titles:
            continue
        verdict = llm.chat_json(
            "judging", "Decide whether any candidate is the same paper. Return JSON.",
            f"Reference: {ref['title']} ({ref.get('year')})\n"
            f"Candidates: {[p.title for p in found]}\n"
            'Return {"match": bool}.',
            RefMatch)
        if verdict is None or not verdict.match:
            unresolved.append(ref["title"])
    return CheckResult(name="reference-verification", passed=not unresolved,
                       detail=f"{len(unresolved)} unresolved of "
                              f"{len(references)}: {unresolved[:3]}")


def _i4_alignment(llm, paper: str, code: str) -> CheckResult:
    votes = []
    for _ in range(3):
        verdict = llm.chat_json(
            "judging",
            "Judge whether the paper's method section faithfully describes "
            "the code. Simplification is aligned; a fundamentally different "
            "algorithm is misaligned. Return JSON.",
            f"Paper:\n{paper}\n\nCode:\n```python\n{code}```\n"
            'Return {"aligned": bool}.',
            Aligned)
        votes.append(verdict.aligned if verdict else False)
    aligned = sum(votes) >= 2
    return CheckResult(name="method-code-alignment", passed=aligned,
                       detail=f"votes aligned={sum(votes)}/3")


def run_audit(llm: LLMClient, config: Config, run_dir: Path,
              http_client: httpx.Client | None = None) -> AuditReport:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    task = load_task(Path(manifest["task_path"]))
    paper = Path(manifest["paper_path"]).read_text()
    code = Path(manifest["best_solution_path"]).read_text()
    references = json.loads(Path(manifest["references_path"]).read_text())
    client = http_client or httpx.Client(timeout=30)

    report = AuditReport(checks=[
        _i1_score(llm, config, task, paper, manifest["best_solution_path"], run_dir),
        _i2_spec(llm, task, code),
        _i3_references(llm, references, client),
        _i4_alignment(llm, paper, code),
    ])
    (run_dir / "audit_report.json").write_text(report.model_dump_json(indent=2))
    return report
```

- [ ] **Step 4: Add the `audit` CLI subcommand**

In `scientist_one/cli.py`, add imports and the subcommand. Add to the imports:

```python
from .audit.run import run_audit
from .llm import LLMClient
```

Add after the `status` parser definition:

```python
    audit_p = sub.add_parser("audit", help="run the CoE integrity audit on a run")
    audit_p.add_argument("run_dir", type=Path)
    audit_p.add_argument("--config", type=Path, default=Path("config.yaml"))
```

Add the branch in the command dispatch:

```python
    elif args.command == "audit":
        config = load_config(args.config)
        llm = LLMClient(config, args.run_dir)
        report = run_audit(llm, config, args.run_dir)
        for check in report.checks:
            mark = {True: "PASS", False: "FAIL", None: "N/A "}[check.passed]
            print(f"[{mark}] {check.name}: {check.detail}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_audit.py -v`
Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add scientist_one/audit/ scientist_one/cli.py tests/test_audit.py
git commit -m "feat: CoE integrity audit (I1-I4) with CLI subcommand"
```

---

### Task 18: Live smoke test and README

**Files:**
- Create: `tests/test_live_smoke.py`, `README.md`

**Interfaces:**
- Consumes: `run_pipeline` (Task 16), real Ollama server with the configured models.

- [ ] **Step 1: Write the live smoke test (opt-in only)**

```python
# tests/test_live_smoke.py
"""Live smoke test: needs Ollama running with the configured models.

Run explicitly with:  pytest -m live tests/test_live_smoke.py -v
Excluded from default runs by pyproject's `-m 'not live'` addopts.
"""
from pathlib import Path

import pytest

from scientist_one.config import Config
from scientist_one.pipeline import run_pipeline


@pytest.mark.live
def test_tiny_live_run(tmp_path):
    config = Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                    investigator={"max_papers": 3},
                    writer={"max_rounds": 2})
    manifest = run_pipeline(config, Path("tasks/bin_packing"), tmp_path)
    # A live run may legitimately end not-promoted with a weak local model;
    # what must hold: it terminates, evaluates real code, and leaves artifacts.
    assert manifest["status"] in ("complete", "not-promoted", "discovery-failed")
    assert (tmp_path / "evidence.jsonl").exists()
    if manifest["status"] != "discovery-failed":
        assert Path(manifest["best_solution_path"]).exists()
        assert Path(manifest["paper_path"]).exists()
```

- [ ] **Step 2: Verify it is skipped by default and runs live**

Run: `pytest tests/test_live_smoke.py -v`
Expected: `1 deselected` (the `-m 'not live'` default filter)

Then, with Ollama up (`docker start ollama`):
Run: `pytest -m live tests/test_live_smoke.py -v --timeout=1800 2>/dev/null || pytest -m live tests/test_live_smoke.py -v`
Expected: 1 PASSED (may take many minutes on local models; network needed for Semantic Scholar/arXiv)

- [ ] **Step 3: Write the README**

```markdown
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
```

- [ ] **Step 4: Run the full offline suite one last time**

Run: `pytest -v`
Expected: all tests pass, live test deselected

- [ ] **Step 5: Commit**

```bash
git add tests/test_live_smoke.py README.md
git commit -m "feat: live smoke test and README"
```

---

## Plan Self-Review Notes

- **Network isolation of the sandbox:** the spec says solver code runs with "no network". The subprocess sandbox (Task 6) enforces isolation-by-process and timeout; full network denial is not portably achievable without containers. Accepted deviation for v1: the evaluator subprocess inherits network. If stricter isolation is wanted later, wrap the subprocess command with `unshare -rn` on Linux.
- **Full-text PDFs:** the spec marks PDF fetching "optional/best-effort"; v1 uses abstracts only (Task 11). The `PaperMeta.url` field preserves the pointer for a future fetcher.
- **Stage 1 citation-graph hop:** approximated by multi-query search + dedupe (Task 10) rather than a true citation-graph expansion; seed queries come from `task.yaml`.

