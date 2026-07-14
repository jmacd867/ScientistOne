# Memory Bandwidth Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `memory_bandwidth` ScientistOne research task that discovers techniques maximizing achieved GPU memory bandwidth (a STREAM-style triad kernel) on this machine's GB10, plus the small solver change needed to let any task opt into non-stdlib imports.

**Architecture:** One additive `allowed_imports: list[str]` field on `TaskSpec`, consumed by `solver.py`'s system-prompt builder. A new `tasks/memory_bandwidth/` directory (`task.yaml`, `starter.py`, `evaluator.py`) follows the exact same shape as the existing `tasks/bin_packing/` task — no pipeline code beyond the solver prompt changes.

**Tech Stack:** Python 3.12, PyTorch (CUDA), pytest.

## Global Constraints

- `metric_direction: higher` for this task (score = achieved GB/s).
- `allowed_imports: [torch]` — the only non-stdlib import this task's solutions may use.
- Task-level `timeout_s: 120` (overrides the config default of 60s).
- Existing tasks (e.g. `bin_packing`) must be unaffected — `allowed_imports` defaults to `[]` and the solver prompt text for that case is byte-for-byte what it is today.
- GPU-dependent tests get a new `gpu` pytest marker, excluded from default `addopts` alongside the existing `live` marker. Do not conflate `gpu` with `live` — `live` means "needs a running Ollama server," `gpu` means "needs a CUDA-capable GPU."
- This machine already has a working CUDA-enabled `torch` (2.11.0+cu130) in the active venv at `/home/jmacd745/coding/.venv` (confirmed via `nvidia-smi` showing an NVIDIA GB10 and `torch.cuda.is_available() == True`). Do **not** attempt to install a fresh `torch` from bare PyPI into a different venv — standard wheels are not guaranteed to have CUDA support for this ARM+GB10 combination, and doing so risks silently shadowing the working build with a broken one. Task 2 below installs the project's own `gpu` extra into the *already-active* venv, where `torch` is already satisfied.

---

### Task 1: `allowed_imports` on `TaskSpec` + dynamic solver prompt

**Files:**
- Modify: `scientist_one/tasks/base.py`
- Modify: `scientist_one/discovery/solver.py`
- Test: `tests/test_tasks.py`
- Test: `tests/test_solver_auditor.py`

**Interfaces:**
- Consumes: existing `TaskSpec` (pydantic `BaseModel` in `scientist_one/tasks/base.py`), existing `solve(llm: LLMClient, task: TaskSpec, idea: dict, feedback: str | None) -> str | None` in `scientist_one/discovery/solver.py`.
- Produces: `TaskSpec.allowed_imports: list[str]` (default `[]`), consumed by Task 3's `task.yaml`. `solver._system(task: TaskSpec) -> str`, a new private helper other tasks don't need to call directly.

- [ ] **Step 1: Write the failing tests for `allowed_imports` parsing**

Add to `tests/test_tasks.py` (after `test_load_task`):

```python
def test_allowed_imports_defaults_empty(tmp_path):
    d = make_task_dir(tmp_path,
        "name: demo\ndescription: a demo\nmetric_direction: lower\n")
    task = load_task(d)
    assert task.allowed_imports == []


def test_allowed_imports_parsed(tmp_path):
    d = make_task_dir(tmp_path,
        "name: demo\ndescription: a demo\nmetric_direction: higher\n"
        "allowed_imports: [torch]\n")
    task = load_task(d)
    assert task.allowed_imports == ["torch"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_tasks.py -v`
Expected: `test_allowed_imports_defaults_empty` and `test_allowed_imports_parsed` FAIL with `AttributeError: 'TaskSpec' object has no attribute 'allowed_imports'` (or a pydantic "extra fields not permitted" error).

- [ ] **Step 3: Add the field to `TaskSpec`**

In `scientist_one/tasks/base.py`, change:

```python
class TaskSpec(BaseModel):
    name: str
    description: str
    metric_direction: Literal["higher", "lower"]
    seed_queries: list[str] = []
    timeout_s: int | None = None
    path: Path
```

to:

```python
class TaskSpec(BaseModel):
    name: str
    description: str
    metric_direction: Literal["higher", "lower"]
    seed_queries: list[str] = []
    allowed_imports: list[str] = []
    timeout_s: int | None = None
    path: Path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_tasks.py -v`
Expected: PASS (all tests in the file, including the two new ones and the pre-existing `test_load_task`/`test_missing_file_raises`).

- [ ] **Step 5: Commit**

```bash
git add scientist_one/tasks/base.py tests/test_tasks.py
git commit -m "feat: add allowed_imports field to TaskSpec"
```

- [ ] **Step 6: Write the failing tests for the solver's dynamic system prompt**

Add to `tests/test_solver_auditor.py`. First add `TaskSpec` to the existing import line:

```python
from scientist_one.tasks.base import TaskSpec, load_task
```

Then append these two tests at the end of the file:

```python
def test_solve_prompt_stdlib_only_by_default(tmp_path):
    captured = {}

    def backend(model, system, user, format):
        captured["system"] = system
        return f"```python\n{CODE}```"

    llm = LLMClient(Config(), tmp_path, backend=backend)
    solve(llm, TASK, IDEA, None)
    assert "only the standard library" in captured["system"]
    assert "torch" not in captured["system"]


def test_solve_prompt_lists_allowed_imports(tmp_path):
    captured = {}

    def backend(model, system, user, format):
        captured["system"] = system
        return f"```python\n{CODE}```"

    llm = LLMClient(Config(), tmp_path, backend=backend)
    task = TaskSpec(name="gpu-task", description="d", metric_direction="higher",
                     allowed_imports=["torch"], path=TASK.path)
    solve(llm, task, IDEA, None)
    assert "torch" in captured["system"]
    assert "only the standard library" not in captured["system"]
```

- [ ] **Step 7: Run the tests to verify they fail**

Run: `pytest tests/test_solver_auditor.py -v`
Expected: `test_solve_prompt_stdlib_only_by_default` FAILs (or errors) because `captured` is never populated the way expected — actually, since the current `_SYSTEM` is a fixed string already containing "only the standard library", this specific test will PASS by accident. `test_solve_prompt_lists_allowed_imports` FAILs because `TaskSpec` doesn't affect the prompt yet: `"torch" not in captured["system"]` — assertion `assert "torch" in captured["system"]` fails.

- [ ] **Step 8: Make the solver system prompt task-dependent**

Replace the top of `scientist_one/discovery/solver.py`:

```python
import ast
import re

from ..llm import LLMClient
from ..tasks.base import TaskSpec

_SYSTEM = ("You are an expert algorithm implementer. Reply with a single "
           "complete Python solution in one ```python code fence. The code "
           "must define exactly the interface in the starter template, use "
           "only the standard library, and be deterministic.")
```

with:

```python
import ast
import re

from ..llm import LLMClient
from ..tasks.base import TaskSpec


def _system(task: TaskSpec) -> str:
    allowed = "only the standard library"
    if task.allowed_imports:
        allowed = "the standard library plus: " + ", ".join(task.allowed_imports)
    return ("You are an expert algorithm implementer. Reply with a single "
            "complete Python solution in one ```python code fence. The code "
            "must define exactly the interface in the starter template, use "
            f"{allowed}, and be deterministic.")
```

Then in `solve()`, change the final line from:

```python
    return _extract_code(llm.chat("reasoning", _SYSTEM, prompt))
```

to:

```python
    return _extract_code(llm.chat("reasoning", _system(task), prompt))
```

- [ ] **Step 9: Run the tests to verify they pass**

Run: `pytest tests/test_solver_auditor.py -v`
Expected: PASS (all four tests in the file).

- [ ] **Step 10: Run the full fast test suite to check for regressions**

Run: `pytest`
Expected: PASS (no `live`- or `gpu`-marked tests run by default).

- [ ] **Step 11: Commit**

```bash
git add scientist_one/discovery/solver.py tests/test_solver_auditor.py
git commit -m "feat: let tasks declare allowed_imports beyond the standard library"
```

---

### Task 2: `gpu` extra, `gpu` pytest marker, verify the active venv's torch

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: a `gpu` pytest marker available for Task 3/4's tests; a `[project.optional-dependencies] gpu` extra documenting the `torch` requirement.

- [ ] **Step 1: Add the `gpu` extra and register the `gpu` marker**

In `pyproject.toml`, change:

```toml
[project.optional-dependencies]
dev = ["pytest"]
```

to:

```toml
[project.optional-dependencies]
dev = ["pytest"]
gpu = ["torch"]
```

And change:

```toml
[tool.pytest.ini_options]
markers = ["live: requires a running Ollama server"]
addopts = "-m 'not live'"
```

to:

```toml
[tool.pytest.ini_options]
markers = [
    "live: requires a running Ollama server",
    "gpu: requires a CUDA-capable GPU",
]
addopts = "-m 'not live and not gpu'"
```

- [ ] **Step 2: Install the new extra into the active venv**

Run: `pip install -e ".[dev,gpu]"`
Expected: succeeds quickly — this machine's active venv (`/home/jmacd745/coding/.venv`, on `PATH` via `~/.bashrc`) already has a working CUDA-enabled `torch`, so pip resolves the `gpu` extra as already satisfied rather than downloading a new build.

- [ ] **Step 3: Verify torch/CUDA are visible from this exact interpreter**

Run: `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
Expected: prints a version string and `True`.

- [ ] **Step 4: Verify the new pytest marker is registered and default runs are unaffected**

Run: `pytest --markers | grep gpu`
Expected: prints `@pytest.mark.gpu: requires a CUDA-capable GPU`.

Run: `pytest`
Expected: PASS, same test count as before this task (still excludes `live` and `gpu`).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add gpu optional-dependency extra and pytest marker"
```

---

### Task 3: `tasks/memory_bandwidth/` — task.yaml, starter.py, evaluator.py

**Files:**
- Create: `tasks/memory_bandwidth/task.yaml`
- Create: `tasks/memory_bandwidth/starter.py`
- Create: `tasks/memory_bandwidth/evaluator.py`
- Test: `tests/test_memory_bandwidth.py`

**Interfaces:**
- Consumes: `TaskSpec.allowed_imports` (Task 1), `load_task` from `scientist_one/tasks/base.py`.
- Produces: the `triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor` solution interface that Task 4's live pipeline run will generate candidate implementations against; `evaluate(solution_path: str, workdir: str) -> dict` with keys `"score"` (float, GB/s) and `"log"` (str).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_memory_bandwidth.py`:

```python
import importlib.util
from pathlib import Path

import pytest

from scientist_one.tasks.base import load_task

pytestmark = pytest.mark.gpu

TASK_DIR = Path("tasks/memory_bandwidth")


def load_evaluator():
    spec = importlib.util.spec_from_file_location("evaluator", TASK_DIR / "evaluator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_task_loads():
    task = load_task(TASK_DIR)
    assert task.metric_direction == "higher"
    assert task.allowed_imports == ["torch"]


def test_starter_triad_scores(tmp_path):
    result = load_evaluator().evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    assert result["score"] > 0
    assert "GB/s" in result["log"]


def test_wrong_shape_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return (a + scalar * b)[:-1]\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))


def test_wrong_dtype_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return (a + scalar * b).to(torch.float16)\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))


def test_incorrect_result_rejected(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "import torch\n"
        "def triad(a, b, scalar):\n"
        "    return a - scalar * b\n"
    )
    with pytest.raises(ValueError):
        load_evaluator().evaluate(str(bad), str(tmp_path))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest -m gpu tests/test_memory_bandwidth.py -v`
Expected: all five tests FAIL/ERROR — `tasks/memory_bandwidth/` doesn't exist yet, so `load_task` raises `FileNotFoundError` and `load_evaluator()` raises `FileNotFoundError` from `spec_from_file_location`/`spec.loader`.

- [ ] **Step 3: Create `task.yaml`**

Create `tasks/memory_bandwidth/task.yaml`:

```yaml
name: memory_bandwidth
description: >
  Maximize achieved GPU memory bandwidth (GB/s) for a STREAM-style triad
  operation (c = a + scalar * b) on CUDA tensors already resident in GPU
  memory. The starter implementation is a naive elementwise expression that
  allocates an avoidable intermediate tensor. Discover kernel-level changes
  (in-place ops, fusion, avoiding intermediate allocations, torch.compile,
  custom kernels) that increase measured bandwidth, without changing the
  output dtype or shape. Higher GB/s is better.
metric_direction: higher
allowed_imports: [torch]
timeout_s: 120
seed_queries:
  - GPU memory bandwidth STREAM benchmark
  - roofline model memory-bound kernel optimization
  - CUDA kernel fusion reduce memory traffic
  - unified memory bandwidth Grace Blackwell superchip
  - PyTorch avoid intermediate tensor allocation elementwise
```

- [ ] **Step 4: Create `starter.py`**

Create `tasks/memory_bandwidth/starter.py`:

```python
import torch


def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU.

    Baseline: naive elementwise expression. This allocates an intermediate
    tensor for `scalar * b` plus the output tensor — two avoidable
    allocations/passes a tuned kernel can remove.
    """
    return a + scalar * b
```

- [ ] **Step 5: Create `evaluator.py`**

Create `tasks/memory_bandwidth/evaluator.py`:

```python
import importlib.util

import torch

_SIZES = [32_000_000, 128_000_000, 512_000_000]  # elements per tensor (float32)
_WARMUP = 5
_REPS = 30
_SCALAR = 3.0
_SEED = 0


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.triad


def _check_solution(triad, a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    c = triad(a, b, scalar)
    if c.shape != a.shape:
        raise ValueError(f"output shape {tuple(c.shape)} != input shape {tuple(a.shape)}")
    if c.dtype != a.dtype:
        raise ValueError(f"output dtype {c.dtype} != input dtype {a.dtype}")
    if c.device != a.device:
        raise ValueError(f"output device {c.device} != input device {a.device}")
    expected = a + scalar * b
    if not torch.allclose(c, expected, rtol=1e-4, atol=1e-4):
        raise ValueError("triad result does not match a + scalar * b within tolerance")
    return c


def evaluate(solution_path: str, workdir: str) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("memory_bandwidth task requires a CUDA-capable GPU")
    triad = _load_solution(solution_path)
    device = torch.device("cuda")
    gbps_by_size, lines = [], []
    for n in _SIZES:
        gen = torch.Generator(device=device).manual_seed(_SEED)
        a = torch.rand(n, device=device, dtype=torch.float32, generator=gen)
        b = torch.rand(n, device=device, dtype=torch.float32, generator=gen)
        _check_solution(triad, a, b, _SCALAR)
        for _ in range(_WARMUP):
            triad(a, b, _SCALAR)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(_REPS):
            triad(a, b, _SCALAR)
        end.record()
        torch.cuda.synchronize()
        elapsed_s = start.elapsed_time(end) / 1000.0
        bytes_per_call = 3 * n * a.element_size()  # read a, read b, write c
        gbps = (bytes_per_call * _REPS) / elapsed_s / 1e9
        gbps_by_size.append(gbps)
        lines.append(f"n={n}: {gbps:.2f} GB/s")
    score = sum(gbps_by_size) / len(gbps_by_size)
    lines.append(f"mean over {len(_SIZES)} sizes: {score:.2f} GB/s")
    return {"score": round(score, 2), "log": "\n".join(lines)}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest -m gpu tests/test_memory_bandwidth.py -v`
Expected: PASS (all five tests). `test_starter_triad_scores` should print a score in the tens-to-low-hundreds of GB/s range (this machine's GB10 unified memory).

- [ ] **Step 7: Run the full fast test suite to check for regressions**

Run: `pytest`
Expected: PASS, same as before (the new tests are `gpu`-marked and excluded by default `addopts`).

- [ ] **Step 8: Commit**

```bash
git add tasks/memory_bandwidth/ tests/test_memory_bandwidth.py
git commit -m "feat: add memory_bandwidth GPU task"
```

---

### Task 4: Live end-to-end smoke test

**Files:**
- Modify: `tests/test_live_smoke.py`

**Interfaces:**
- Consumes: `run_pipeline` from `scientist_one/pipeline.py` (already imported in this file), `tasks/memory_bandwidth/` (Task 3).
- Produces: nothing consumed elsewhere — this is the final verification that `allowed_imports` actually reaches the solver inside a real pipeline run and that a CUDA solution executes correctly through the sandbox subprocess.

- [ ] **Step 1: Add the live+gpu smoke test**

Append to `tests/test_live_smoke.py`:

```python
@pytest.mark.live
@pytest.mark.gpu
def test_tiny_live_run_memory_bandwidth(tmp_path):
    config = Config(discovery={"branches": 1, "iterations": 1, "survivors": 1},
                    investigator={"max_papers": 3},
                    writer={"max_rounds": 2})
    manifest = run_pipeline(config, Path("tasks/memory_bandwidth"), tmp_path)
    assert manifest["status"] in ("complete", "not-promoted", "discovery-failed")
    assert (tmp_path / "evidence.jsonl").exists()
    if manifest["status"] != "discovery-failed":
        assert Path(manifest["best_solution_path"]).exists()
        assert Path(manifest["paper_path"]).exists()
```

- [ ] **Step 2: Run it against the real, running Ollama server + this machine's GPU**

Run: `docker exec ollama ollama list` (confirm `gemma4:26b`/`gemma4:12b` are pulled; start the container first if it isn't running)
Run: `pytest -m "live and gpu" tests/test_live_smoke.py -v`
Expected: PASS — `manifest["status"]` is one of the three allowed values, `evidence.jsonl` exists, and (unless discovery failed to produce any working solution, which is an acceptable outcome for a small local model per the existing `test_tiny_live_run` test's own comment) a solution and paper file exist on disk.

- [ ] **Step 3: Run the full fast test suite one more time to confirm no regressions**

Run: `pytest`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_live_smoke.py
git commit -m "test: add live smoke test for memory_bandwidth task"
```
