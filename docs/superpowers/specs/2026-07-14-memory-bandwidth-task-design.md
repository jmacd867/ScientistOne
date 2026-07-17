# Memory Bandwidth Task — Design

**Date:** 2026-07-14
**Status:** Approved (pending implementation plan)

## Goal

Add a ScientistOne research task, `tasks/memory_bandwidth/`, where the pipeline
discovers techniques that maximize achieved GPU memory bandwidth (GB/s) on this
machine's NVIDIA GB10 (DGX Spark). The benchmark is a STREAM-style triad kernel
(`c = a + scalar * b`) run on CUDA tensors already resident in GPU memory — the
standard proxy for the bandwidth ceiling that also limits LLM token-generation
speed on this hardware. Score is achieved GB/s; `metric_direction: higher`.

**Non-goals:** CPU-side (Grace ARM) bandwidth, host↔device transfer bandwidth,
persistent system/driver tuning (NUMA, huge pages, memory clocks), anything
requiring root or modifying machine state outside the task's own subprocess.

## Problem: the solver is hardcoded to stdlib-only

[`solver.py`](../../../scientist_one/discovery/solver.py)'s system prompt tells
the solving LLM to "use only the standard library." There is no CUDA access from
the standard library, so no GPU task can work against the framework as it stands.
This blocks the goal directly, so fixing it is in scope (not a detour).

## Decisions made

| Decision | Choice |
|---|---|
| Memory path | GPU/CUDA (GB10 unified memory), not CPU |
| Solution shape | Pure algorithm/kernel choice — solver has no access to system/driver tuning |
| Benchmark kernel | STREAM-style triad: `c = a + scalar * b` |
| Library access | New `allowed_imports` field on `TaskSpec`; solver prompt lists task-declared extra libraries instead of "stdlib only" |
| GPU library | `torch`, added as a new `gpu` optional-dependency extra in `pyproject.toml` |
| Anti-gaming | Evaluator asserts output dtype/shape match input before scoring, so a solution can't cheat bandwidth by silently dropping precision |

## Framework change

`scientist_one/tasks/base.py`: add `allowed_imports: list[str] = []` to
`TaskSpec`.

`scientist_one/discovery/solver.py`: `_SYSTEM` becomes a function of the task —
when `task.allowed_imports` is non-empty, the prompt says "You may use the
standard library plus: `torch`" instead of "use only the standard library."
When empty, behavior is unchanged (existing tasks like `bin_packing` are
unaffected).

`pyproject.toml`: add `[project.optional-dependencies] gpu = ["torch"]`.
Installed locally via `pip install -e ".[dev,gpu]"` (not a base dependency —
most tasks don't need it, and it's a large, hardware-specific install).

## Task files

### `task.yaml`
- `metric_direction: higher`
- `allowed_imports: [torch]`
- `timeout_s: 120` (override of the config default of 60s — CUDA context init
  plus timing across multiple tensor sizes needs headroom)
- `seed_queries` aimed at the Problem Investigator's literature retrieval:
  STREAM benchmark methodology, roofline model / memory-bound kernel
  optimization, CUDA kernel fusion to reduce memory traffic, unified memory
  bandwidth on Grace-Blackwell-class SoCs, avoiding intermediate tensor
  allocations in elementwise ops.

### `starter.py`
```python
def triad(a: torch.Tensor, b: torch.Tensor, scalar: float) -> torch.Tensor:
    """Compute c = a + scalar * b for CUDA tensors already on the GPU."""
    return a + scalar * b
```
Naive baseline: this allocates an intermediate tensor for `scalar * b` plus the
output tensor — two avoidable allocations/passes a tuned kernel can remove.

### `evaluator.py`
For each of several fixed tensor sizes (large enough to be bandwidth-bound
rather than cache-bound, sized to comfortably finish within the 120s budget):
1. Allocate `a`, `b` on CUDA with a fixed seed and dtype (`float32`).
2. Call the solution's `triad` once; assert result `dtype`/`shape` match the
   inputs, then check numerical correctness against a reference computation
   within tolerance. Both checks must pass before timing — this is what
   prevents a solution from gaming bandwidth by silently truncating precision
   or shape.
3. Warm up (a few untimed calls), then time N repetitions using
   `torch.cuda.Event` start/end + `torch.cuda.synchronize()`.
4. Compute GB/s from the *fixed, known* bytes-per-call (`3 * n * elem_size`) —
   not from whatever the solution's tensor reports — so the number reflects
   actual work done at the contracted size/dtype.
5. Score = mean GB/s across sizes. Log includes per-size GB/s for readability.

If CUDA is unavailable, the evaluator raises; the sandbox's existing exception
handling in [`sandbox.py`](../../../scientist_one/sandbox.py) turns that into
`ok=False` with the stderr as the log — no special-casing needed there.

## Testing

- Existing test suite runs with the fake LLM backend and must stay green;
  `solver.py`'s prompt-building logic gets a unit test covering both the
  empty-`allowed_imports` (unchanged behavior) and non-empty cases.
- The `memory_bandwidth` evaluator is exercised directly (not via a live
  Ollama run) with a hand-written correct/incorrect solution file, verifying
  score direction and that a shape/dtype-violating "solution" is rejected.
  This requires the local GPU, which is a different dependency than the
  existing `live` marker (requires a running Ollama server) — so it gets its
  own new `gpu` pytest marker, excluded from the default `addopts` alongside
  `live`, run explicitly with `pytest -m gpu`.
