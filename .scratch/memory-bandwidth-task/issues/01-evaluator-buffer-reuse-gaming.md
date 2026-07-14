Status: needs-triage

# `memory_bandwidth` evaluator: identity-memoization gaming vector

The `evaluate()` function in `tasks/memory_bandwidth/evaluator.py` reuses the
same `a`/`b` tensors (and the fixed `scalar`) across the one-shot correctness
check, the warmup calls, and all `_REPS` timed calls for a given size. A
solution that memoizes its result keyed on `(id(a), id(b), scalar)` — or any
similarly narrow cache — would pass the single correctness check, then return
the cached tensor during every timed rep with near-zero real memory traffic,
inflating its measured GB/s without doing the claimed work.

This is out of scope for the anti-gaming checks the task was built with
(shape/dtype/device mismatch, verified in
[docs/superpowers/plans/2026-07-14-memory-bandwidth-task.md](../../../docs/superpowers/plans/2026-07-14-memory-bandwidth-task.md)),
and the discovery pipeline's integrity auditor (`scientist_one/discovery/auditor.py`)
is a semantic backstop that should flag "gaming the metric instead of solving
the problem." But it's a real gap in the evaluator itself, worth closing
directly rather than relying solely on an LLM judge.

**Suggested fix:** re-fill `b` with fresh random values between the warmup
phase and the timed phase (and/or between reps). Legitimate STREAM-style
benchmarks already do this; it breaks identity-based memoization without
changing the measured memory-bandwidth work.

Flagged during the final whole-branch review of the `memory_bandwidth` task
(2026-07-14); not a blocker for that merge.
