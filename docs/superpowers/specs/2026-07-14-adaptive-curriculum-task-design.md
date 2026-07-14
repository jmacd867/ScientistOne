# Adaptive Curriculum Task — Design

**Date:** 2026-07-14
**Status:** Approved (pending implementation plan)

## Goal

Add a ScientistOne research task, `tasks/adaptive_curriculum/`, where the
pipeline discovers a teaching/review-scheduling policy that minimizes the
number of study sessions needed to bring a population of simulated learners
to mastery across a real prerequisite-linked curriculum (NumPy → JAX →
mathematically-founded ML → LLMs → GRPO). The policy must decide, each
session, whether to introduce a new topic or review an already-introduced
one — combining curriculum sequencing and spaced-repetition scheduling into
one decision, scored against a deterministic synthetic-student simulator.

This is the first of two planned sub-projects exploring "how LLMs can impact
education" (see `docs/superpowers/specs/` for context from brainstorming).
The second — a knowledge-vault / concept-connection system — is out of scope
here and will get its own design later.

**Non-goals:** real student data or human-in-the-loop evaluation (breaks the
determinism the whole pipeline depends on); a knowledge-graph/"meaningful
connections" system (separate follow-on project); per-task LLM model
overrides (current global `config.yaml` `models:` section is reused as-is).

## Why this needs a genuinely black-box task framing

gemma4:26b almost certainly has prior knowledge of real spaced-repetition
algorithms (SM-2, FSRS) and the "desirable difficulty" research behind them.
That prior knowledge is fine and expected — a human researcher would bring
the same domain background — but if `task.yaml`'s description reveals the
simulator's internal reward formula or names known algorithms, there is no
discovery happening: the Solver just recites a known structure without ever
needing the solve→evaluate→feedback loop that is this whole pipeline's
point. The Solver never sees `evaluator.py`'s source (existing
architecture), so the real lever is what we choose to put in
`task.yaml`'s `description` and the state given to the policy function.
**Commitment:** the task description states only the observable interface
and the optimization goal — never the internal formulas, the retention
threshold's significance, or algorithm names like "spaced repetition,"
"SM-2," or "FSRS." See the exact draft text in "task.yaml" below.

## Topic graph

20 topics, a real DAG (not a linear chain) reflecting genuine prerequisite
structure on the way to GRPO. Difficulty is 1 (easiest) to 5 (hardest).

| id | prerequisites | difficulty |
|---|---|---|
| `numpy_arrays` | — | 1 |
| `numpy_indexing` | `numpy_arrays` | 1 |
| `numpy_broadcasting` | `numpy_arrays` | 2 |
| `numpy_linalg` | `numpy_broadcasting` | 2 |
| `calculus_foundations` | — | 2 |
| `jax_basics` | `numpy_broadcasting` | 2 |
| `jax_autodiff` | `jax_basics`, `calculus_foundations` | 3 |
| `jax_vmap` | `jax_basics` | 2 |
| `probability_foundations` | — | 2 |
| `linear_regression` | `numpy_linalg`, `calculus_foundations` | 2 |
| `loss_functions` | `probability_foundations`, `linear_regression` | 2 |
| `gradient_descent_optimizers` | `jax_autodiff`, `loss_functions` | 3 |
| `logistic_regression` | `loss_functions`, `gradient_descent_optimizers` | 2 |
| `neural_networks` | `logistic_regression`, `jax_vmap` | 3 |
| `attention_transformers` | `neural_networks`, `jax_autodiff` | 4 |
| `language_modeling` | `attention_transformers` | 3 |
| `rl_foundations` | `probability_foundations` | 3 |
| `policy_gradients` | `rl_foundations`, `calculus_foundations` | 4 |
| `ppo` | `policy_gradients` | 4 |
| `grpo` | `ppo`, `language_modeling` | 5 |

Stored as `tasks/adaptive_curriculum/data/topics.json` — a JSON **array**, all 20
entries in exactly the order listed in the table above (this order is load-
bearing: it is the tie-break used by both the starter and the textbook
baseline policy below, not just presentation order):
```json
[
  {"id": "numpy_arrays", "name": "NumPy array basics", "prerequisites": [], "difficulty": 1},
  {"id": "numpy_indexing", "name": "NumPy indexing & slicing", "prerequisites": ["numpy_arrays"], "difficulty": 1},
  ...
]
```
(one entry per row of the table above, in table order, `name` filled in
descriptively — the table already fully specifies every entry; nothing here
is left open).

## Synthetic learner population

7 fixed archetypes (no randomness — determinism is a hard requirement),
varying exactly two parameters to avoid an over-parameterized, hard-to-
calibrate model:

| archetype | `learning_rate` | `difficulty_sensitivity` |
|---|---|---|
| Fast learner | 1.5 | 0.5 |
| Above average | 1.2 | 0.8 |
| Average (baseline) | 1.0 | 1.0 |
| Struggles with hard topics | 1.0 | 1.6 |
| Slow but steady | 0.7 | 0.9 |
| Quick but shaky retention | 1.3 | 1.2 |
| Below average | 0.6 | 1.3 |

Stored as `tasks/adaptive_curriculum/data/learners.json`.

A policy is scored by running it independently through all 7 learners and
averaging — a policy that only works for "Fast learner" and fails
"Struggles with hard topics" does not win on the population-average score.

## Simulator mechanics (`tasks/adaptive_curriculum/simulator.py`)

Fully deterministic (no RNG anywhere) — required so `run_evaluation`'s
re-run-and-compare (and the CoE audit's I1 score-verification) hold.

**Per-topic per-learner state:** `introduced: bool`, `stability: float`
(undefined/0 until introduced), `last_touched_session: int | None`.

**Recall probability** (standard exponential forgetting curve, the same
functional form used in SM-2/FSRS):
```
recall_probability(topic, session) =
    0.0                                            if not introduced
    exp(-(session - last_touched_session) / stability)   otherwise
```

**Difficulty scaling per learner:**
```
difficulty_factor(topic, learner) =
    1.0 / (1.0 + learner.difficulty_sensitivity * (topic.difficulty - 1) / 4)
```
At difficulty 1, this is ~1.0 regardless of sensitivity. At difficulty 5, a
high-sensitivity learner's factor drops sharply; a low-sensitivity learner's
barely moves.

**Introducing a not-yet-introduced topic:**
```
prereq_readiness = min(recall_probability(p, session) for p in topic.prerequisites)
                   (1.0 if no prerequisites)
new_stability = max(0.1, BASE_INTRODUCTION_GAIN * learner.learning_rate
                          * difficulty_factor(topic, learner) * prereq_readiness)
```
`BASE_INTRODUCTION_GAIN = 1.0`. Introducing with weak prerequisites is
allowed (not blocked) but starts the topic with low stability — "you can
start calculus with shaky algebra, it'll just be harder to make it stick."
Re-introducing an already-introduced topic is a no-op (wastes the session).
The `0.1` floor keeps `stability` strictly positive (it's a divisor above).

**Reviewing an already-introduced topic:**
```
recall_before = recall_probability(topic, session)
retrievability_bonus = max(0, 1 - abs(recall_before - 0.7))
new_stability = stability + BASE_REVIEW_GAIN * learner.learning_rate
                             * difficulty_factor(topic, learner) * retrievability_bonus
```
`BASE_REVIEW_GAIN = 2.0`. This is "desirable difficulty": review timed near
~70% recall gives the biggest stability gain; reviewing something fresh
(~100%) or already forgotten (~0%) gives little. Reviewing a
not-yet-introduced topic is a no-op (wastes the session).

Both actions set `last_touched_session = session` on success (no-ops don't).

**Mastery & scoring:** `MASTERY_THRESHOLD = 0.85`. A learner's
`sessions_to_mastery` is the first session at which every topic's
`recall_probability >= MASTERY_THRESHOLD` simultaneously, capped at
`BUDGET = 300`. If not reached by the cap:
`score = BUDGET + 20 * (count of topics below threshold at the cap)` — keeps
the signal smooth for policies that get close but don't finish, instead of a
flat "failed" score indistinguishable from a much worse policy.

**Note on calibration:** `BASE_INTRODUCTION_GAIN`, `BASE_REVIEW_GAIN`, and
the archetype parameters are a reasoned starting point, not empirically
validated numbers — the implementation plan must include a calibration step
(run a few hand-written policies, including the naive starter and a
"review-at-0.7" policy, and adjust constants if the score range is either
trivially easy or effectively unreachable for every policy tried).

## Policy interface (`tasks/adaptive_curriculum/starter.py`)

```python
def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session.

    state: {topic_id: {"introduced": bool, "estimated_retention": float,
                        "sessions_since_touched": int | None}}
    topics: {topic_id: {"name": str, "prerequisites": list[str], "difficulty": float}}
    session: current session number (0-indexed)

    Returns: {"action": "introduce" | "review", "topic_id": str}
    """
```
`estimated_retention` is exactly `recall_probability` (0.0 if not
introduced) — a legible, learner-app-realistic signal. The policy is never
given `stability` directly, nor told that 0.7 is a special value; it must
infer that from how its score changes across Discovery iterations.

**Starter (naive) baseline:** iterate topics in `topics.json` list order;
introduce the first not-yet-introduced one; if all introduced, review
whichever has gone longest without being touched (ties broken by
`topics.json` order). Ignores prerequisite-readiness — deliberately leaves
room to improve.

## Task description (`tasks/adaptive_curriculum/task.yaml`)

```yaml
name: adaptive_curriculum
description: >
  Discover a policy for teaching a curriculum of interdependent topics to a
  population of simulated learners. Implement choose_action(state, topics,
  session), which each session decides whether to introduce a new topic or
  review an already-introduced one. Each learner has a private, evolving
  retention level per topic that changes based on your choices and the
  passage of time; you cannot observe the underlying model directly, only
  the state dict provided each call and feedback (score, log) after a full
  run. Your goal: minimize the average number of sessions needed until
  every topic is retained above threshold, across a population of learners
  with varying ability.
metric_direction: lower
seed_queries:
  - spaced repetition scheduling algorithm
  - curriculum sequencing prerequisite learning
  - mastery learning optimal review timing
  - forgetting curve memory retention model
```
(Seed queries name real research areas for the Problem Investigator's
literature grounding — this is separate from the black-box framing above,
which governs what the *Solver* is told, not what papers get retrieved.)

## Evaluator (`tasks/adaptive_curriculum/evaluator.py`)

```
def evaluate(solution_path, workdir) -> {"score": float, "log": str}:
    load policy from solution_path (importlib, same pattern as bin_packing)
    load topics.json, learners.json
    for each of the 7 learners (independently — a fresh episode per
            learner, session clock restarts at 0, no state or history
            carries over between learners; the same choose_action
            function is simply called again from a clean slate):
        run simulator.run_episode(policy.choose_action, topics, learner, budget=300)
        record sessions_to_mastery (or the smooth over-budget score)
    discovered_score = mean(the 7 results)
    run the same 7-learner loop with a fixed textbook baseline policy to
    get baseline_score: each session, introduce the first topics.json-order
    topic whose prerequisites are all >= 0.8 retained and that isn't yet
    introduced; if none qualify, review the introduced topic with the
    lowest current retention (ties broken by topics.json order)
    return {"score": discovered_score,
            "log": "<per-learner scores>\nbaseline_score=<...>\n..."}
```
`baseline_score` lives in the log string (not a separate return field) —
the existing `EvalOutcome`/sandbox contract only carries `score` and `log`,
and `numbers_in_payload` already scans log strings for numbers, so a paper
claim like "improved N% over the textbook baseline" traces correctly through
the existing evidence-chain machinery with no pipeline changes needed.

## Testing

- Unit tests directly on `simulator.py` mechanics (no LLM, no Ollama):
  recall decays correctly over elapsed sessions; reviewing grows stability;
  retrievability bonus peaks at recall ≈ 0.7; mastery detection fires
  correctly at the threshold; introducing with weak/no prerequisites
  produces reduced (not zero, not crashing) stability growth.
- A non-degeneracy test: a hand-written policy that reviews each topic as
  its retention crosses ~0.75 must score meaningfully better than the naive
  starter baseline on the same learner population — confirms the benchmark
  has real signal for Discovery to search over.
- `load_task` on the new directory (structural test, matches the pattern
  used for `bin_packing`).
- Evaluator test confirming the starter produces a valid, bounded (not
  infinite, not immediately trivial) score, and that the textbook baseline
  computed inside `evaluate()` also produces a sensible, different score.

## Out of scope / follow-on work

- **Knowledge vault / concept-connection system** — a separate project;
  "meaningful connections" between concepts is a different core problem
  (graph quality, not scheduling optimality) needing its own design.
- **Per-task LLM model override** — this task reuses the pipeline's global
  `config.yaml` `models:` section (`gemma4:26b` reasoning / `gemma4:12b`
  judging), same as every other task. A task-level override would be a
  real but separate feature if a future task needs different models.
