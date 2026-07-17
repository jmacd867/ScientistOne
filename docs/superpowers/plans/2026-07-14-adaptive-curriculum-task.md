# Adaptive Curriculum Task Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `tasks/adaptive_curriculum/`, a new ScientistOne research task where the
pipeline discovers a curriculum-sequencing + spaced-repetition-scheduling policy scored
against a deterministic synthetic-learner simulator over a 20-topic NumPy→JAX→ML→GRPO
prerequisite graph.

**Architecture:** A standalone, dependency-free task plugin (`task.yaml`, `starter.py`,
`evaluator.py`, `data/*.json`), plus a new `simulator.py` module inside the task
directory implementing the deterministic learner-state mechanics. No changes to the
core `scientist_one` package are needed — this task follows the existing
`tasks/bin_packing/` plugin pattern exactly.

**Tech Stack:** Python 3.11+ stdlib only (`math`, `json`, `dataclasses`, `importlib`),
`pytest`. No new dependencies.

## Global Constraints

- Determinism: `simulator.py` must contain no randomness anywhere (no `random` module,
  no reliance on anything except Python 3.11+'s guaranteed dict insertion-order) —
  required so `scientist_one/sandbox.py`'s `run_evaluation` is reproducible and the CoE
  audit's I1 score re-verification holds.
- `data/topics.json` must be a JSON array with exactly the 20 entries below, in exactly
  this order — the order is load-bearing (it is the deterministic tie-break used by
  both `starter.py` and the evaluator's textbook-baseline policy), not just
  presentation order.
- Simulator constants (validated by direct simulation, not placeholders):
  `BASE_INTRODUCTION_GAIN = 20.0`, `BASE_REVIEW_GAIN = 40.0`,
  `MASTERY_THRESHOLD = 0.85`, `BUDGET = 3000`. Over-budget score formula:
  `score = BUDGET + 20 * (count of topics below threshold at the cap)`.
- Black-box task framing: `task.yaml`'s `description` must never mention "spaced
  repetition," "SM-2," "FSRS," or reveal the internal reward formula or the
  significance of any specific retention threshold. Use the exact text given in Task 3
  below verbatim.
- Task-plugin conventions (must match `tasks/bin_packing/` exactly): `evaluator.py`
  exposes `evaluate(solution_path: str, workdir: str) -> {"score": float, "log": str}`;
  `starter.py` exposes the policy entry point the Solver implements against
  (`choose_action`); test files load task-directory modules via
  `importlib.util.spec_from_file_location`, never via a regular package import (`tasks/`
  is not an installed Python package — it's excluded from
  `[tool.setuptools.packages.find]` in `pyproject.toml`).

---

## File Structure

- `tasks/adaptive_curriculum/simulator.py` — deterministic learner-state mechanics
  (`TopicState`, `recall_probability`, `difficulty_factor`, `introduce`, `review`,
  `run_episode`). No dependency on the rest of the task; pure functions/dataclass.
- `tasks/adaptive_curriculum/data/topics.json` — the 20-topic DAG.
- `tasks/adaptive_curriculum/data/learners.json` — the 7 learner archetypes.
- `tasks/adaptive_curriculum/task.yaml` — task metadata, black-box description, seed
  queries.
- `tasks/adaptive_curriculum/starter.py` — naive baseline `choose_action` policy (what
  the Solver starts from and improves on).
- `tasks/adaptive_curriculum/evaluator.py` — loads a solution's `choose_action`, runs it
  through all 7 learners via `simulator.run_episode`, plus a textbook baseline for
  comparison; returns the sandbox contract shape.
- `tests/test_adaptive_curriculum.py` — one file, covering simulator unit tests, data
  structural tests, starter behavior tests, evaluator tests, and the non-degeneracy
  test — mirrors `tests/test_bin_packing.py`'s single-file convention for a task plugin.

---

### Task 1: Simulator core mechanics

**Files:**
- Create: `tasks/adaptive_curriculum/simulator.py`
- Test: `tests/test_adaptive_curriculum.py` (new file)

**Interfaces:**
- Produces: `TopicState` (dataclass: `introduced: bool = False`,
  `stability: float = 0.0`, `last_touched_session: int | None = None`),
  `recall_probability(topic_state: TopicState, session: int) -> float`,
  `difficulty_factor(difficulty: float, difficulty_sensitivity: float) -> float`,
  `introduce(topic_state: TopicState, topic: dict, learner: dict, states: dict[str, TopicState], session: int) -> None`,
  `review(topic_state: TopicState, topic: dict, learner: dict, session: int) -> None`,
  `run_episode(choose_action: Callable, topics: list[dict], learner: dict, budget: int = BUDGET) -> float`,
  module constants `BASE_INTRODUCTION_GAIN`, `BASE_REVIEW_GAIN`, `MASTERY_THRESHOLD`,
  `BUDGET`. `topic` dicts have keys `id`, `name`, `prerequisites`, `difficulty`.
  `learner` dicts have keys `name`, `learning_rate`, `difficulty_sensitivity`.
- Consumes: nothing from earlier tasks (this is the first task).

- [ ] **Step 1: Write the failing tests for the pure mechanics functions**

Create `tests/test_adaptive_curriculum.py`:

```python
import importlib.util
import math
from pathlib import Path

import pytest

TASK_DIR = Path("tasks/adaptive_curriculum")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, TASK_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_simulator():
    return load_module("adaptive_curriculum_simulator_test", "simulator.py")


def test_recall_decays_over_elapsed_sessions():
    sim = load_simulator()
    ts = sim.TopicState(introduced=True, stability=2.0, last_touched_session=0)
    r1 = sim.recall_probability(ts, 1)
    r5 = sim.recall_probability(ts, 5)
    assert r1 == pytest.approx(math.exp(-1 / 2.0))
    assert 0.0 < r5 < r1


def test_recall_probability_zero_when_not_introduced():
    sim = load_simulator()
    ts = sim.TopicState()
    assert sim.recall_probability(ts, 10) == 0.0


def test_review_grows_stability():
    sim = load_simulator()
    topic = {"difficulty": 1, "prerequisites": []}
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}
    ts = sim.TopicState(introduced=True, stability=1.0, last_touched_session=0)
    before = ts.stability
    sim.review(ts, topic, learner, session=1)
    assert ts.stability > before
    assert ts.last_touched_session == 1


def test_review_is_noop_when_not_introduced():
    sim = load_simulator()
    topic = {"difficulty": 1, "prerequisites": []}
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}
    ts = sim.TopicState()
    sim.review(ts, topic, learner, session=5)
    assert ts.introduced is False
    assert ts.stability == 0.0
    assert ts.last_touched_session is None


def test_retrievability_bonus_peaks_near_recall_0_7():
    sim = load_simulator()
    topic = {"difficulty": 1, "prerequisites": []}
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}

    def gain_at_recall(recall_before):
        ts = sim.TopicState(introduced=True, stability=1.0, last_touched_session=0)
        session = -1.0 * math.log(recall_before)  # solves recall_probability(ts, session) == recall_before
        before = ts.stability
        sim.review(ts, topic, learner, session)
        return ts.stability - before

    gain_at_07 = gain_at_recall(0.7)
    gain_at_03 = gain_at_recall(0.3)
    gain_at_095 = gain_at_recall(0.95)
    assert gain_at_07 > gain_at_03
    assert gain_at_07 > gain_at_095


def test_mastery_threshold_boundary():
    sim = load_simulator()
    ts = sim.TopicState(introduced=True, stability=1.0, last_touched_session=0)
    assert sim.recall_probability(ts, 0) >= sim.MASTERY_THRESHOLD
    assert sim.recall_probability(ts, 50) < sim.MASTERY_THRESHOLD


def test_introduce_with_no_prerequisites_reaches_full_stability():
    sim = load_simulator()
    topic = {"difficulty": 1, "prerequisites": []}
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}
    ts = sim.TopicState()
    sim.introduce(ts, topic, learner, {}, session=0)
    assert ts.introduced is True
    assert ts.stability == pytest.approx(sim.BASE_INTRODUCTION_GAIN)
    assert ts.last_touched_session == 0


def test_introduce_with_weak_prerequisite_reduces_but_does_not_zero_stability():
    sim = load_simulator()
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}
    topic_no_prereq = {"difficulty": 1, "prerequisites": []}
    topic_with_prereq = {"difficulty": 1, "prerequisites": ["p"]}

    ts_no_prereq = sim.TopicState()
    sim.introduce(ts_no_prereq, topic_no_prereq, learner, {}, session=0)

    weak_prereq_state = sim.TopicState(introduced=True, stability=0.5, last_touched_session=0)
    ts_weak = sim.TopicState()
    sim.introduce(ts_weak, topic_with_prereq, learner, {"p": weak_prereq_state}, session=10)

    assert 0.1 <= ts_weak.stability < ts_no_prereq.stability


def test_introduce_is_noop_when_already_introduced():
    sim = load_simulator()
    topic = {"difficulty": 1, "prerequisites": []}
    learner = {"learning_rate": 1.0, "difficulty_sensitivity": 1.0}
    ts = sim.TopicState(introduced=True, stability=5.0, last_touched_session=2)
    sim.introduce(ts, topic, learner, {}, session=99)
    assert ts.stability == 5.0
    assert ts.last_touched_session == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive_curriculum.py -v`
Expected: FAIL/ERROR — `tasks/adaptive_curriculum/simulator.py` does not exist yet.

- [ ] **Step 3: Implement `simulator.py`**

Create `tasks/adaptive_curriculum/simulator.py`:

```python
import math
from dataclasses import dataclass

BASE_INTRODUCTION_GAIN = 20.0
BASE_REVIEW_GAIN = 40.0
MASTERY_THRESHOLD = 0.85
BUDGET = 3000


@dataclass
class TopicState:
    introduced: bool = False
    stability: float = 0.0
    last_touched_session: int | None = None


def recall_probability(topic_state: TopicState, session: int) -> float:
    if not topic_state.introduced:
        return 0.0
    elapsed = session - topic_state.last_touched_session
    return math.exp(-elapsed / topic_state.stability)


def difficulty_factor(difficulty: float, difficulty_sensitivity: float) -> float:
    return 1.0 / (1.0 + difficulty_sensitivity * (difficulty - 1) / 4)


def introduce(topic_state: TopicState, topic: dict, learner: dict,
              states: dict, session: int) -> None:
    if topic_state.introduced:
        return
    prereqs = topic["prerequisites"]
    if prereqs:
        prereq_readiness = min(recall_probability(states[p], session) for p in prereqs)
    else:
        prereq_readiness = 1.0
    df = difficulty_factor(topic["difficulty"], learner["difficulty_sensitivity"])
    topic_state.stability = max(
        0.1, BASE_INTRODUCTION_GAIN * learner["learning_rate"] * df * prereq_readiness)
    topic_state.introduced = True
    topic_state.last_touched_session = session


def review(topic_state: TopicState, topic: dict, learner: dict, session: int) -> None:
    if not topic_state.introduced:
        return
    recall_before = recall_probability(topic_state, session)
    retrievability_bonus = max(0.0, 1 - abs(recall_before - 0.7))
    df = difficulty_factor(topic["difficulty"], learner["difficulty_sensitivity"])
    topic_state.stability += (
        BASE_REVIEW_GAIN * learner["learning_rate"] * df * retrievability_bonus)
    topic_state.last_touched_session = session


def run_episode(choose_action, topics: list, learner: dict,
                budget: int = BUDGET) -> float:
    topics_by_id = {t["id"]: t for t in topics}
    topic_ids = list(topics_by_id)
    states = {tid: TopicState() for tid in topic_ids}
    topics_view = {
        tid: {"name": t["name"], "prerequisites": t["prerequisites"],
              "difficulty": t["difficulty"]}
        for tid, t in topics_by_id.items()
    }

    for session in range(budget):
        if all(recall_probability(states[tid], session) >= MASTERY_THRESHOLD
               for tid in topic_ids):
            return float(session)

        state_view = {}
        for tid in topic_ids:
            ts = states[tid]
            sst = (None if ts.last_touched_session is None
                   else session - ts.last_touched_session)
            state_view[tid] = {
                "introduced": ts.introduced,
                "estimated_retention": recall_probability(ts, session),
                "sessions_since_touched": sst,
            }

        action = choose_action(state_view, topics_view, session)
        tid = action["topic_id"]
        if action["action"] == "introduce":
            introduce(states[tid], topics_by_id[tid], learner, states, session)
        elif action["action"] == "review":
            review(states[tid], topics_by_id[tid], learner, session)

    below = sum(1 for tid in topic_ids
                if recall_probability(states[tid], budget) < MASTERY_THRESHOLD)
    return float(budget + 20 * below)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_curriculum.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/adaptive_curriculum/simulator.py tests/test_adaptive_curriculum.py
git commit -m "feat: adaptive_curriculum simulator core mechanics"
```

---

### Task 2: Topic and learner data files

**Files:**
- Create: `tasks/adaptive_curriculum/data/topics.json`
- Create: `tasks/adaptive_curriculum/data/learners.json`
- Test: `tests/test_adaptive_curriculum.py` (append)

**Interfaces:**
- Consumes: nothing (static data files).
- Produces: the exact 20-entry topic array (keys `id`, `name`, `prerequisites`,
  `difficulty`) and 7-entry learner array (keys `name`, `learning_rate`,
  `difficulty_sensitivity`) that Tasks 3 and 4 load by path.

- [ ] **Step 1: Write the failing structural tests**

Append to `tests/test_adaptive_curriculum.py`:

```python
import json


def load_topics():
    return json.loads((TASK_DIR / "data" / "topics.json").read_text())


def load_learners():
    return json.loads((TASK_DIR / "data" / "learners.json").read_text())


def test_topics_json_has_20_entries_no_duplicates():
    topics = load_topics()
    assert len(topics) == 20
    ids = [t["id"] for t in topics]
    assert len(ids) == len(set(ids))
    assert ids[0] == "numpy_arrays"
    assert ids[-1] == "grpo"


def test_topics_json_prerequisites_appear_earlier_in_array_order():
    topics = load_topics()
    seen = set()
    for t in topics:
        for p in t["prerequisites"]:
            assert p in seen, (
                f"{t['id']} depends on {p!r}, which must appear earlier in "
                "topics.json (array order is load-bearing)")
        seen.add(t["id"])


def test_topics_json_entries_have_required_fields():
    for t in load_topics():
        assert isinstance(t["name"], str) and t["name"]
        assert isinstance(t["prerequisites"], list)
        assert 1 <= t["difficulty"] <= 5


def test_learners_json_has_7_archetypes_with_positive_params():
    learners = load_learners()
    assert len(learners) == 7
    for learner in learners:
        assert isinstance(learner["name"], str) and learner["name"]
        assert learner["learning_rate"] > 0
        assert learner["difficulty_sensitivity"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive_curriculum.py -v -k "topics_json or learners_json"`
Expected: FAIL — data files do not exist yet.

- [ ] **Step 3: Create the data files**

Create `tasks/adaptive_curriculum/data/topics.json`:

```json
[
  {"id": "numpy_arrays", "name": "NumPy array basics", "prerequisites": [], "difficulty": 1},
  {"id": "numpy_indexing", "name": "NumPy indexing & slicing", "prerequisites": ["numpy_arrays"], "difficulty": 1},
  {"id": "numpy_broadcasting", "name": "NumPy broadcasting rules", "prerequisites": ["numpy_arrays"], "difficulty": 2},
  {"id": "numpy_linalg", "name": "NumPy linear algebra operations", "prerequisites": ["numpy_broadcasting"], "difficulty": 2},
  {"id": "calculus_foundations", "name": "Calculus foundations (derivatives, chain rule)", "prerequisites": [], "difficulty": 2},
  {"id": "jax_basics", "name": "JAX basics (arrays, jit)", "prerequisites": ["numpy_broadcasting"], "difficulty": 2},
  {"id": "jax_autodiff", "name": "JAX automatic differentiation (grad)", "prerequisites": ["jax_basics", "calculus_foundations"], "difficulty": 3},
  {"id": "jax_vmap", "name": "JAX vectorization with vmap", "prerequisites": ["jax_basics"], "difficulty": 2},
  {"id": "probability_foundations", "name": "Probability foundations", "prerequisites": [], "difficulty": 2},
  {"id": "linear_regression", "name": "Linear regression", "prerequisites": ["numpy_linalg", "calculus_foundations"], "difficulty": 2},
  {"id": "loss_functions", "name": "Loss functions (MSE, cross-entropy)", "prerequisites": ["probability_foundations", "linear_regression"], "difficulty": 2},
  {"id": "gradient_descent_optimizers", "name": "Gradient descent optimizers", "prerequisites": ["jax_autodiff", "loss_functions"], "difficulty": 3},
  {"id": "logistic_regression", "name": "Logistic regression", "prerequisites": ["loss_functions", "gradient_descent_optimizers"], "difficulty": 2},
  {"id": "neural_networks", "name": "Neural networks (MLPs)", "prerequisites": ["logistic_regression", "jax_vmap"], "difficulty": 3},
  {"id": "attention_transformers", "name": "Attention & transformers", "prerequisites": ["neural_networks", "jax_autodiff"], "difficulty": 4},
  {"id": "language_modeling", "name": "Language modeling", "prerequisites": ["attention_transformers"], "difficulty": 3},
  {"id": "rl_foundations", "name": "Reinforcement learning foundations", "prerequisites": ["probability_foundations"], "difficulty": 3},
  {"id": "policy_gradients", "name": "Policy gradient methods", "prerequisites": ["rl_foundations", "calculus_foundations"], "difficulty": 4},
  {"id": "ppo", "name": "Proximal Policy Optimization (PPO)", "prerequisites": ["policy_gradients"], "difficulty": 4},
  {"id": "grpo", "name": "Group Relative Policy Optimization (GRPO)", "prerequisites": ["ppo", "language_modeling"], "difficulty": 5}
]
```

Create `tasks/adaptive_curriculum/data/learners.json`:

```json
[
  {"name": "Fast learner", "learning_rate": 1.5, "difficulty_sensitivity": 0.5},
  {"name": "Above average", "learning_rate": 1.2, "difficulty_sensitivity": 0.8},
  {"name": "Average", "learning_rate": 1.0, "difficulty_sensitivity": 1.0},
  {"name": "Struggles with hard topics", "learning_rate": 1.0, "difficulty_sensitivity": 1.6},
  {"name": "Slow but steady", "learning_rate": 0.7, "difficulty_sensitivity": 0.9},
  {"name": "Quick but shaky retention", "learning_rate": 1.3, "difficulty_sensitivity": 1.2},
  {"name": "Below average", "learning_rate": 0.6, "difficulty_sensitivity": 1.3}
]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_curriculum.py -v -k "topics_json or learners_json"`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/adaptive_curriculum/data/topics.json tasks/adaptive_curriculum/data/learners.json tests/test_adaptive_curriculum.py
git commit -m "feat: adaptive_curriculum topic graph and learner archetype data"
```

---

### Task 3: task.yaml and starter policy

**Files:**
- Create: `tasks/adaptive_curriculum/task.yaml`
- Create: `tasks/adaptive_curriculum/starter.py`
- Test: `tests/test_adaptive_curriculum.py` (append)

**Interfaces:**
- Consumes: `scientist_one.tasks.base.load_task` (existing,
  `scientist_one/tasks/base.py`) — requires `task.yaml`, `starter.py`, `evaluator.py`
  all present in the directory, so `evaluator.py` gets a placeholder here and is
  filled in fully in Task 4 (this task's `load_task` test only checks `task.yaml` and
  `starter.py` content; it does not exercise `evaluator.py`).
- Produces: `starter.py`'s `choose_action(state: dict, topics: dict, session: int) -> dict`,
  matching the policy interface every future Solver-generated solution must implement.
  `topics` here is the `topics_view` dict `run_episode` builds in Task 1 (keys `name`,
  `prerequisites`, `difficulty`), NOT the raw `topics.json` array.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_curriculum.py`:

```python
from scientist_one.tasks.base import load_task


def load_starter():
    return load_module("adaptive_curriculum_starter_test", "starter.py")


def test_task_yaml_loads_with_correct_metadata():
    task = load_task(TASK_DIR)
    assert task.name == "adaptive_curriculum"
    assert task.metric_direction == "lower"
    assert len(task.seed_queries) > 0


def test_starter_introduces_first_topic_when_none_introduced():
    starter = load_starter()
    topics = {
        "a": {"name": "A", "prerequisites": [], "difficulty": 1},
        "b": {"name": "B", "prerequisites": [], "difficulty": 1},
    }
    state = {
        "a": {"introduced": False, "estimated_retention": 0.0, "sessions_since_touched": None},
        "b": {"introduced": False, "estimated_retention": 0.0, "sessions_since_touched": None},
    }
    action = starter.choose_action(state, topics, 0)
    assert action == {"action": "introduce", "topic_id": "a"}


def test_starter_reviews_longest_untouched_when_all_introduced():
    starter = load_starter()
    topics = {
        "a": {"name": "A", "prerequisites": [], "difficulty": 1},
        "b": {"name": "B", "prerequisites": [], "difficulty": 1},
    }
    state = {
        "a": {"introduced": True, "estimated_retention": 0.5, "sessions_since_touched": 3},
        "b": {"introduced": True, "estimated_retention": 0.5, "sessions_since_touched": 5},
    }
    action = starter.choose_action(state, topics, 10)
    assert action == {"action": "review", "topic_id": "b"}


def test_starter_tie_break_uses_topics_dict_order():
    starter = load_starter()
    topics = {
        "a": {"name": "A", "prerequisites": [], "difficulty": 1},
        "b": {"name": "B", "prerequisites": [], "difficulty": 1},
    }
    state = {
        "a": {"introduced": True, "estimated_retention": 0.5, "sessions_since_touched": 4},
        "b": {"introduced": True, "estimated_retention": 0.5, "sessions_since_touched": 4},
    }
    action = starter.choose_action(state, topics, 10)
    assert action == {"action": "review", "topic_id": "a"}


def test_starter_runs_full_episode_on_real_data_without_crashing():
    sim = load_simulator()
    starter = load_starter()
    topics = load_topics()
    learner = load_learners()[0]
    score = sim.run_episode(starter.choose_action, topics, learner, budget=200)
    assert score > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive_curriculum.py -v -k "task_yaml or starter"`
Expected: FAIL — `task.yaml` and `starter.py` do not exist yet (and `evaluator.py`
does not exist yet either, so `load_task` will raise `FileNotFoundError`; that's
expected until Task 4's placeholder lands in the next step).

- [ ] **Step 3: Create `task.yaml`, `starter.py`, and a placeholder `evaluator.py`**

Create `tasks/adaptive_curriculum/task.yaml`:

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

Create `tasks/adaptive_curriculum/starter.py`:

```python
def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session.

    state: {topic_id: {"introduced": bool, "estimated_retention": float,
                        "sessions_since_touched": int | None}}
    topics: {topic_id: {"name": str, "prerequisites": list[str], "difficulty": float}}
    session: current session number (0-indexed)

    Returns: {"action": "introduce" | "review", "topic_id": str}

    Baseline: introduce topics in topics.json order; once all are
    introduced, review whichever has gone longest without being touched
    (ties broken by topics.json order). Ignores prerequisite-readiness —
    deliberately leaves room to improve.
    """
    topic_ids = list(topics)
    for tid in topic_ids:
        if not state[tid]["introduced"]:
            return {"action": "introduce", "topic_id": tid}

    target = max(
        topic_ids,
        key=lambda tid: (state[tid]["sessions_since_touched"], -topic_ids.index(tid)),
    )
    return {"action": "review", "topic_id": target}
```

Create a placeholder `tasks/adaptive_curriculum/evaluator.py` (filled in fully in
Task 4 — this exists now only so `load_task` finds all three required files):

```python
def evaluate(solution_path: str, workdir: str) -> dict:
    raise NotImplementedError("implemented in Task 4")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_curriculum.py -v -k "task_yaml or starter"`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tasks/adaptive_curriculum/task.yaml tasks/adaptive_curriculum/starter.py tasks/adaptive_curriculum/evaluator.py tests/test_adaptive_curriculum.py
git commit -m "feat: adaptive_curriculum task.yaml and naive starter policy"
```

---

### Task 4: Evaluator

**Files:**
- Modify: `tasks/adaptive_curriculum/evaluator.py` (replace Task 3's placeholder)
- Test: `tests/test_adaptive_curriculum.py` (append)

**Interfaces:**
- Consumes: `simulator.run_episode`, `simulator.BUDGET` (Task 1);
  `data/topics.json`, `data/learners.json` (Task 2).
- Produces: `evaluate(solution_path: str, workdir: str) -> {"score": float, "log": str}`
  — the sandbox contract `scientist_one/sandbox.py:run_evaluation` calls via
  `mod.evaluate(sys.argv[2], sys.argv[3])`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_adaptive_curriculum.py`:

```python
import re


def load_evaluator():
    return load_module("adaptive_curriculum_evaluator_test", "evaluator.py")


def test_evaluator_starter_produces_bounded_nontrivial_score(tmp_path):
    ev = load_evaluator()
    result = ev.evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    assert 10.0 < result["score"] < 3000.0 + 20 * 20
    assert "baseline_score=" in result["log"]


def test_evaluator_log_reports_sensible_distinct_baseline_score(tmp_path):
    ev = load_evaluator()
    result = ev.evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    match = re.search(r"baseline_score=([\d.]+)", result["log"])
    assert match is not None
    baseline_score = float(match.group(1))
    assert baseline_score > 0
    assert baseline_score != result["score"]  # starter and textbook are different policies


def test_evaluator_obviously_bad_solution_scores_far_worse_than_starter(tmp_path):
    bad = tmp_path / "bad_solution.py"
    bad.write_text(
        "def choose_action(state, topics, session):\n"
        "    tid = next(iter(topics))\n"
        "    if not state[tid]['introduced']:\n"
        "        return {'action': 'introduce', 'topic_id': tid}\n"
        "    return {'action': 'review', 'topic_id': tid}\n"
    )
    ev = load_evaluator()
    starter_result = ev.evaluate(str(TASK_DIR / "starter.py"), str(tmp_path))
    bad_result = ev.evaluate(str(bad), str(tmp_path))
    assert bad_result["score"] > starter_result["score"] * 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_adaptive_curriculum.py -v -k evaluator`
Expected: FAIL — `evaluator.py` still raises `NotImplementedError`.

- [ ] **Step 3: Implement `evaluator.py`**

Replace `tasks/adaptive_curriculum/evaluator.py`:

```python
import importlib.util
import json
import statistics
from pathlib import Path


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.choose_action


def _load_simulator():
    spec = importlib.util.spec_from_file_location(
        "adaptive_curriculum_simulator", Path(__file__).parent / "simulator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data():
    data_dir = Path(__file__).parent / "data"
    topics = json.loads((data_dir / "topics.json").read_text())
    learners = json.loads((data_dir / "learners.json").read_text())
    return topics, learners


def _textbook_baseline(state: dict, topics: dict, session: int) -> dict:
    topic_ids = list(topics)
    for tid in topic_ids:
        if state[tid]["introduced"]:
            continue
        prereqs = topics[tid]["prerequisites"]
        if all(state[p]["estimated_retention"] >= 0.8 for p in prereqs):
            return {"action": "introduce", "topic_id": tid}

    introduced = [tid for tid in topic_ids if state[tid]["introduced"]]
    target = min(
        introduced,
        key=lambda tid: (state[tid]["estimated_retention"], topic_ids.index(tid)),
    )
    return {"action": "review", "topic_id": target}


def evaluate(solution_path: str, workdir: str) -> dict:
    policy = _load_solution(solution_path)
    sim = _load_simulator()
    topics, learners = _load_data()

    lines = []
    discovered_scores = []
    baseline_scores = []
    for learner in learners:
        d_score = sim.run_episode(policy, topics, learner)
        b_score = sim.run_episode(_textbook_baseline, topics, learner)
        discovered_scores.append(d_score)
        baseline_scores.append(b_score)
        lines.append(f"{learner['name']}: discovered={d_score:.1f} baseline={b_score:.1f}")

    discovered_score = statistics.mean(discovered_scores)
    baseline_score = statistics.mean(baseline_scores)
    lines.append(f"discovered_score={discovered_score:.4f}")
    lines.append(f"baseline_score={baseline_score:.4f}")
    return {"score": round(discovered_score, 4), "log": "\n".join(lines)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_adaptive_curriculum.py -v`
Expected: All tests PASS (full file, all tasks so far).

- [ ] **Step 5: Commit**

```bash
git add tasks/adaptive_curriculum/evaluator.py tests/test_adaptive_curriculum.py
git commit -m "feat: adaptive_curriculum evaluator with textbook baseline comparison"
```

---

### Task 5: Non-degeneracy proof and calibration documentation

**Files:**
- Modify: `tests/test_adaptive_curriculum.py` (append)
- Modify: `tasks/adaptive_curriculum/simulator.py` (add a module docstring documenting
  the calibration result — no logic changes; constants were already validated in
  Task 1)

**Interfaces:**
- Consumes: `simulator.run_episode` (Task 1), `starter.choose_action` (Task 3),
  `data/topics.json`, `data/learners.json` (Task 2).
- Produces: nothing new — this task is a verification and documentation step
  confirming the whole task (as built across Tasks 1-4) has real signal for Discovery
  to search over, using the population-level comparison across all 7 learners (not a
  single learner), as the design spec requires.

- [ ] **Step 1: Write the failing non-degeneracy test**

Append to `tests/test_adaptive_curriculum.py`:

```python
def test_population_non_degeneracy_bad_policy_much_worse_than_naive():
    """Confirms the benchmark responds to policy quality across the full
    7-learner population (not just one learner) -- an obviously-bad policy
    that only ever touches a single topic must score far worse than the
    naive starter, on average across all archetypes."""
    sim = load_simulator()
    starter = load_starter()
    topics = load_topics()
    learners = load_learners()

    def bad_policy(state, topics, session):
        tid = next(iter(topics))
        if not state[tid]["introduced"]:
            return {"action": "introduce", "topic_id": tid}
        return {"action": "review", "topic_id": tid}

    naive_scores = [sim.run_episode(starter.choose_action, topics, l) for l in learners]
    bad_scores = [sim.run_episode(bad_policy, topics, l) for l in learners]

    naive_mean = sum(naive_scores) / len(naive_scores)
    bad_mean = sum(bad_scores) / len(bad_scores)
    assert bad_mean > naive_mean * 2


def test_naive_starter_does_not_hit_budget_cap_for_any_learner():
    """Regression guard for the calibration bug found before implementation:
    the original constants made every policy hit the BUDGET cap with no
    differentiation. This confirms the corrected constants keep naive's
    per-learner scores below the cap (score < BUDGET means mastery was
    actually reached, not just approached)."""
    sim = load_simulator()
    starter = load_starter()
    topics = load_topics()
    learners = load_learners()

    scores = [sim.run_episode(starter.choose_action, topics, l) for l in learners]
    assert all(score < sim.BUDGET for score in scores)
```

- [ ] **Step 2: Run tests to verify they fail or pass**

Run: `pytest tests/test_adaptive_curriculum.py -v -k "non_degeneracy or budget_cap"`
Expected: Both PASS immediately — the constants were already validated by direct
simulation while writing this plan (see the spec's "Calibration result" section), so
no further tuning is needed. If either test unexpectedly fails, do not proceed:
re-run the same before/after-simulation approach used to derive the current constants
(`BASE_INTRODUCTION_GAIN`, `BASE_REVIEW_GAIN` in `simulator.py`, `BUDGET`) — adjust and
re-test until both pass, then update the spec's "Calibration result" section to match
whatever changed.

- [ ] **Step 3: Document the calibration result in `simulator.py`**

Add a module docstring at the top of `tasks/adaptive_curriculum/simulator.py` (above
the existing `import math` line):

```python
"""Deterministic synthetic-learner simulator for the adaptive_curriculum task.

Constants below were validated by direct simulation before implementation
(see docs/superpowers/specs/2026-07-14-adaptive-curriculum-task-design.md,
"Calibration result"): the first-drafted values made mastery unreachable for
every policy tried. BASE_INTRODUCTION_GAIN and BASE_REVIEW_GAIN were scaled
20x and BUDGET scaled 10x from that draft to produce a well-behaved score
range (naive baseline: 60-180 sessions across the 7 learner archetypes, no
policy hitting the cap).
"""
```

- [ ] **Step 4: Run the full test suite**

Run: `pytest tests/test_adaptive_curriculum.py -v`
Expected: All tests PASS.

Run: `pytest` (full suite)
Expected: All existing tests still PASS — this task adds a new task plugin and
touches no shared `scientist_one/` code, so no regressions are possible, but confirm
anyway.

- [ ] **Step 5: Commit**

```bash
git add tasks/adaptive_curriculum/simulator.py tests/test_adaptive_curriculum.py
git commit -m "test: adaptive_curriculum population-level non-degeneracy proof"
```
