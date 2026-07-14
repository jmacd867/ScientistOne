import importlib.util
import json
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
