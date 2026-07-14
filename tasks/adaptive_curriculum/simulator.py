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
