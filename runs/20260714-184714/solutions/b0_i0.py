import math

def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using an Adaptive Spaced Repetition policy.

    The strategy uses a risk-based approach based on the Ebbinghaus Forgetting Curve model. 
    It estimates when retention is likely to fall below a critical threshold (e.g., 0.7).
    We prioritize introducing new topics that have their prerequisites met, and then 
    prioritize reviewing existing topics whose predicted decay threatens stability.

    Rationale: Minimize the time-to-mastery by balancing acquisition of new knowledge 
    with maintenance of current knowledge based on estimated retention levels.
    """

    # Constants for our heuristic model
    RETENTION_THRESHOLD = 0.75  # Target minimum acceptable memory strength
    RISK_TOLERANCE = 0.3        # How much 'decay' we allow before triggering review (1 - threshold)
    DECAY_CONSTANT_BASE = 0.2   # Base decay rate per session for all topics

    topic_ids = list(strings := sorted(topics.keys()))
    unintroduced = [tid for tid in topic_ids if not state[tid]["introduced"]]
    
    def get_priority_score(tid):
        """Calculate a priority score based on predicted decay risk."""
        s = state[tid]
        # If never touched, it's effectively at 0 retention or just introduced.
        if s["sessions_since_touched"] is None:
            return -float('inf') # Lowest priority for unintroduced in review phase

        # We use an exponential decay model approximation: R(t) = e^(-beta * t)
        # where 't' is sessions since last touch. 
        # Since we don't see the true beta, we estimate it using state['estimated_retention'].
        # If estimated_retention was recorded after a session N steps ago:
        # R(now) = R(old) * exp(-beta * delta_t). 
        # However, since 'state' provides current estimation and time elapsed (sessions_since):
        # We assume the state['estimated_retention'] is updated per-session.
        
        days_passed = s["sessions_since_touched"]
        current_r = s["estimated_retention"]
        
        # Risk calculation: How close are we to falling below threshold? 
        # We use the sessions since touched as a proxy for time elapsed 't'.
        # A higher score means more urgent review.
        return days_passed * (1.0 - current_r)

    def is_prerequisite_met(tid):
        """Check if all prerequisites of tid are already introduced."""
        for prereq in topics[tid]["prerequisites"]:
            if not state[prereq]["introduced"]:
                return False
        return True

    # 1. Try to introduce new available topics (Prerequisite-aware)
    available_to_introduce = [tid for tid in unintroduced if is_prerequisite_met(tid)]
    
    if available_to_introduce:
        # Pick the first one that satisfies prerequisites (maintaining deterministic order)
        return {"action": "introduce", "topic_id": available_to_introduce[0]}

    # 2. If no new topics can be introduced, check if we MUST review due to decay risk
    introduced = [tid for tid in topic_ids if state[tid]["introduced"]]
    if not introduced:
        # This case handles the edge case where nothing is introduced yet (shouldn't happen with logic above)
        return {"action": "introduce", "topic_id": topic_ids[0]}

    review_candidates = []
    for tid in introduced:
        s = state[tid]
        # Risk level based on decay and time elapsed. 
        # We boost score if sessions since touched is high OR retention estimate is low.
        risk_score = get_priority_score(tid)
        review_candidates.append((risk_score, tid))

    if review_candidates:
        # Sort by risk descending (highest risk first), then tie-break with topic order index
        # We use -topic_ids.index to ensure deterministic behavior matching original baseline for ties
        review_candidates.sort(key=lambda x: (-x[0], topic_ids.index(x[1])))
        return {"action": "review", "topic_id": review_candidates[0][1]}

    # Fallback (should not be reached)
    target = max(topic_ids, key=lambda tid: -topic_ids.index(tid))
    return {"action": "introduce" if state[target]["introduced"] == False else "review", "topic_id": target}


def strings(iterable):
    """Helper for type hinting compatibility."""
    if hasattr(iterable, '__iter__'):
        yield from iterable

