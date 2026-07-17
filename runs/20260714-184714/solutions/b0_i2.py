import math

def choose_action(state: dict, topics: dict, session: int) -> dict:
    """
    An adaptive Spaced Repetition policy using a risk-based approach to decay estimation.
    The agent prioritizes introducing new content based on prerequisites and 
    triggers reviews when the estimated retention of an introduced topic is predicted 
    to fall below a safety threshold (risk tolerance).

    Rationale: Uses 'sessions_since_touched' as proxy for time elapsed since last reinforcement,
    and uses difficulty/estimated_retention to estimate decay risk.
    """
    topic_ids = list(topics)
    # Sort topic IDs by importance or order in the provided dict (deterministic tie-breaking)
    ordered_tids = sorted(topic_ids, key=lambda tid: topics[tid].get("difficulty", 0), reverse=True)

    # Constants for policy tuning
    RETENTION_THRESHOLD = 0.75  # Target retention level to maintain (tau in the prompt)
    RISK_TOLERANCE_COEFF = 1.2   # Sensitivity of review trigger; higher means more aggressive reviews
    DECAY_SENSITIVITY = 0.8     # Multiplier for difficulty's impact on decay

    def get_risk(tid):
        """Calculates the risk that retention has fallen below threshold."""
        info = state[tid]
        if not info["introduced"]:
            return -1.0 # Not a candidate for review yet
        
        # sessions_since_touched is our proxy for elapsed time since last reinforcement (t)
        dt = float(info["sessions_since_touched"] or 0)
        difficulty = topics[tid].get("difficulty", 1.0)
        current_retention = info["estimated_retention"]

        # We model decay as R(t) = initial * exp(-lambda * t). 
        # Here we use a heuristic: risk increases with dt and difficulty, scaled by current estimate uncertainty.
        # If retention is already low or time passed is high relative to the 'stability' of the topic.
        decay_rate = (difficulty ** DECAY_SENSITIVITY) * 0.15
        predicted_retention = current_retention * math.exp(-decay_rate * dt)

        # Risk calculation: how far below threshold are we? Or more simply, a probability-like metric.
        if predicted_retention < RETENTION_THRESHOLD:
            return 2.0 - (predicted_retention / RETENTION_THRESHOLD) # Higher risk if lower retention
        else:
            # Risk is low but non-zero due to uncertainty/difficulty
            return max(0, RISK_TOLERANCE_COEFF * (1 - predicted_retention/RETENTION_THRESHOLD))

    # 1. Priority Check: Can we introduce a new topic?
    # We look for the first unintroduced topic whose prerequisites are all met and introduced.
    for tid in ordered_tids:
        if not state[tid]["introduced"]:
            prereqs = topics[tid].get("prerequisites", [])
            all_met = True
            for p in prereqs:
                if not state[p]["introduced"]:
                    all_met = False
                    break
            # If prerequisites are met, we introduce it to progress the curriculum. 
            # We only 'introduce' if there isn't a critical review needed first (handled below).
            pass # Continue searching for introduction candidates

    for tid in ordered_tids:
        if not state[tid]["introduced"]:
            prereqs = topics[tid].get("prerequisites", [])
            all_met = True
            for p in prereqs:
                if not state[p]["introduced"]:
                    all_met = False
                    break
            # If we can introduce, check if any existing topic is "crashing" (risk too high)
            critical_review_needed = False
            for other_tid in topic_ids:
                if state[other_tid]["introduced"] and get_risk(other_tid) > 1.0:
                    critical_review_needed = True
                    break

            # If no critical review is needed, introduce the next available topic per prerequisites
            if not critical_review_needed:
                return {"action": "introduce", "topic_id": tid}

    # 2. Review Check: Find highest risk introduced topic to prevent decay below threshold
    introduced_topics = [tid for tid in topic_ids if state[tid]["introduced"]]
    if not introduced_topics:
        # Fallback (should only happen at session 0)
        return {"action": "introduce", "topic_id": ordered_tids[0]}

    best_review_target = None
    max_risk = -1.0

    for tid in introduced_topics:
        r = get_risk(tid)
        # Tie-break using the order of topic IDs to ensure determinism and stable behavior
        if r > max_risk:
            max_risk = r
            best_review_target = tid
        elif abs(r - max_risk) < 1e-9: # Floating point tie break
             # Use index in ordered list as secondary key (smaller index is higher priority/tie breaker)
             if topic_ids.index(tid) < topic_ids.index(best_review_target):
                 best_review_target = tid

    # If the highest risk found doesn't even exceed a threshold, we still review 
    # if no new topics can be introduced (logic flows here).
    return {"action": "review", "topic_id": best_review_target}
