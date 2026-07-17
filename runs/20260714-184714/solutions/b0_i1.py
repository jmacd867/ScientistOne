def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using an Adaptive Spaced Repetition policy.

    The strategy uses a risk-based approach inspired by the Ebbinghaus forgetting curve. 
    We estimate the 'stability' of each topic based on its estimated retention and time since last touch.
    An action is triggered (review) when the predicted decay suggests that any introduced 
    topic might fall below an acceptable threshold, or if a new introduction would be blocked by prerequisites.

    Rationale: We prioritize introducing topics to build breadth but switch to review mode 
    when 'retention risk' accumulates across the population of learned items. This balances 
    the acquisition-decay trade-off mentioned in cognitive psychology research.
    """
    topic_ids = list(topics)
    introduced_ids = [tid for tid in topic_ids if state[tid]["introduced"]]
    not_introduced_ids = [tid for tid in topic_ids if not state[tid]["introduced"]]

    # 1. Check availability of new topics (Prerequisite check)
    available_new_topics = []
    for tid in not_introduced_ids:
        prereqs = topics[tid].get("prerequisites", [])
        if all(state[p]["introduced"] for p in prereqs):
            available_new_topics.append(tid)

    # 2. Evaluate "Risk" of existing knowledge decay
    # We define risk as the probability that retention will drop below a threshold (e.g., 0.7).
    # Since we don't have direct access to beta, we use 'sessions_since_touched' and estimated_retention.
    risk_threshold = 1.5  # Sensitivity parameter: how many sessions of neglect are allowed?
    needs_review = False

    for tid in introduced_ids:
        sdt = state[tid]["sessions_onced_touched"] if "sessions_onced_touched" in state[tid] else 0 # Fallback safety
        # Using the provided key 'sessions_since_touched' from template
        intervals = state[tid].get("sessions_since_touched", 0) or 0
        retention = state[tid]["estimated_retention"]

        # Heuristic: If retention is already low OR if time passed makes it likely to drop below a critical point.
        # We use the decay factor as an implicit function of intervals and current estimation.
        if retention < 0.75 or (intervals > risk_threshold * topics[tid]["difficulty"]):
            needs_review = True
            break

    # 3. Decision Logic: Introduction vs Review
    
    # If we have new reachable topics, try to introduce them unless a review is urgent.
    if available_new_topics and not needs_review:
        # Pick the easiest/most accessible topic first to build momentum (or follow order)
        target = available_new_topics[0]
        return {"action": "introduce", "topic_id": target}

    # If no new topics are reachable, or we must review...
    if needs_review and introduced_ids:
        # Select the topic with highest 'decay risk' (lowest retention * time factor)
        target = max(
            introduced_ids,
            key=lambda tid: (-state[tid]["estimated_retention"], 
                             -(state[tid].get("sessions_since_touched") or 0),
                             -topic_ids.index(tid))
        )
        return {"action": "review", "topic_id": target}

    # Fallback: If we are stuck (no new topics reachable and no urgent review needed by heuristic),
    # pick the next available topic or a random introduced one to prevent stalling.
    if not introduced_ids and not available_new_topics:
        return {"action": "introduce", "topic_id": topic_ids[0]} # Should theoretically be impossible if logic holds

    if available_new_topics:
         # If we aren't 'urgent' but no reviews are flagged, keep introducing.
         target = available_new_topics[0]
         return {"action": "introduce", "topic_id": target}
    else:
        # All topics introduced or blocked; review the most neglected topic in existing set.
        if not introduced_ids: # Should only happen if first session and no intro possible (impossible by design)
             target = topic_ids[0] 
             return {"action": "introduce", "topic_id": target}

        # Find max neglect among all introduced topics as a fallback review strategy.
        target = max(
            introduced_ids,
            key=lambda tid: (state[tid].get("sessions_since_touched") or 0, -topic_ids.index(tid))
        )
        return {"action": "review", "topic_id": target}

