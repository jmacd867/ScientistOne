def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using a Threshold-Based Expansionist Control approach.

    The strategy aims to balance the introduction of new material with timely reviews 
    to prevent decay in retention levels across different learner types. It uses an 
    adaptive thresholding mechanism that prioritizes introducing topics until they are all 
    unlocked, then shifts focus towards a prioritized review queue based on 'risk' (decay).

    Rationale: To minimize the time to mastery for the whole population, we must prevent 
    topics from falling below critical thresholds. We prioritize introductions only when 
    prerequisites allow and topics are not yet introduced; otherwise, we perform reviews 
    on a topic that has high 'risk' (high sessions_since_touched) or low estimated retention.

    Args:
        state: {topic_id: {"introduced": bool, "estimated_retention": float,
                        "sessions_since_touched": int | None}}
        topics: {topic_id: {"name": str, "prerequisites": list[str], "difficulty": float}}
        session: current session number (0-indexed)

    Returns: {"action": "introduce" | "review", "topic_id": str}
    """
    # 1. Identify all topic IDs and their order for deterministic tie-breaking
    all_ids = list(topics.keys())
    introduced_count = sum(1 for tid in all_ids if state[tid]["introduced"])

    # 2. Check for introduction candidates (Prerequisites must be met)
    introduction_candidates = []
    for tid in all_ids:
        if not state[tid]["introduced"]:
            prereqs_met = True
            for prereq in topics[tid].get("prerequisites", []):
                # A topic is ready if it has been introduced previously 1 or more sessions ago.
                # However, since we only see the current 'state', checking state['introduced'] Is sufficient.
                if not state[prereq]["introduced"]:
                    prereqs_met = False
                    break
            if prereqs_met:
                introduction_candidates.append(tid)

    # 3. Strategy Phase Logic: Expansion vs Maintenance/Review phase transition logic based on progress.
    # If there are unintroduced topics that CAN be introduced, we attempt to introduce them first.
    # This follows the "Expansionist" part of our strategy.
    if introduction_candidates:
        # Prioritize candidates with lower difficulty or those appearing earlier in list for stability
        target = min(introduction_candidates, key=lambda tid: (topics[tid]["difficulty"], all_ids.index(tid)))
        return {"action": "introduce", "topic_id": target}

    # 4. Review Phase Logic: If no introductions are possible or needed, focus on maintenance.
    # We calculate a 'risk score' for each introduced topic based on decay risk and estimated retention.
    review_candidates = [tid for tid in all_ids if state[tid]["introduced"]]

    if not review_candidates:
        # Fallback (should theoretically be impossible given problem constraints) 
        return {"action": "introduce", "topic_id": all_ids[0]}

    def calculate_risk(tid):
        """Calculate a risk score. Higher means more urgent to review."""
        retention = state[tid]["estimated_retention"]
        # sessions_since_touched is None if just introduced or never touched, treat as 0 for safety.
        last_touch = state[tid].get("sessions_since_touched")
        if last_touch is None:
            last_touch = 0
            
        # Risk increases with decay (time since touch) and decreases with high retention.
        # We use a weighted sum of time-decay risk and low-retention risk.
        return -retention + (float(last_touch) * 2.5)

    target = max(review_candidates, key=lambda tid: (calculate_risk(tid), -all_ids.index(tid)))
    
    # Check if we should try to introduce a topic even though it's not "ready" in our local logic?
    # Actually, the prompt implies 'introduce' is for new topics and 'review' for existing ones. 
    # If all introduced are mastered (not visible here but implied), review continues on most stale.

    return {"action": "review", "topic_id": target}

