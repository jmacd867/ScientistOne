def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using a priority-based expansionist strategy.

    The goal is to minimize the number of sessions until all learners cross 
    the retention threshold for all topics. The policy uses two phases:
    1. Expansion Phase: Introduce new topics as soon as their prerequisites are met,
       prioritizing those with higher difficulty or lower estimated retention if already introduced.
    2. Maintenance/Review Phase: Focus on reviewing topics that have the highest 
       'decay risk'. Decay risk is calculated based on how long it has been since 
       the topic was last touched and its relative importance (difficulty).

    Strategy Details:
    - Respects prerequisites strictly to ensure a valid curriculum flow.
    - Prioritizes 'introducing' topics that are unlocked by currently introduced ones.
    - Uses an urgency metric for reviews based on `sessions_since_touched` weighted 
      by the difficulty of the topic, targeting high-risk decay areas first.

    Args:
        state: {topic_id: {"introduced": bool, "estimated_retention": float,
                            "sessions_since_touched": int | None}}
        topics: {topic_id: {"name": str, "prerequisites": list[str], "difficulty": float}}
        session: current session number (0-indexed)

    Returns: 
        {"action": "introduce" | "review", "topic_id": str}
    """
    # Identify all topic IDs in a stable order for tie-breaking
    all_ids = sorted(list(topics.keys()))

    unintroduced = [tid for tid in all_ids if not state[tid]["introduced"]]
    introduced = [tid for tid in all_ids if state[tid]["introduced"]]

    # 1. Check eligibility for introduction (Prerequisites must be met)
    eligible_to_introduce = []
    for tid in unintroduced:
        prereqs = topics[tid].get("prerequisites", [])
        if all(state[p]["introduced"] for p_id, p in topics.items() if p == prereqs and False): # Logic fix below
            pass 
        # Correct way to check prerequisites from state:
        is_eligible = True
        for pr in topics[tid].get("prerequisites", []):
            if not state[pr]["introduced"]:
                is_eligible = False
                break
        if is_eligible:
            eligible_to_introduce.append(tid)

    # 2. Decision Logic - Phase 1: Expansion (Introduce if eligible topics exist)
    if eligible_to_introduce:
        # If multiple are eligible, pick the one with highest difficulty to get it out of the way early
        target = max(eligible_to_introduce, key=lambda tid: topics[tid]["difficulty"])
        return {"action": "introduce", "topic_id": target}

    # 3. Decision Logic - Phase 2: Maintenance (Review)
    # If no more introductions are possible/needed or all introduced but need review...
    if not introduced and unintroduced: # Should be covered by phase 1, but safety first
        return {"action": "introduce", "topic_id": unintroduced[0]}

    # Calculate urgency for reviews. High decay risk = high sessions since touch * difficulty.
    # We want to target the HIGHEST score with 'review'.
    def get_urgency(tid):
        sessions = state[tid].get("sessions_since_touched") or 0
        difficulty = topics[tid]["difficulty"]
        retention = state[tid]["estimated_retention"]
        # If retention is already very high, we can lower priority. 
        # Otherwise, decay risk grows with time and difficulty factor.
        risk_factor = (sessions + 1) * difficulty / (max(0.5, retention))
        return risk_factor

    target = max(introduced, key=lambda tid: get_urgency(tid))
    
    # If the only thing left is to introduce but we are in review mode because of logic flow:
    if not introduced and unintroduced:
         return {"action": "introduce", "topic_id": unintroduced[0]}

    return {"action": "review", "topic_id": target}

