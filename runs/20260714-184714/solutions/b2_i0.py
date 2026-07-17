def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using Threshold-Based Expansionist Control.

    The policy implements a two-phase controller for curriculum design:
    Phase 1 (Expansion): Prioritize introducing new topics in order of their availability
                         (prerequisites met), until all possible topics are introduced.
    Phase 2 (Retention/Maintenance): Once the expansion phase is complete, prioritize 
                                     reviewing topics that have been dormant longest to 
                         combat retention decay.

    Rationale: This approach balances exploring new knowledge with maintaining existing 
               knowledge by focusing on high-risk (longest untouched) nodes once a baseline 
               curriculum coverage is achieved.

    Args:
        state: {topic_id: {"introduced": bool, "estimated_retention": float,
                            "sessions_since_touched": int | None}}
        topics: {topic_id: {"name": str, "prerequisites": list[str], "difficulty": float}}
        session: current session number (0-indexed)

    Returns: 
        {"action": "introduce" | "review", "topic_id": str}
    """
    # Sort topic IDs by their appearance in the dict to ensure deterministic behavior for ties
    all_ids = list(topics.keys())
    
    unintroduced = [tid for tid in all_ids if not state[tid]["introduced"]]

    if unintroduced:
        # PHASE 1: Expansionist approach. Try to introduce topics whose prerequisites are met.
        # We iterate through the available pool and find the first one that is "ready".
        for tid in unintroduced:
            prereqs = topics[tid].get("prerequisites", [])
            if all(state[p]["introduced"] for p in prereqs):
                return {"action": "introduce", "topic_id": tid}

        # If no new topic can be introduced due to prerequisites, 
        # we must review an already introduced topic that is 'ready' or just pick the first unintroduced.
        # To avoid deadlocks in expansion phase if a prereq isn't met yet but nothing else is intro-able:
        for tid in all_ids:
            if not state[tid]["introduced"]:
                return {"action": "introduce", "topic_id": tid}

    # PHASE 2: Maintenance/Review. All topics introduced or no progress possible on new ones.
    # We prioritize the topic that has been dormant (sessions_since_touched) for the longest time.
    introduced_ids = [tid for tid in all_ids if state[tid]["introduced"]]

    if not introduced_ids:
        # Fallback safety case: If somehow no topics are marked as introduced, 
        # pick any topic to introduce (though Phase 1 should have caught this).
        return {"action": "introduce", "topic_id": all_ids[0]}

    # Find the target for review. We use a tuple key for deterministic tie-breaking:
    # Primary sort: sessions_since_touched descending (highest value = longest dormant)
    # Secondary sort: original index in list ascending to ensure stability/determinism.
    target = max(
        introduced_ids,
        key=lambda tid: (
            state[tid]["sessions_since_touched"] if state[tid]["sessions_since_touched"] is not None else -1,
            -all_ids.index(tid) # Negative index so smaller original indices win in max() tie-break
        ),
    )

    return {"action": "review", "topic_id": target}

