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
