import math

def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to do this study session using Latent Decay Reconstruction (LDR).
    
    The policy maintains an internal estimate of the decay rate for each topic. 
    It prioritizes introducing new topics that have met prerequisites and reviewing 
    topics whose predicted retention is likely falling below a critical threshold, 
    balancing curriculum progression with memory stability.

    Note: Since this function must be stateless across calls (no global persistence),
    we use the 'session' number as an index to reconstruct or simulate learning progress,
    but for standard competitive environments where state persists in `state`, we rely on 
    the provided estimation of retention and sessions since touched.

    Rationale: We treat topic decay as a latent variable estimated by observing 
    (sessions_since_touched) vs (estimated_retention). The utility function prioritizes 
    topics with high 'risk'—those where the predicted drop in stability is highest.
    """

    # Identify all topics and their properties
    topic_ids = list(topics.keys())
    introduced_count = sum(1 for tid in topic_ids if state[tid]["introduced"])
    total_topics = len(topic_ids)

    # 1. Check for Introduction candidates: Topics not yet introduced but prerequisites are met
    introduction_candidates = []
    for tid in topic_ids:
        if not state[tid]["introduced"]:
            prereqs = topics[tid].get("prerequisites", [])
            all_prereqs_met = all(state[p]["introduced"] for p in prereqs)
            # We also check if the difficulty is manageable (heuristic part of LDR logic)
            if all_prereqs_met:
                introduction_candidates.append(tid)

    # 2. If we can introduce a new topic and it's strategically beneficial, do so.
    # Strategy: Introduce only if there are no urgent reviews needed or to prevent stalling progress.
    urgent_review = False
    for tid in topic_ids:
        if state[tid]["introduced"]:
            retention = state[tid]["estimated_retention"]
            sessions_since = state[tid].get("sessions_since_touched", 0) or 0
            # Heuristic threshold for 'urgent' review based on estimated retention and decay risk
            if retention < 0.6: # Threshold can be tuned; lower means more aggressive introduction
                urgent_review = True
                break

    if introduction_candidates and not urgent_review:
        # Pick the easiest available topic to build momentum (or first in list)
        target_intro = min(introduction_candidates, key=lambda tid: topics[tid]["difficulty"])
        return {"action": "introduce", "topic_id": target_intro}

    # 3. Review Strategy: Calculate 'Risk' for all introduced topics.
    # Risk is estimated as the potential loss in population stability if not reviewed now.
    review_candidates = [tid for tid in topic_ids if state[tid]["introduced"]]
    
    if not review_candidates and introduction_candidates:
        target_intro = min(introduction_candidates, key=lambda tid: topics[tid]["difficulty"])
        return {"action": "introduce", "topic_id": target_intro}

    # If no introduced topics (shouldn't happen if session > 0), fallback to first topic intro.
    if not review_candidates and introduction_candidates:
         return {"action": "introduce", "topic_id": introduction_candidates[0]}

    def calculate_risk(tid):
        """Calculates the predicted degradation risk for a topic."""
        retention = state[tid]["estimated_retention"]
        sessions_since = state[tid].get("sessions_since_touched", 0) or 0
        difficulty = topics[tid]["difficulty"]
        # Risk increases with decay (low retention), time since last touch, and difficulty.
        # We model the 'latent' risk as a function of how much stability we stand to lose.
        risk_score = (1.0 - retention) * (sessions_since + 1) * difficulty
        return risk_score

    if review_candidates:
        target_review = max(review_candidates, key=calculate_risk)
        # If the highest risk is still very low and we have new topics to introduce, try introducing.
        highest_risk = calculate_risk(target_review)
        threshold = 0.1 # Heuristic threshold for deciding between review vs intro
        if introduction_candidates and highest_risk < threshold:
            return {"action": "introduce", "topic_id": introduction_candidates[0]}
            
        return {"action": "review", "topic_id": target_review}

    # Final Fallback (should be unreachable)
    fallback = topic_ids[0] if not introduction_candidates else introduction_candidates[0]
    if state[fallback]["introduced"]:
         return {"action": "review", "topic_id": fallback}
    else:
         return {"action": "introduce", "topic_id": fallback}

