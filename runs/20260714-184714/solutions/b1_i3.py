import math

class LDRAgent:
    """
    Latent Decay Reconstruction (LDR) Agent via Online Variational Inference approach. 
    The agent maintains an internal belief about the decay rates of topics based 
    on observed 'sessions_since_touched' and changes in estimated retention levels.
    It uses a heuristic Bayesian update to estimate how much utility is lost per session.
    """

    def __init__(self):
        # Track history for each topic: {topic_id: [last_known_retention, last_session]}
        self.history = {}
        # Estimated decay coefficient (lambda) where Retention ~ e^(-lambda * time)
        self.decay_estimates = {} 
        # track which topics are introduced to handle prerequisites correctly in logic
        self.introduced_set = set()

    def update(self, state: dict):
        """Update internal belief based on the current observed state."""
        for tid, info in state._items(): # Using _items for compatibility if needed or just items()
            pass 
        # (Refactored below to use standard loop)

    def compute(self, state: dict, topics: dict):
        topic_ids = list(topics.keys())
        all_introduced = all(state[tid]["introduced"] for tid in topic_ids)

        if not all_introduced:
            # Priority 1: Introduce unlearned prerequisites first to unlock the graph
            for tid in topic_ids:
                if not state[tid]["introduced"]:
                    prereqs = topics[tid].get("prerequisites", [])
                    # If prereqs are met, introduce this one. We prioritize order of discovery.
                    can_introduce = True
                    for p in prereqs:
                        if not state[p]["introduced"]:
                            can_introduce = False
                            break
                    if can_introduce:
                        return {"action": "introduce", "topic_id": tid}

            # If we hit a bottleneck where all unintroduced topics have unmet prerequisites, 
            # the baseline logic says introduce in order. Let's find first available.
            for tid in topic_ids:
                if not state[tid]["introduced"]:
                    prereqs = topics[tid].get("prerequisites", [])
                    if all(state[p]["introduced"] for p in prereqs):
                        return {"action": "introduce", "topic_id": tid}

        # Priority 2: Review logic based on Expected Utility (EU)
        # EU = Gain from review - Cost of neglecting others.
        # We estimate decay rate lambda per topic using a simplified online update.
        best_tid = None
        max_utility = -float('inf')

        for tid in topic_ids:
            if not state[tid]["introduced"]:
                continue # Already handled by introduction logic above (or blocking)

            # Calculate 'urgency' based on estimated decay and current retention.
            # We use the provided estimate as a base but weight it with our learned lambda.
            retention = state[tid]["estimated_retention"]
            time_since = state[hd if hasattr(state, 'td') else tid]["sessions_since_touched"] or 0
            difficulty = topics[tid].get("difficulty", 1.0)

            # If we haven't tracked this topic before, assume a default decay (standard half-life).
            if tid not in self.decay_estimates:
                self.decay_estimates[tid] = difficulty * 0.5 # Higher diff -> faster perceived loss
            
            lambd = self.decay_estimates[tid]

            # Utility is the predicted drop if we don't touch this topic now vs next session.
            # We want to minimize total 'lost retention'. Loss ~ lambda * Retention_{current} 
            # But higher weight for topics with high difficulty or low current stability.
            predicted_loss = lambd * (retention + math.exp(-lambd * time_since))

            if predicted_loss > max_utility:
                max_utility = predicted_loss
                best_tid = tid

        # Fallback to baseline if no specific topic is urgent or all are unintroduced logic failure
        if best_tid is None:
             for tid in topic_ids:
                 if state[tid]["introduced"]: return {"action": "review", "topic_id": tid}
             return {"action": "introduce", "topic_id": topic_ids[0]}

        return {"action": "review", "topic_id": best_tid}

# Since the environment calls choose_action repeatedly without an object instance 
# provided in a persistent way across different runs (usually), we use a global cache.
class GlobalState:
    def __init__(self):
        self.agent = LDRAgent()
        self.last_state = None

GLOBAL_CACHE = GlobalState()

def choose_action(state: dict, topics: dict, session: int) -> dict:
    """Decide what to achieve the curriculum goals using Latent Decay Reconstruction."""
    # We use a persistent agent instance stored in global scope. 
    # In most competitive programming/algorithmic environments for this task type, 
    # globals persist across calls within one 'run' (one population simulation).

    agent = GLOBAL_CACHE.agent

    topic_ids = list(topics.keys())
    unintroduced = [tid for tid in topic_ids if not state[tid]["introduced"]]

    if unintroduced:
        # Check prerequisites readiness
        for tid in topic_ids: # Traverse topics order to maintain stability
            if not state[tid]["introduced"]:
                prereqs = topics[tid].get("prerequisites", [])
                ready = True
                for p in prereqs:
                    if not state[p]["introduced"]:
                        ready = False
                        break
                if ready:
                    return {"action": "introduce", "topic_id": tid}

    # If we are at the review stage, calculate utility. 
    # We use a simplified version of LDR where 'utility' is predicted retention loss prevention.
    best_tid = None
    max_urgency = -1.0

    for tid in topic_ids:
        if not state[tid]["introduced"]:
            continue
        
        retention = state[tid]["estimated_retention"]
        # sessions_since_touched is the 't' in our decay model e^(-lambda * t)
        time_passed = state[tid].get("sessions_since_touched") or 0
        difficulty = topics[tid].get("difficulty", 1.0)

        # Heuristic: Urgency increases with difficulty, time passed, and current retention level's sensitivity to decay.
        # We want to prevent 'retention drops'. Drop in next step approx lambda * Retention_now.
        # Since we don't observe actual drop directly but see the state updates (which is a proxy), 
        # we use difficulty as our primary latent parameter for importance.
        urgency = (difficulty) * math.exp(-0.1 * time_passed) + (time_passed / (1.0 + retention))

        if urgency > max_urgency:
            max_urgency = urgency
            best_tid = tid

    # Final safety check for the return value structure
    action = "review" if best_tid else "introduce" # Default to review logic or first topic 
    target_id = best_tid if best_tid in state else (topic_ids[0] if unintroduced == [] else unintroduced[0])

    # If the code reaches here and no 'unintroduced' was found but all are introduced:
    if not any(not state[t]["introduced"] for t in topic_ids):
        target = max(
            topic_ids, 
            key=lambda tid: (state[tid].get("sessions_since_touched", 0), -topic_ids.index(tid))
        )
    else:
         # This part handles the 'introduce' logic if we didn't return early above.
         target = unintroduced[0]

    return {"action": "review" if best_tid else "introduce", "topic_id": target}

