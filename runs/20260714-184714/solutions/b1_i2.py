import math

class LDRPolicy:
    """
    Latent Decay Reconstruction (LDR) via Online Variational Inference Approximation.
    This agent maintains an internal belief about topic difficulty and decay rates, 
    estimating the 'hidden' retention levels by observing transitions in state signals.
    It uses a heuristic utility function to balance introducing new topics vs reviewing old ones.
    """

    def __init__(self):
        # Internal model of each topic: {topic_id: {"decay": float, "difficulty": float}}
        self.model = {}
        # Track history for gradient-like updates (session -> action/outcome)
        self.history = [] 
        # Estimated 'true' retention level per topic based on observed state transitions
        self.estimated_retention = {}

    def _update_beliefs(self, topics: dict, state: dict):
        """Update the latent parameters (decay and difficulty estimation)."""
        for tid in topics:
            # Initialize if new
            if tid not in self.model:
                self._init_topic(tid, topics[tid])

            curr = state[tid]
            prev = None # In a real implementation, we would store the previous session's state 
                        # to calculate delta-retention (the 'reconstruction').
                        # Since choose_action is stateless per call in this interface, 
                        # we use an approximation based on sessions_since_touched.

            if curr["introduced"]:
                self.estimated_retention[tid] = max(0.1, self._infer_current_level(tid, state))
            else:
                self.estimated_retention[tid] = 0.0

    def _init_topic(self, tid, topic_meta):
        # Start with a conservative decay rate (low) and difficulty from metadata
        diff = topic_meta["difficulty"] if "difficulty" in topic_meta else 1.0
        self.model[tid] = {
            "decay": 0.05 + (diff * 0.2), # Base assumption: harder topics decay faster? Or just higher uncertainty
            "confidence": 0.1,           # Bayesian-like confidence in our estimate
            "difficulty_weight": diff    # Scaling factor for how much a review helps
        }

    def _infer_current_level(self, tid, state):
        """Approximates latent retention using the observable 'sessions_since_touched'."""
        if not state[tid]["introduced"]: return 0.0
        s = state[tid].get("sessions_since_touched", 0) or 0
        # Decay model: R(t) = R0 * exp(-lambda * t). We estimate lambda via our internal 'decay' param.
        return math.exp(-self.model[tid]["decay"] * s)

    def decide(self, state: dict, topics: dict, session: int):
        # 1. Update model with current observations (Latent Reconstruction step)
        for tid in topics:
            if tid not in self.model:
                self._init_topic(tid, topics[tid])
            self._update_beliefs({tid: topics[tid]}, state)

        # Check if there are any unintroduced topics that have prerequisites met
        available_to_introduce = []
        for tid, meta in topics.items():
            if not state[tid]["introduced"]:
                prereqs = meta.get("prerequisites", [])
                all_met = all(state[p]["introduced"] for ply in [prereqs] for p in ply) # simple check logic error fix: 1-level deep loop? No, just use list comp below
                # Corrected prereq check implementation:
                ready = True
                for pr in meta.get("prerequisites", []):
                    if not state[pr]["introduced"]:
                        ready = False
                        break
                if ready:
                    available_to_introduce.append(tid)

        # 2. Calculate Utility of "Introduce" vs "Review"
        # We use a Lookahead-style heuristic (Expected Gain in Population Stability).
        # Cost is the risk that reviewing something else lets another topic decay too much.

        if available_to_introduce:
            best_intro = None
            max_gain = -float('inf')
            for tid in available_to_introduce:
                # Utility of introduction ~ 1 / (difficulty * uncertainty)
                # We want to introduce topics that are 'easy' but unintroduced first, or high priority.
                gain = -(self.model[tid]["difficulty_weight"]) # Simple heuristic for intro utility
                if gain > max_gain:
                    max_gain = gain
                    best_intro = tid

            # Decision threshold logic (Exploration vs Exploitation) 
            # If the most critical topic is about to fail, we MUST review.
            critical_review = self._get_most_urgent_topic(state)
            if critical_review:
                return {"action": "review", "topic_id": critical_review}

            # To avoid over-introducing and causing a cascade of decay (the 'shaky' problem), 
            # we only introduce if the current stability is above a threshold.
            avg_retention = sum(self._infer_current_level(t, state) for t in topics if state[t]["introduced"]) / len(topics) \
                if any(state[t]["introduced"] for t in topics) else 1.0

            # If stability is high and we have new stuff to show, introduce it.
            if avg_retention > 0.4 or not critical_review: # Threshold heuristic
                 return {"action": "introduce", "topic_id": best_intro} if best_intro else self._fallback(state, topics)

        # Fallback/Default to review logic (The 'Stability' strategy)
        target = self._get_most_urgent_topic(state)
        if target:
            return {"action": "review", "topic_id": target}
        else: 
             # If nothing introduced yet, just pick the first available intro.
             for tid in topics:
                 if not state[tid]["introduced"]: return {"action": "introduce", "topic_id": tid}

        return self._fallback(state, topics)

    def _get_most_urgent_topic(self, state):
        """Finds the topic with lowest predicted retention."""
        candidates = []
        for tid in state:
            if state[tid]["introduced"]:
                # Utility is inversely proportional to estimated survival time. 
                # We prioritize topics where (retention * decay_rate) is highest/most dangerous? No, just low retention.
                level = self._infer_current_level(tid, state) # This uses the internal model's view of 'state' vs its own belief logic
                if level < 0.5:  # Threshold for urgency (below 50% is critical)
                    candidates.append((level, tid))

        if not candidates: return None
        return min(candidates)[1] # Return topic with lowest estimated retention

    def _fallback(self, state, topics):
        """Baseline fallback to ensure progress."""
        topic_ids = list(topics)
        for tid in topic_ids:
            if not state[tid]["introduced"]:
                return {"action": "introduce", "topic_id": tid}
        target = max(topic_ids, key=lambda tid: (state[tid].get("sessions_since_touched") or 0))
        return {"action": "review", "topic_id": target}

# Global instance to maintain state across calls in the same simulation run.
# The interface provided is stateless per-call, so we use a persistent object pattern via module level variable if needed,
# but since I must provide ONE function `choose_action`, and Python functions are objects: 
class AgentWrapper:
    def __init__(self):
        self.agent = LDRPolicy()

    def call(self, state, topics, session):
        return self.agent.decide(state, topics, session)

# Note: In a real environment-driven loop (like the one described), 
# we'd need to persist 'AgentWrapper'. Since I can only provide ONE function definition block, 
# and `choose_action` is called repeatedly with same context but new state/session.
# We use an attribute on the function itself as a static variable hack for persistence in Python within one script scope.

def choose_action(state: dict, topics: dict, session: int) -> dict:
    if not hasattr(choose_action, "_agent"):
        choose_action._agent = LDRPolicy()
    return choose_action._agent.decide(state, topics, session)

