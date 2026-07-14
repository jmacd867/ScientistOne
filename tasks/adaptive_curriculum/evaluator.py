import importlib.util
import json
import statistics
from pathlib import Path


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.choose_action


def _load_simulator():
    spec = importlib.util.spec_from_file_location(
        "adaptive_curriculum_simulator", Path(__file__).parent / "simulator.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_data():
    data_dir = Path(__file__).parent / "data"
    topics = json.loads((data_dir / "topics.json").read_text())
    learners = json.loads((data_dir / "learners.json").read_text())
    return topics, learners


def _textbook_baseline(state: dict, topics: dict, session: int) -> dict:
    topic_ids = list(topics)
    for tid in topic_ids:
        if state[tid]["introduced"]:
            continue
        prereqs = topics[tid]["prerequisites"]
        if all(state[p]["estimated_retention"] >= 0.8 for p in prereqs):
            return {"action": "introduce", "topic_id": tid}

    introduced = [tid for tid in topic_ids if state[tid]["introduced"]]
    target = min(
        introduced,
        key=lambda tid: (state[tid]["estimated_retention"], topic_ids.index(tid)),
    )
    return {"action": "review", "topic_id": target}


def evaluate(solution_path: str, workdir: str) -> dict:
    policy = _load_solution(solution_path)
    sim = _load_simulator()
    topics, learners = _load_data()

    lines = []
    discovered_scores = []
    baseline_scores = []
    for learner in learners:
        d_score = sim.run_episode(policy, topics, learner)
        b_score = sim.run_episode(_textbook_baseline, topics, learner)
        discovered_scores.append(d_score)
        baseline_scores.append(b_score)
        lines.append(f"{learner['name']}: discovered={d_score:.1f} baseline={b_score:.1f}")

    discovered_score = statistics.mean(discovered_scores)
    baseline_score = statistics.mean(baseline_scores)
    lines.append(f"discovered_score={discovered_score:.4f}")
    lines.append(f"baseline_score={baseline_score:.4f}")
    return {"score": round(discovered_score, 4), "log": "\n".join(lines)}
