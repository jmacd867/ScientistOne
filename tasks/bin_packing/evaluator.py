import importlib.util
import json
import math
from collections import Counter
from pathlib import Path


def _load_solution(solution_path: str):
    spec = importlib.util.spec_from_file_location("solution", solution_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.pack


def evaluate(solution_path: str, workdir: str) -> dict:
    pack = _load_solution(solution_path)
    instances = json.loads(
        (Path(__file__).parent / "data" / "instances.json").read_text())
    ratios, lines = [], []
    for inst in instances:
        items, capacity = inst["items"], inst["capacity"]
        bins = pack(list(items), capacity)
        flat = [x for b in bins for x in b]
        if Counter(flat) != Counter(items):
            raise ValueError(f"{inst['name']}: packed items differ from input")
        for b in bins:
            if sum(b) > capacity + 1e-9:
                raise ValueError(f"{inst['name']}: bin overflow {sum(b)} > {capacity}")
        lower_bound = max(1, math.ceil(sum(items) / capacity))
        ratio = len(bins) / lower_bound
        ratios.append(ratio)
        lines.append(f"{inst['name']}: bins={len(bins)} lb={lower_bound} ratio={ratio:.4f}")
    score = sum(ratios) / len(ratios)
    lines.append(f"instances={len(instances)} mean_ratio={score:.4f}")
    return {"score": round(score, 4), "log": "\n".join(lines)}
