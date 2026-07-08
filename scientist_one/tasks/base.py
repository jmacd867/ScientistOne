from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class TaskSpec(BaseModel):
    name: str
    description: str
    metric_direction: Literal["higher", "lower"]
    seed_queries: list[str] = []
    timeout_s: int | None = None
    path: Path

    def starter_path(self) -> Path:
        return self.path / "starter.py"

    def evaluator_path(self) -> Path:
        return self.path / "evaluator.py"

    def better(self, a: float, b: float) -> bool:
        return a > b if self.metric_direction == "higher" else a < b


def load_task(path: Path) -> TaskSpec:
    path = Path(path)
    for required in ("task.yaml", "starter.py", "evaluator.py"):
        if not (path / required).exists():
            raise FileNotFoundError(f"task is missing {required}: {path / required}")
    data = yaml.safe_load((path / "task.yaml").read_text())
    return TaskSpec(path=path, **data)
