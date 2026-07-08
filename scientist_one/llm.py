import json
import re
import time
from pathlib import Path
from typing import Callable

from pydantic import BaseModel, ValidationError

from .config import Config

Backend = Callable[[str, str, str, dict | None], str]


class LLMError(Exception):
    pass


class FakeBackend:
    """Scripted backend for tests: returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, model, system, user, format):
        return self.responses.pop(0)


def _ollama_backend(host: str) -> Backend:
    import ollama

    client = ollama.Client(host=host)

    def call(model: str, system: str, user: str, format: dict | None) -> str:
        resp = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=format,
        )
        return resp["message"]["content"]

    return call


class LLMClient:
    def __init__(self, config: Config, log_dir: Path, backend: Backend | None = None):
        self.config = config
        self.log_path = Path(log_dir) / "llm_calls.jsonl"
        self.backend = backend or _ollama_backend(config.ollama_host)

    def _model(self, role: str) -> str:
        return {"reasoning": self.config.models.reasoning,
                "judging": self.config.models.judging}[role]

    def _call(self, role: str, system: str, user: str, format: dict | None) -> str:
        model = self._model(role)
        start = time.monotonic()
        response = self.backend(model, system, user, format)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a") as f:
            f.write(json.dumps({
                "model": model, "system": system, "user": user,
                "response": response, "duration_s": round(time.monotonic() - start, 2),
            }) + "\n")
        return response

    def chat(self, role: str, system: str, user: str) -> str:
        return self._call(role, system, user, None)

    def chat_json(self, role: str, system: str, user: str,
                  schema: type[BaseModel]) -> BaseModel | None:
        fmt = schema.model_json_schema()
        for _ in range(3):
            raw = self._call(role, system, user, fmt)
            try:
                return schema.model_validate_json(_extract_json(raw))
            except (ValidationError, ValueError):
                continue
        return None


def _extract_json(text: str) -> str:
    """Pull the first JSON object out of possibly-chatty model output."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object found")
    return match.group(0)
