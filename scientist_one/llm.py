import json
import re
import time
from pathlib import Path
from typing import Callable

import httpx
from pydantic import BaseModel, ValidationError

from .config import Config, LLMConfig

Backend = Callable[[str, str, str, dict | None], str]


class LLMError(Exception):
    pass


class FakeBackend:
    """Scripted backend for tests: returns canned responses in order."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)

    def __call__(self, model, system, user, format):
        return self.responses.pop(0)


def _ollama_backend(host: str, llm_config: LLMConfig) -> Backend:
    import ollama

    # timeout bounds a hung/slow request (network stall, dead server); it is
    # independent of max_output_tokens, which bounds a request that IS
    # responding but has fallen into degenerate repetition and would
    # otherwise run until the server's own internal token ceiling. The
    # penalty/temperature options push against that same repetition failure
    # mode directly, rather than just capping how long it's allowed to run.
    client = ollama.Client(host=host, timeout=llm_config.timeout_s)

    def call(model: str, system: str, user: str, format: dict | None) -> str:
        stream = client.chat(
            model=model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            format=format,
            think=llm_config.think,
            stream=True,
            options={
                "num_predict": llm_config.max_output_tokens,
                "temperature": llm_config.temperature,
                "repeat_penalty": llm_config.repeat_penalty,
                "frequency_penalty": llm_config.frequency_penalty,
                "presence_penalty": llm_config.presence_penalty,
            },
        )
        content_parts: list[str] = []
        printed_header = False
        printed_thinking_label = False
        printed_answer_label = False
        for chunk in stream:
            msg = chunk["message"]
            if not printed_header:
                print(f"\n=== {model} ===", flush=True)
                printed_header = True
            thinking = msg.get("thinking")
            if thinking:
                if not printed_thinking_label:
                    print("[thinking] ", end="", flush=True)
                    printed_thinking_label = True
                print(thinking, end="", flush=True)
            content = msg.get("content")
            if content:
                if not printed_answer_label:
                    print("\n[answer] ", end="", flush=True)
                    printed_answer_label = True
                print(content, end="", flush=True)
                content_parts.append(content)
        if printed_header:
            print(flush=True)
        return "".join(content_parts)

    return call


class LLMClient:
    def __init__(self, config: Config, log_dir: Path, backend: Backend | None = None):
        self.config = config
        self.log_path = Path(log_dir) / "llm_calls.jsonl"
        self.backend = backend or _ollama_backend(config.ollama_host, config.llm)

    def _model(self, role: str) -> str:
        return {"reasoning": self.config.models.reasoning,
                "judging": self.config.models.judging}[role]

    def _call(self, role: str, system: str, user: str, format: dict | None) -> str:
        model = self._model(role)
        start = time.monotonic()
        error = None
        try:
            response = self.backend(model, system, user, format)
        except (httpx.HTTPError, TimeoutError) as exc:
            # A hung/timed-out/unreachable backend degrades like any other
            # malformed or empty LLM response: chat() callers already treat
            # "" as falsy and fall back, chat_json()'s retry loop already
            # treats unparseable output as a failed attempt.
            response = ""
            error = str(exc)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        entry = {"model": model, "system": system, "user": user,
                 "response": response, "duration_s": round(time.monotonic() - start, 2)}
        if error is not None:
            entry["error"] = error
        with self.log_path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
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
