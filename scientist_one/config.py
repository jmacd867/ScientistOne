from pathlib import Path

import yaml
from pydantic import BaseModel


class ModelConfig(BaseModel):
    reasoning: str = "gemma4:26b"
    judging: str = "gemma4:12b"


class DiscoveryConfig(BaseModel):
    branches: int = 3
    iterations: int = 4
    survivors: int = 2


class InvestigatorConfig(BaseModel):
    max_papers: int = 15


class WriterConfig(BaseModel):
    max_rounds: int = 3


class VerifierConfig(BaseModel):
    numeric_tolerance: float = 0.01


class SolverConfig(BaseModel):
    timeout_s: int = 60


class LLMConfig(BaseModel):
    timeout_s: int = 300
    max_output_tokens: int = 4096


class Config(BaseModel):
    models: ModelConfig = ModelConfig()
    ollama_host: str = "http://localhost:11434"
    discovery: DiscoveryConfig = DiscoveryConfig()
    investigator: InvestigatorConfig = InvestigatorConfig()
    writer: WriterConfig = WriterConfig()
    verifier: VerifierConfig = VerifierConfig()
    solver: SolverConfig = SolverConfig()
    llm: LLMConfig = LLMConfig()


def load_config(path: Path | None = None) -> Config:
    if path is None or not Path(path).exists():
        return Config()
    data = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(data)
