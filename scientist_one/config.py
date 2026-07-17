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
    # How many verify->refine cycles to run before giving up and leaving the
    # paper as a draft. A single small ID typo (e.g. a dropped leading zero)
    # often isn't fixed on the first refiner attempt; observed production
    # runs needed a second or third pass to actually converge.
    max_refine_rounds: int = 3


class SolverConfig(BaseModel):
    timeout_s: int = 60


class LLMConfig(BaseModel):
    timeout_s: int = 300
    max_output_tokens: int = 4096
    # Anti-repetition sampling. Ollama's own defaults (temperature=1.0,
    # repeat_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0) were
    # not enough to stop gemma4:26b from degenerating into loops like
    # "in in in in..." on long-context prompts; these values push harder
    # against exact and near-exact repetition without flattening output
    # into incoherence.
    temperature: float = 0.7
    repeat_penalty: float = 1.3
    frequency_penalty: float = 0.4
    presence_penalty: float = 0.4
    # gemma4 is a "thinking" model: it burns generated tokens on an internal
    # reasoning trace before the final answer. Since max_output_tokens counts
    # thinking tokens too, a complex prompt can exhaust the whole budget on
    # reasoning and leave zero tokens for the actual answer — content comes
    # back empty despite a normal-looking duration. None of this pipeline's
    # prompts expect or read a separate thinking trace, so it's off by
    # default; set true (or "low"/"medium"/"high") to re-enable per Ollama's
    # `think` chat parameter.
    think: bool = False


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
