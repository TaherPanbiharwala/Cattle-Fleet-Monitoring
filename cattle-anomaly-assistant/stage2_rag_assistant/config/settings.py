"""Stage 2 config loader. Reuses the shared StrictModel idiom from
shared/schemas/ (extra="forbid") rather than porting the old
src/herd_simulator/config.py dataclass+validator-function machinery — a
config typo should fail exactly as loudly as a schema typo does.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from shared.schemas._base import StrictModel

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"

LLMProvider = Literal["anthropic", "openai", "google", "openrouter", "fake"]


class LLMConfig(StrictModel):
    provider: LLMProvider
    model: str


class RouterConfig(StrictModel):
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0)


class GenerationConfig(StrictModel):
    max_tokens: int = Field(gt=0)
    timeout_s: float = Field(gt=0)
    max_retries: int = Field(ge=0)


class FallbackConfig(StrictModel):
    tau: float = Field(ge=0, le=1)
    high_severity_margin: float = Field(ge=0, le=0.5)


class KBConfig(StrictModel):
    db_path: str


class Stage2Config(StrictModel):
    llm: LLMConfig
    router: RouterConfig
    generation: GenerationConfig
    fallback: FallbackConfig
    kb: KBConfig


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Stage2Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Stage2Config.model_validate(raw)
