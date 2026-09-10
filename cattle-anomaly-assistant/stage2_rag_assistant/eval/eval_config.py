"""Layer 2 (Ragas) judge config loader. See eval_config.yaml's own comment
for why this is a separate file from stage2_rag_assistant/config/settings.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml

from shared.schemas._base import StrictModel

DEFAULT_EVAL_CONFIG_PATH = Path(__file__).parent / "eval_config.yaml"

JudgeProvider = Literal["anthropic", "openai", "google", "fake"]


class JudgeLLMConfig(StrictModel):
    provider: JudgeProvider
    model: str


class JudgeEmbeddingConfig(StrictModel):
    provider: JudgeProvider
    model: str


class EvalConfig(StrictModel):
    judge_llm: JudgeLLMConfig
    judge_embedding: JudgeEmbeddingConfig


def load_eval_config(path: str | Path = DEFAULT_EVAL_CONFIG_PATH) -> EvalConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return EvalConfig.model_validate(raw)
