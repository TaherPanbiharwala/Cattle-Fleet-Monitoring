"""Deterministic, network-free stand-in for a real embedding model. Only
AnswerRelevancy needs this. The actual vector content is irrelevant for
plumbing tests — it just needs to conform to BaseRagasEmbedding's shape.
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from stage2_rag_assistant.eval import _ragas_compat  # noqa: F401  (import for its side effect, before ragas)
from ragas.embeddings.base import BaseRagasEmbedding

if TYPE_CHECKING:
    from stage2_rag_assistant.eval.eval_config import JudgeEmbeddingConfig

_VECTOR_DIM = 16


class FakeRagasEmbedding(BaseRagasEmbedding):
    """A deterministic hash-derived vector per input text — same text always
    embeds to the same vector (so a metric comparing two embeddings of
    identical text gets a sane result), different text embeds differently,
    with no real semantic meaning and no network call."""

    def __init__(self) -> None:
        super().__init__(cache=None)
        self.call_log: list[str] = []

    def _vector_for(self, text: str) -> list[float]:
        self.call_log.append(text)
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [b / 255.0 for b in digest[:_VECTOR_DIM]]

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return self._vector_for(text)

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self._vector_for(text)


def build_ragas_embedding(config: "JudgeEmbeddingConfig") -> BaseRagasEmbedding:
    """Mirrors build_ragas_llm()'s factory shape exactly."""
    if config.provider == "fake":
        return FakeRagasEmbedding()
    raise NotImplementedError(
        f"Judge embedding provider {config.provider!r} is not implemented yet — see LLM_ASSISTANT_STATUS.md "
        "Open Question 2. Set judge_embedding.provider: fake in eval_config.yaml until a real provider is "
        "chosen and built."
    )
