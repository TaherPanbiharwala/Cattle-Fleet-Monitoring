from __future__ import annotations

import asyncio

import pytest

from stage2_rag_assistant.eval.eval_config import JudgeEmbeddingConfig
from stage2_rag_assistant.eval.ragas_embeddings import FakeRagasEmbedding, build_ragas_embedding


def test_same_text_embeds_identically(fake_ragas_embedding):
    emb = fake_ragas_embedding()
    assert emb.embed_text("hello") == emb.embed_text("hello")


def test_different_text_embeds_differently(fake_ragas_embedding):
    emb = fake_ragas_embedding()
    assert emb.embed_text("hello") != emb.embed_text("world")


def test_sync_and_async_agree(fake_ragas_embedding):
    emb = fake_ragas_embedding()
    assert asyncio.run(emb.aembed_text("hello")) == emb.embed_text("hello")


def test_factory_returns_fake_for_fake_provider():
    emb = build_ragas_embedding(JudgeEmbeddingConfig(provider="fake", model="x"))
    assert isinstance(emb, FakeRagasEmbedding)


def test_factory_raises_for_unimplemented_provider():
    with pytest.raises(NotImplementedError):
        build_ragas_embedding(JudgeEmbeddingConfig(provider="openai", model="x"))
