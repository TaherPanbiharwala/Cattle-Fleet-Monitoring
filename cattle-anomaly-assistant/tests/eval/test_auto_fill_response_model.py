"""Regression tests for a real bug found comparing a live run's Layer-2
scores against ragas's own source: answer_relevancy was silently forced to
0.000 on every single case since M2b, in every eval run (fake generation
or real), because the generic auto-filler's blanket int->1 default
collided with AnswerRelevancy's inverted-semantics `noncommittal` field.
See ragas_llm.py's _auto_fill_value docstring for the full mechanism.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision

from stage2_rag_assistant.eval.ragas_llm import auto_fill_response_model


class _NoncommittalLike(BaseModel):
    question: str
    noncommittal: int


class _OrdinaryVerdict(BaseModel):
    verdict: int
    reason: str


def test_noncommittal_field_fills_to_zero_not_the_generic_default():
    filled = auto_fill_response_model(_NoncommittalLike)
    assert filled.noncommittal == 0


def test_other_int_fields_keep_the_generic_default_of_one():
    """The fix must not regress Faithfulness/ContextPrecision/ContextRecall,
    which all use a 0/1 verdict field with the opposite (1=positive)
    semantics and rely on the original default."""
    filled = auto_fill_response_model(_OrdinaryVerdict)
    assert filled.verdict == 1


async def test_answer_relevancy_is_not_trivially_zero_against_the_default_responder(
    fake_ragas_llm, fake_ragas_embedding
):
    llm = fake_ragas_llm()  # default auto-fill responder, exactly what run_eval.py uses un-scripted
    metric = AnswerRelevancy(llm=llm, embeddings=fake_ragas_embedding())
    score = await metric.ascore(user_input="Why was this cow flagged?", response="placeholder text")
    assert score.value != 0.0


async def test_context_precision_still_scores_one_after_the_fix(fake_ragas_llm):
    """Same auto-filler, different metric's verdict field — must be
    unaffected by the noncommittal-specific carve-out. Ragas's own average-
    precision formula lands on 0.9999999999, not exactly 1.0, even with
    every verdict=1 — a pre-existing floating-point quirk of ragas itself,
    not something this fix introduced, hence approx rather than ==.
    """
    llm = fake_ragas_llm()
    metric = ContextPrecision(llm=llm)
    score = await metric.ascore(
        user_input="Why was this cow flagged?",
        retrieved_contexts=["Some retrieved fact."],
        reference="Some retrieved fact.",
    )
    assert score.value == pytest.approx(1.0)
