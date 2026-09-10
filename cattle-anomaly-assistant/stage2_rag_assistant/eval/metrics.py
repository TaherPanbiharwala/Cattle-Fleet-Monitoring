"""PRD Layer 2 (Ragas) field mapping, per-case scoring, and aggregation.
See LLM_Diagnostic_Assistant_PRD.md Section 11.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

from pydantic import Field

from shared.schemas import AnomalyExplanationResponse, GoldenCase
from shared.schemas._base import StrictModel

if TYPE_CHECKING:
    from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


class RagasInputs(StrictModel):
    user_input: str
    response: str
    retrieved_contexts: list[str]
    reference: str


def case_to_metric_inputs(case: GoldenCase, response: AnomalyExplanationResponse) -> RagasInputs:
    """PRD §11: faithfulness/answer-relevancy off `rationale` + `cited_facts`
    (response = .rationale, not .anomaly_summary — a different field with a
    different audience); context precision/recall against `gold_key_facts`.
    Ragas's reference field is a single string, but gold_key_facts is a
    list — newline-joined so each fact reads as a distinct sentence.
    """
    user_input = case.query_text or f"Explain the flagged anomaly for cow {case.input_record.cow_id}."
    return RagasInputs(
        user_input=user_input,
        response=response.rationale,
        retrieved_contexts=[fact.text for fact in response.cited_facts],
        reference="\n".join(case.gold_key_facts),
    )


class CaseScore(StrictModel):
    case_id: str
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    errors: dict[str, str] = Field(default_factory=dict)


async def score_case(
    case: GoldenCase,
    response: AnomalyExplanationResponse,
    *,
    faithfulness: "Faithfulness",
    answer_relevancy: "AnswerRelevancy",
    context_precision: "ContextPrecision",
    context_recall: "ContextRecall",
) -> CaseScore:
    """Only called for path_taken == 'llm_grounded' responses — see
    run_eval.py for why fallback-path responses are excluded before this
    is ever invoked. Each metric's failure is caught independently so one
    bad metric doesn't discard the other three (mirrors
    generator.generate_explanation's 'never raise on an expected failure
    mode' philosophy).
    """
    inputs = case_to_metric_inputs(case, response)
    scores: dict[str, float] = {}
    errors: dict[str, str] = {}

    async def _run(name: str, coro) -> None:
        try:
            result = await coro
            scores[name] = result.value
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            errors[name] = str(exc)

    await _run(
        "faithfulness",
        faithfulness.ascore(
            user_input=inputs.user_input, response=inputs.response, retrieved_contexts=inputs.retrieved_contexts
        ),
    )
    await _run(
        "answer_relevancy",
        answer_relevancy.ascore(user_input=inputs.user_input, response=inputs.response),
    )
    await _run(
        "context_precision",
        context_precision.ascore(
            user_input=inputs.user_input, reference=inputs.reference, retrieved_contexts=inputs.retrieved_contexts
        ),
    )
    await _run(
        "context_recall",
        context_recall.ascore(
            user_input=inputs.user_input, retrieved_contexts=inputs.retrieved_contexts, reference=inputs.reference
        ),
    )

    return CaseScore(case_id=case.case_id, errors=errors, **scores)


def aggregate(scores: list[CaseScore]) -> dict:
    result: dict = {"total_cases_scored": len(scores)}
    for name in _METRIC_NAMES:
        values = [v for s in scores if (v := getattr(s, name)) is not None]
        result[name] = {
            "mean": statistics.mean(values) if values else None,
            "median": statistics.median(values) if values else None,
            "count": len(values),
        }
    error_counts: dict[str, int] = {}
    for s in scores:
        for name in s.errors:
            error_counts[name] = error_counts.get(name, 0) + 1
    result["errors_by_metric"] = error_counts
    return result
