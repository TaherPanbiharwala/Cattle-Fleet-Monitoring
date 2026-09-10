"""Verifies score_case()'s own orchestration (field mapping, calling all 4
real ragas metrics, per-metric error isolation) using FakeRagasLLM's own
default auto-filling responder (see ragas_llm.py). This proves the
plumbing runs end-to-end against the REAL ragas metric classes without
crashing — it is NOT evidence that any produced score reflects real
generation quality. A fake judge that fills in placeholder text can't
meaningfully judge faithfulness or relevancy; only a real LLM judge, run
later once a provider exists, can.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from shared.schemas import AnomalyExplanationResponse, CitedFact, GoldenCase, Stage1OutputSummary
from stage2_rag_assistant.eval.metrics import score_case
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record


def _metrics(fake_ragas_llm, fake_ragas_embedding):
    llm = fake_ragas_llm()  # no responses/responder -> uses the default auto-fill responder
    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=fake_ragas_embedding()),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }


def _case_and_response(cited_facts: list[CitedFact]) -> tuple[GoldenCase, AnomalyExplanationResponse]:
    record = generate_mock_record("cow-001", anomaly_flag=True)
    case = GoldenCase(
        case_id="case-1",
        category="injected",
        source="synthetic_injection",
        injection_type="mock_cbt_deviation",
        input_record=record,
        query_text=None,
        gold_anomaly_flag=True,
        gold_driving_signals=record.driving_signals,
        gold_key_facts=["Elevated core body temperature indicates a possible fever."],
        reviewed_by="mechanical",
    )
    response = AnomalyExplanationResponse(
        query_id="q-1",
        cow_id="cow-001",
        path_taken="llm_grounded",
        anomaly_summary="summary",
        confidence=0.8,
        rationale="Core body temperature is elevated relative to baseline.",
        cited_facts=cited_facts,
        contributing_signals=["cbt"],
        suggested_next_step="monitor",
        stage1_output=Stage1OutputSummary(anomaly_flag=True, anomaly_score=0.8, driving_signals=["cbt"]),
        disagreement_flag=False,
        latency_ms=1,
        model_version="test",
        timestamp=datetime.now(timezone.utc),
    )
    return case, response


async def test_score_case_populates_all_four_metrics(fake_ragas_llm, fake_ragas_embedding):
    case, response = _case_and_response(
        cited_facts=[CitedFact(fact_id="f1", source="literature_links:1", text="Fever indicates illness.")]
    )
    score = await score_case(case, response, **_metrics(fake_ragas_llm, fake_ragas_embedding))
    assert score.case_id == "case-1"
    assert score.errors == {}
    assert score.faithfulness is not None
    assert score.answer_relevancy is not None
    assert score.context_precision is not None
    assert score.context_recall is not None


async def test_score_case_isolates_a_failing_metric_instead_of_aborting(fake_ragas_llm, fake_ragas_embedding):
    """Empty cited_facts -> empty retrieved_contexts. Faithfulness/
    ContextPrecision/ContextRecall all require non-empty retrieved_contexts
    (confirmed by reading ragas source) and raise; AnswerRelevancy doesn't
    take retrieved_contexts at all and should still succeed.
    """
    case, response = _case_and_response(cited_facts=[])
    score = await score_case(case, response, **_metrics(fake_ragas_llm, fake_ragas_embedding))
    assert score.answer_relevancy is not None
    assert "faithfulness" in score.errors
    assert "context_precision" in score.errors
    assert "context_recall" in score.errors
    assert score.faithfulness is None
    assert score.context_precision is None
    assert score.context_recall is None
