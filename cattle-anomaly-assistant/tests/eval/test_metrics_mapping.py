from __future__ import annotations

from datetime import datetime, timezone

from shared.schemas import AnomalyExplanationResponse, CitedFact, GoldenCase, Stage1OutputSummary
from stage2_rag_assistant.eval.metrics import case_to_metric_inputs
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record


def _case(query_text: str | None, gold_key_facts: list[str]) -> GoldenCase:
    record = generate_mock_record("cow-001", anomaly_flag=True)
    return GoldenCase(
        case_id="case-1",
        category="injected",
        source="synthetic_injection",
        injection_type="mock_cbt_deviation",
        input_record=record,
        query_text=query_text,
        gold_anomaly_flag=True,
        gold_driving_signals=record.driving_signals,
        gold_key_facts=gold_key_facts,
        reviewed_by="mechanical",
    )


def _response(rationale: str, anomaly_summary: str, cited_facts: list[CitedFact]) -> AnomalyExplanationResponse:
    return AnomalyExplanationResponse(
        query_id="q-1",
        cow_id="cow-001",
        path_taken="llm_grounded",
        anomaly_summary=anomaly_summary,
        confidence=0.8,
        rationale=rationale,
        cited_facts=cited_facts,
        contributing_signals=["cbt"],
        suggested_next_step="monitor",
        stage1_output=Stage1OutputSummary(anomaly_flag=True, anomaly_score=0.8, driving_signals=["cbt"]),
        disagreement_flag=False,
        latency_ms=1,
        model_version="test",
        timestamp=datetime.now(timezone.utc),
    )


def test_response_field_uses_rationale_not_anomaly_summary():
    case = _case(None, ["fact one"])
    response = _response(rationale="THE RATIONALE", anomaly_summary="THE SUMMARY", cited_facts=[])
    inputs = case_to_metric_inputs(case, response)
    assert inputs.response == "THE RATIONALE"
    assert "THE SUMMARY" not in inputs.response


def test_retrieved_contexts_comes_from_cited_facts_text():
    case = _case(None, ["fact one"])
    facts = [
        CitedFact(fact_id="f1", source="literature_links:1", text="fact text one"),
        CitedFact(fact_id="f2", source="literature_links:2", text="fact text two"),
    ]
    response = _response(rationale="r", anomaly_summary="s", cited_facts=facts)
    inputs = case_to_metric_inputs(case, response)
    assert inputs.retrieved_contexts == ["fact text one", "fact text two"]


def test_reference_is_newline_joined_gold_key_facts():
    case = _case(None, ["first fact.", "second fact."])
    response = _response(rationale="r", anomaly_summary="s", cited_facts=[])
    inputs = case_to_metric_inputs(case, response)
    assert inputs.reference == "first fact.\nsecond fact."


def test_user_input_falls_back_to_template_when_query_text_is_none():
    case = _case(None, ["fact"])
    response = _response(rationale="r", anomaly_summary="s", cited_facts=[])
    inputs = case_to_metric_inputs(case, response)
    assert "cow-001" in inputs.user_input


def test_user_input_uses_query_text_when_present():
    case = _case("why was she flagged?", ["fact"])
    response = _response(rationale="r", anomaly_summary="s", cited_facts=[])
    inputs = case_to_metric_inputs(case, response)
    assert inputs.user_input == "why was she flagged?"
