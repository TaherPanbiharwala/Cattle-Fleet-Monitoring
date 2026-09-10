from __future__ import annotations

import json
from datetime import datetime, timezone

from shared.schemas import AnomalyExplanationQuery, AnomalyExplanationResponse
from stage2_rag_assistant.config.settings import (
    FallbackConfig,
    GenerationConfig,
    KBConfig,
    LLMConfig,
    RouterConfig,
    Stage2Config,
)
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record
from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

TAU = 0.6
MARGIN = 0.3


def _config(db_path) -> Stage2Config:
    return Stage2Config(
        llm=LLMConfig(provider="fake", model="fake-model"),
        router=RouterConfig(max_tokens=32, timeout_s=10.0),
        generation=GenerationConfig(max_tokens=512, timeout_s=20.0, max_retries=1),
        fallback=FallbackConfig(tau=TAU, high_severity_margin=MARGIN),
        kb=KBConfig(db_path=str(db_path)),
    )


def _generated_response(*, llm_asserts_anomaly: bool, confidence: float) -> str:
    return json.dumps(
        {
            "anomaly_summary": "mock summary",
            "llm_asserts_anomaly": llm_asserts_anomaly,
            "confidence": confidence,
            "rationale": "mock rationale",
            "contributing_signals": ["cbt"],
            "suggested_next_step": "monitor",
        }
    )


def _record(driving_signals, *, anomaly_flag: bool, anomaly_score: float):
    record = generate_mock_record("cow-e2e", anomaly_flag=anomaly_flag)
    return record.model_copy(update={"driving_signals": driving_signals, "anomaly_score": anomaly_score})


def test_agree_and_confident_returns_llm_grounded(tmp_kb_db, fake_llm_client):
    record = _record(["cbt"], anomaly_flag=True, anomaly_score=0.9)
    client = fake_llm_client(responses=[_generated_response(llm_asserts_anomaly=True, confidence=0.9)])
    response = run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "llm_grounded"
    _assert_common_invariants(response, record)


def test_agree_but_low_confidence_falls_back_to_stage1(tmp_kb_db, fake_llm_client):
    record = _record(["cbt"], anomaly_flag=True, anomaly_score=0.9)
    client = fake_llm_client(responses=[_generated_response(llm_asserts_anomaly=True, confidence=0.3)])
    response = run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_stage1_output"
    _assert_common_invariants(response, record)


def test_disagreement_with_confident_stage1_overrides_high_llm_confidence(tmp_kb_db, fake_llm_client):
    record = _record(["cbt"], anomaly_flag=True, anomaly_score=0.9)
    client = fake_llm_client(responses=[_generated_response(llm_asserts_anomaly=False, confidence=0.95)])
    response = run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_stage1_output"
    assert response.disagreement_flag is True
    _assert_common_invariants(response, record)


def test_unmapped_driving_signal_returns_insufficient_data(tmp_kb_db, fake_llm_client):
    record = _record(["totally_unknown_signal"], anomaly_flag=True, anomaly_score=0.8)
    client = fake_llm_client(responses=[])  # generator must never be called
    response = run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_insufficient_data"
    assert client.call_log == []
    _assert_common_invariants(response, record)


def test_out_of_scope_query_short_circuits_before_retrieval_or_generation(tmp_kb_db, fake_llm_client):
    record = _record(["cbt"], anomaly_flag=True, anomaly_score=0.9)
    query = AnomalyExplanationQuery(
        query_id="q-oos",
        cow_id=record.cow_id,
        raw_text="what's the weather like today?",
        submitted_by="farmer",
        timestamp=datetime.now(timezone.utc),
    )
    client = fake_llm_client(responses=['{"intent": "out_of_scope"}'])
    response = run_pipeline(record, query, config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_insufficient_data"
    assert len(client.call_log) == 1  # only the router's call — generator never ran
    _assert_common_invariants(response, record)


def test_system_auto_query_with_no_free_text_never_calls_the_router_llm(tmp_kb_db, fake_llm_client):
    record = _record(["cbt"], anomaly_flag=True, anomaly_score=0.9)
    client = fake_llm_client(responses=[_generated_response(llm_asserts_anomaly=True, confidence=0.9)])
    run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=client)
    assert len(client.call_log) == 1  # the generator's call only — router made zero calls


def _assert_common_invariants(response: AnomalyExplanationResponse, record) -> None:
    # Round-trips through the frozen schema exactly as a real API boundary would.
    AnomalyExplanationResponse.model_validate_json(response.model_dump_json())
    assert response.stage1_output.anomaly_flag == record.anomaly_flag  # FR-13
    assert response.stage1_output.anomaly_score == record.anomaly_score
    assert response.latency_ms > 0
