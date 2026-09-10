from __future__ import annotations

import json

from shared.schemas import AnomalyExplanationQuery
from stage2_rag_assistant.config.settings import GenerationConfig
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record
from stage2_rag_assistant.pipeline.generator import generate_explanation
from stage2_rag_assistant.pipeline.retriever import RetrievalResult

CONFIG = GenerationConfig(max_tokens=512, timeout_s=20.0, max_retries=1)

_VALID_RESPONSE = json.dumps(
    {
        "anomaly_summary": "Lying time has increased sharply.",
        "llm_asserts_anomaly": True,
        "confidence": 0.8,
        "rationale": "Matches the sustained lying-increase pattern.",
        "contributing_signals": ["lying_time"],
        "suggested_next_step": "monitor",
    }
)


def _record_with_signals(signals: list[str]):
    record = generate_mock_record("cow-001", anomaly_flag=True)
    return record.model_copy(update={"driving_signals": signals})


def _query() -> AnomalyExplanationQuery:
    from datetime import datetime, timezone

    return AnomalyExplanationQuery(
        query_id="q-1", cow_id="cow-001", raw_text=None, submitted_by="system_auto", timestamp=datetime.now(timezone.utc)
    )


EMPTY_RETRIEVAL = RetrievalResult(matched_categories=[], is_empty=True)


def test_valid_json_produces_generated_content(fake_llm_client):
    client = fake_llm_client(responses=[_VALID_RESPONSE])
    record = _record_with_signals(["lying_time"])
    result = generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    assert result is not None
    assert result.llm_asserts_anomaly is True
    assert result.confidence == 0.8


def test_one_retry_recovers_from_malformed_output(fake_llm_client):
    client = fake_llm_client(responses=["not json", _VALID_RESPONSE])
    record = _record_with_signals(["lying_time"])
    result = generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    assert result is not None
    assert len(client.call_log) == 2


def test_malformed_twice_returns_none(fake_llm_client):
    client = fake_llm_client(responses=["not json", "still not json"])
    record = _record_with_signals(["lying_time"])
    result = generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    assert result is None
    assert len(client.call_log) == 2  # 1 initial + 1 retry (max_retries=1), then give up


def test_schema_valid_json_missing_a_required_field_also_retries(fake_llm_client):
    incomplete = json.dumps({"anomaly_summary": "x"})  # missing every other required field
    client = fake_llm_client(responses=[incomplete, _VALID_RESPONSE])
    record = _record_with_signals(["lying_time"])
    result = generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    assert result is not None


def test_multi_signal_prompt_includes_the_reasoning_instruction(fake_llm_client):
    client = fake_llm_client(responses=[_VALID_RESPONSE])
    record = _record_with_signals(["cbt", "lying_time"])
    generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    prompt = client.call_log[0]["user_prompt"]
    assert "IMPORTANT" in prompt
    assert "cbt" in prompt and "lying_time" in prompt


def test_single_signal_prompt_omits_the_multi_signal_instruction(fake_llm_client):
    client = fake_llm_client(responses=[_VALID_RESPONSE])
    record = _record_with_signals(["lying_time"])
    generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    prompt = client.call_log[0]["user_prompt"]
    assert "IMPORTANT" not in prompt


def test_prompt_never_leaks_raw_sensor_language_beyond_the_record(fake_llm_client):
    """Non-goals check: the prompt should describe the record's own numbers,
    never something like a raw accelerometer sample stream."""
    client = fake_llm_client(responses=[_VALID_RESPONSE])
    record = _record_with_signals(["cbt"])
    generate_explanation(record, EMPTY_RETRIEVAL, _query(), llm_client=client, config=CONFIG)
    prompt = client.call_log[0]["system_prompt"] + client.call_log[0]["user_prompt"]
    assert "raw sensor" not in prompt.lower() or "never reference raw sensor data" in prompt
