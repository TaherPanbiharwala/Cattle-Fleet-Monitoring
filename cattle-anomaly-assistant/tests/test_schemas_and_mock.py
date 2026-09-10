"""M0 'done when' check (PRD Section 14): both tracks can import shared/schemas
and the mock generator produces valid AnomalyRecords.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.schemas import (
    AnomalyExplanationQuery,
    AnomalyExplanationResponse,
    AnomalyRecord,
    CitedFact,
    GoldenCase,
    Stage1OutputSummary,
)
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_batch, generate_mock_record


def test_mock_batch_is_schema_valid_and_realistic():
    records = generate_mock_batch(200, seed=42)

    assert all(isinstance(r, AnomalyRecord) for r in records)
    # Round-trip through JSON, since that's how Stage 2 will actually receive
    # records over an API boundary, not as live Python objects.
    for record in records:
        AnomalyRecord.model_validate_json(record.model_dump_json())

    anomaly_rate = sum(r.anomaly_flag for r in records) / len(records)
    assert 0.05 < anomaly_rate < 0.30, f"expected a minority-anomaly distribution, got {anomaly_rate:.2f}"

    flagged = [r for r in records if r.anomaly_flag]
    assert all(1 <= len(r.driving_signals) <= 2 for r in flagged)
    assert all(r.driving_signals == [] for r in records if not r.anomaly_flag)

    assert any(r.activity_magnitude_deviation_sigma is None for r in records)
    assert any(r.herd_isolation_score is None for r in records)
    # AnomalyHistoryEntry is part of the frozen contract but the generator
    # used to hardcode it empty — assert it's actually exercised non-empty.
    assert any(r.recent_anomaly_history for r in records)


def test_mock_batch_is_reproducible_for_a_given_seed():
    first = generate_mock_batch(20, seed=7)
    second = generate_mock_batch(20, seed=7)
    assert [r.model_dump_json() for r in first] == [r.model_dump_json() for r in second]


def test_schema_rejects_malformed_records():
    valid = generate_mock_record("cow-001").model_dump()
    with pytest.raises(ValidationError):
        AnomalyRecord.model_validate({**valid, "anomaly_score": 1.5})
    with pytest.raises(ValidationError):
        AnomalyRecord.model_validate({**valid, "behavior_state": "sleeping"})


def test_schema_rejects_unknown_fields():
    """Pydantic's default (extra='ignore') would silently drop a typo'd field
    instead of erroring — StrictModel (extra='forbid') is what makes that loud.
    """
    valid = generate_mock_record("cow-001").model_dump()
    with pytest.raises(ValidationError):
        AnomalyRecord.model_validate({**valid, "unexpected_field_that_does_not_exist": 1})


def test_schema_rejects_naive_timestamps():
    valid = generate_mock_record("cow-001").model_dump()
    with pytest.raises(ValidationError):
        AnomalyRecord.model_validate({**valid, "timestamp": "2026-01-01T00:00:00"})  # no tz offset


def test_schema_rejects_behavior_distribution_not_summing_to_one():
    valid = generate_mock_record("cow-001").model_dump()
    bad_distribution = {"walking": 1.0, "grazing": 1.0, "resting": 1.0, "miscellaneous": 1.0}
    with pytest.raises(ValidationError):
        AnomalyRecord.model_validate({**valid, "behavior_state_distribution_24h": bad_distribution})


def test_golden_case_composes_with_an_anomaly_record():
    record = generate_mock_record("cow-002", anomaly_flag=True)
    case = GoldenCase(
        case_id="case-0001",
        category="injected",
        source="synthetic_injection",
        injection_type="grazing_drop_30pct_3day",
        input_record=record,
        query_text=None,
        gold_anomaly_flag=True,
        gold_driving_signals=record.driving_signals,
        gold_key_facts=["Sustained grazing-time reduction is associated with early illness onset."],
        reviewed_by="mechanical",
    )
    assert case.input_record.cow_id == "cow-002"


def test_explanation_query_and_response_construct():
    now = datetime.now(timezone.utc)
    query = AnomalyExplanationQuery(
        query_id="q-0001",
        cow_id="cow-003",
        raw_text=None,
        submitted_by="system_auto",
        timestamp=now,
    )
    response = AnomalyExplanationResponse(
        query_id=query.query_id,
        cow_id=query.cow_id,
        path_taken="llm_grounded",
        anomaly_summary="Grazing time dropped sharply over the last 3 days.",
        confidence=0.82,
        rationale="Sustained grazing-time reduction matches a known early-illness pattern.",
        cited_facts=[
            CitedFact(
                fact_id="fact-001",
                source="literature_links:12",
                text="Sustained grazing-time reduction over 3+ days is an early indicator of illness onset.",
            )
        ],
        contributing_signals=["behavior_state"],
        suggested_next_step="monitor",
        stage1_output=Stage1OutputSummary(anomaly_flag=True, anomaly_score=0.78, driving_signals=["behavior_state"]),
        disagreement_flag=False,
        latency_ms=842,
        model_version="mock-0.0.0",
        timestamp=now,
    )
    assert response.query_id == query.query_id
    assert response.cited_facts[0].fact_id == "fact-001"
