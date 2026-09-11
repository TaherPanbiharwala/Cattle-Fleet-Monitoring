from __future__ import annotations

from datetime import datetime, timezone

import pytest

from shared.schemas import AnomalyExplanationResponse, GoldenCase, Stage1OutputSummary
from stage2_rag_assistant.calibration.bucketing import (
    NUM_BUCKETS,
    aggregate_by_bucket,
    bucket_label,
    evaluate_case,
    reconstruct_llm_asserts_anomaly,
    split_calibration_test,
)
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record


def _case(case_id: str, gold_anomaly_flag: bool = True) -> GoldenCase:
    record = generate_mock_record("cow-x", anomaly_flag=True)
    return GoldenCase(
        case_id=case_id,
        category="injected",
        source="synthetic_injection",
        injection_type=None,
        input_record=record,
        query_text=None,
        gold_anomaly_flag=gold_anomaly_flag,
        gold_driving_signals=record.driving_signals,
        gold_key_facts=[],
        reviewed_by="mechanical",
    )


def _response(
    path_taken: str = "llm_grounded",
    confidence: float | None = 0.75,
    stage1_anomaly_flag: bool = True,
    disagreement_flag: bool = False,
) -> AnomalyExplanationResponse:
    return AnomalyExplanationResponse(
        query_id="q-1",
        cow_id="cow-x",
        path_taken=path_taken,
        anomaly_summary=None,
        confidence=confidence,
        rationale="r",
        cited_facts=[],
        contributing_signals=[],
        suggested_next_step="monitor",
        stage1_output=Stage1OutputSummary(anomaly_flag=stage1_anomaly_flag, anomaly_score=0.8, driving_signals=[]),
        disagreement_flag=disagreement_flag,
        latency_ms=1,
        model_version="test",
        timestamp=datetime.now(timezone.utc),
    )


# ---- split_calibration_test ----


def test_split_is_deterministic():
    cases = [_case(f"c-{i:02d}") for i in range(7)]
    first = split_calibration_test(cases)
    second = split_calibration_test(cases)
    assert [c.case_id for c in first[0]] == [c.case_id for c in second[0]]
    assert [c.case_id for c in first[1]] == [c.case_id for c in second[1]]


def test_split_is_disjoint_and_covers_every_case():
    cases = [_case(f"c-{i:02d}") for i in range(9)]
    calibration, test = split_calibration_test(cases)
    calib_ids, test_ids = {c.case_id for c in calibration}, {c.case_id for c in test}
    assert calib_ids & test_ids == set()
    assert calib_ids | test_ids == {c.case_id for c in cases}


@pytest.mark.parametrize("n,expected_calib,expected_test", [(0, 0, 0), (1, 1, 0), (2, 1, 1), (7, 4, 3), (8, 4, 4)])
def test_split_sizes(n, expected_calib, expected_test):
    cases = [_case(f"c-{i:02d}") for i in range(n)]
    calibration, test = split_calibration_test(cases)
    assert len(calibration) == expected_calib
    assert len(test) == expected_test


# ---- bucket_label ----


@pytest.mark.parametrize(
    "confidence,expected",
    [
        (0.0, "0.0-0.1"),
        (0.05, "0.0-0.1"),
        (0.1, "0.1-0.2"),
        (0.29999999999999999, "0.3-0.4"),  # float noise around a decile boundary
        (0.3, "0.3-0.4"),
        (0.5, "0.5-0.6"),
        (0.75, "0.7-0.8"),
        (0.7999999999999999, "0.8-0.9"),  # float noise just under 0.8
        (0.99, "0.9-1.0"),
        (1.0, "0.9-1.0"),  # last bucket is closed, not [1.0, 1.1)
    ],
)
def test_bucket_label_boundaries(confidence, expected):
    assert bucket_label(confidence) == expected


def test_every_decile_boundary_has_a_home():
    for i in range(NUM_BUCKETS + 1):
        bucket_label(i / NUM_BUCKETS)  # must not raise


# ---- reconstruct_llm_asserts_anomaly ----


@pytest.mark.parametrize(
    "stage1_flag,disagreement_flag,expected",
    [
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, True),
    ],
)
def test_reconstruct_llm_asserts_anomaly(stage1_flag, disagreement_flag, expected):
    response = _response(stage1_anomaly_flag=stage1_flag, disagreement_flag=disagreement_flag)
    assert reconstruct_llm_asserts_anomaly(response) == expected


# ---- evaluate_case ----


@pytest.mark.parametrize("path_taken", ["fallback_stage1_output", "fallback_insufficient_data"])
def test_evaluate_case_excludes_non_llm_grounded_paths(path_taken):
    case = _case("c-01")
    response = _response(path_taken=path_taken, confidence=None)
    assert evaluate_case(case, response) is None


def test_evaluate_case_correct_result():
    case = _case("c-01", gold_anomaly_flag=True)
    response = _response(confidence=0.85, stage1_anomaly_flag=True, disagreement_flag=False)
    result = evaluate_case(case, response)
    assert result is not None
    assert result.bucket == "0.8-0.9"
    assert result.llm_asserts_anomaly is True
    assert result.correct is True


def test_evaluate_case_incorrect_result():
    case = _case("c-01", gold_anomaly_flag=True)
    response = _response(confidence=0.85, stage1_anomaly_flag=False, disagreement_flag=False)
    result = evaluate_case(case, response)
    assert result is not None
    assert result.llm_asserts_anomaly is False  # stage1 said no, no disagreement -> llm also said no
    assert result.correct is False  # gold says True


def test_evaluate_case_never_exposes_stage1_output_shaped_fields():
    case = _case("c-01")
    response = _response()
    result = evaluate_case(case, response)
    assert set(result.model_dump().keys()) == {
        "case_id",
        "scenario_id",
        "bucket",
        "confidence",
        "llm_asserts_anomaly",
        "gold_anomaly_flag",
        "correct",
    }


# ---- aggregate_by_bucket ----


def test_aggregate_empty_input_emits_all_buckets_zeroed():
    stats = aggregate_by_bucket([])
    assert len(stats) == NUM_BUCKETS
    assert all(s.count == 0 and s.correct_count == 0 and s.accuracy is None for s in stats)


def test_aggregate_mixed_bucket_computes_partial_accuracy():
    case = _case("c-01")
    correct = evaluate_case(case, _response(confidence=0.85, stage1_anomaly_flag=True, disagreement_flag=False))
    incorrect = evaluate_case(
        _case("c-02", gold_anomaly_flag=False), _response(confidence=0.85, stage1_anomaly_flag=True, disagreement_flag=False)
    )
    stats_by_bucket = {s.bucket: s for s in aggregate_by_bucket([correct, incorrect])}
    target = stats_by_bucket["0.8-0.9"]
    assert target.count == 2
    assert target.correct_count == 1
    assert target.accuracy == 0.5
    other_buckets = [s for label, s in stats_by_bucket.items() if label != "0.8-0.9"]
    assert all(s.count == 0 for s in other_buckets)
