"""M1a/M1d contracts: safety, grouped evaluation, and fail-closed fusion."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from stage1_anomaly_detection.behavior_classifier.context import (
    BehaviorDailyContext,
    CusumDailyResult,
    adapt_cusum_window,
    aggregate_daily_context,
    fuse_daily_records,
)
from stage1_anomaly_detection.behavior_classifier.errors import BehaviorError
from stage1_anomaly_detection.behavior_classifier.wasp import CLASS_NAMES, PLATFORM_CODES
from stage2_rag_assistant.pipeline.record_assembler import assemble


def _context(cow: str, day: str, *, state: str = "walking", deployment: str = "runtime-a") -> dict[str, object]:
    return BehaviorDailyContext(
        cow_id=cow,
        local_date=day,
        timezone="America/Chicago",
        deployment_id=deployment,
        source_kind="same_cow_runtime",
        model_sha256="a" * 64,
        behavior_state=state,
        behavior_state_confidence=0.8,
        behavior_state_distribution_24h={"walking": 1.0 if state == "walking" else 0.0, "grazing": 1.0 if state == "grazing" else 0.0, "resting": 1.0 if state == "resting" else 0.0, "miscellaneous": 1.0 if state == "miscellaneous" else 0.0},
        observed_windows=8,
        expected_windows=8,
        coverage=1.0,
    ).model_dump(mode="json")


def _cusum(cow: str, day: str, *, deployment: str = "runtime-a") -> dict[str, object]:
    return CusumDailyResult(
        cow_id=cow,
        local_date=day,
        timezone="America/Chicago",
        deployment_id=deployment,
        source_kind="same_cow_runtime",
        window_id=f"{cow}:{day}",
        cbt_c=38.5,
        cbt_deviation_sigma=1.3,
        cbt_cusum_value=5.1,
        lying_time_pct_24h=42.0,
        lying_time_deviation_sigma=-0.2,
        thi=68.0,
        activity_magnitude_deviation_sigma=None,
        anomaly_flag=True,
        anomaly_score=0.51,
        driving_signals=["cbt"],
    ).model_dump(mode="json")


def test_safe_class_mapping_never_uses_restless() -> None:
    assert CLASS_NAMES == ("resting", "grazing", "walking", "miscellaneous")
    assert PLATFORM_CODES == {"resting": 0, "grazing": 1, "walking": 3, "miscellaneous": 5}
    assert 4 not in PLATFORM_CODES.values()


def test_daily_context_requires_coverage_and_uses_stable_tie_break() -> None:
    prediction_rows = [
        {"cow_id": "T01", "timestamp": "2023-08-01T01:00:00+00:00", "deployment_id": "runtime-a", "source_kind": "same_cow_runtime", "model_sha256": "a" * 64, "behavior_state": "walking", "behavior_state_confidence": 0.8},
        {"cow_id": "T01", "timestamp": "2023-08-01T01:00:05+00:00", "deployment_id": "runtime-a", "source_kind": "same_cow_runtime", "model_sha256": "a" * 64, "behavior_state": "grazing", "behavior_state_confidence": 0.7},
    ]
    context = aggregate_daily_context(prediction_rows, timezone="UTC", expected_windows=2)[0]
    assert context["behavior_state"] == "grazing"  # lexical tie break
    assert context["coverage"] == 1.0
    with pytest.raises(BehaviorError, match="CONTEXT_COVERAGE_INSUFFICIENT"):
        aggregate_daily_context(prediction_rows, timezone="UTC", expected_windows=3)


def test_fusion_is_schema_valid_preserves_cusum_and_keeps_history_per_cow() -> None:
    contexts = [_context("T01", "2023-08-01"), _context("T01", "2023-08-02"), _context("T02", "2023-08-01", state="grazing")]
    cusums = [_cusum("T01", "2023-08-01"), _cusum("T01", "2023-08-02"), _cusum("T02", "2023-08-01")]
    records = fuse_daily_records(contexts, cusums)
    assert records[0]["timestamp"] == "2023-08-02T05:00:00Z"
    assert records[1]["recent_anomaly_history"] == [{"timestamp": records[0]["timestamp"], "flag": True, "driving_signals": ["cbt"]}]
    assert records[2]["recent_anomaly_history"] == []
    assert records[1]["driving_signals"] == ["cbt"]
    assert "behavior_state" not in records[1]["driving_signals"]
    assert assemble(records[1], None).record.cow_id == "T01"


def test_fusion_rejects_public_sources_duplicates_and_behavior_driver() -> None:
    public = _context("T01", "2023-08-01")
    public["source_kind"] = "wasp_public"
    with pytest.raises(BehaviorError, match="CONTEXT_SOURCE_NOT_RUNTIME"):
        fuse_daily_records([public], [_cusum("T01", "2023-08-01")])
    with pytest.raises(BehaviorError, match="CONTEXT_ALIGNMENT_FAILED"):
        fuse_daily_records([_context("T01", "2023-08-01"), _context("T01", "2023-08-01")], [_cusum("T01", "2023-08-01")])
    invalid = _cusum("T01", "2023-08-01")
    invalid["driving_signals"] = ["behavior_state"]
    with pytest.raises(BehaviorError, match="CONTEXT_ALIGNMENT_FAILED"):
        fuse_daily_records([_context("T01", "2023-08-01")], [invalid])


def test_existing_cusum_adapter_marks_current_public_data_ineligible() -> None:
    raw = {"cow_id": "T01", "window_id": "T01:2023-08-01", "date": "2023-08-01", "cbt_c": 38.5, "cbt_deviation_sigma": 1.0, "cbt_cusum_value": 5.0, "lying_time_pct_24h": 40.0, "lying_time_deviation_sigma": 0.0, "thi": 69.0, "activity_magnitude_deviation_sigma": None, "anomaly_flag": True, "anomaly_score": 0.5, "driving_signals": ["cbt"]}
    adapted = adapt_cusum_window(raw, deployment_id="mmcows-2023", source_kind="mmcows_public", timezone="America/Chicago")
    with pytest.raises(BehaviorError, match="CONTEXT_SOURCE_NOT_RUNTIME"):
        fuse_daily_records([_context("T01", "2023-08-01", deployment="mmcows-2023")], [adapted])


def _write_wasp_event(path: Path, *, label: str, cow: str, event: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Time", "MPU9250_AX", "MPU9250_AY", "MPU9250_AZ", "MPU9250_GX", "MPU9250_GY", "MPU9250_GZ"])
        label_index = CLASS_NAMES.index(label)
        start = datetime(2023, 8, 1, tzinfo=timezone.utc)
        for index in range(75):
            base = label_index * 10.0
            writer.writerow([(start + timedelta(seconds=index / 10)).isoformat(), base + index / 100, base + 1, base + 2, base + 3, base + 4, base + 5])


@pytest.mark.skipif(__import__("importlib").util.find_spec("xgboost") is None, reason="behavior optional dependencies are not installed")
def test_feature_artifact_and_nested_loco_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np

    from stage1_anomaly_detection.behavior_classifier import benchmark
    from stage1_anomaly_detection.behavior_classifier.artifact import verify_artifact
    from stage1_anomaly_detection.behavior_classifier.errors import BehaviorError
    from stage1_anomaly_detection.behavior_classifier.features import FEATURE_NAMES, extract_features
    from stage1_anomaly_detection.behavior_classifier.runtime import predict_runtime_windows

    matrix = np.arange(300, dtype=float).reshape(50, 6)
    first = extract_features(matrix)
    assert first.shape == (112,)
    assert len(FEATURE_NAMES) == 112
    assert np.array_equal(first, extract_features(matrix))
    dataset = tmp_path / "wasp"
    folders = {"resting": "Resting", "grazing": "Grazing", "walking": "Walking", "miscellaneous": "Miscellaneous behaviors"}
    for cow_number in range(1, 5):
        for event_number, (label, folder) in enumerate(folders.items(), start=1):
            _write_wasp_event(dataset / folder / f"{event_number}_{label}_C{cow_number:02d}_20230801_000000.csv", label=label, cow=f"C{cow_number:02d}", event=event_number)
    monkeypatch.setattr(benchmark, "XGBOOST_GRID", ({"n_estimators": 3, "max_depth": 2, "learning_rate": 0.3, "subsample": 1.0, "colsample_bytree": 1.0},))
    monkeypatch.setattr(benchmark, "INNER_SPLITS", 2)
    result = benchmark.run_benchmark(dataset, tmp_path / "benchmark")
    report = json.loads((result.output_dir / "benchmark_report.json").read_text(encoding="utf-8"))
    assert report["evaluation_protocol"].startswith("nested leave-one-cow-out")
    assert (result.output_dir / "fold_metrics.jsonl").is_file()
    assert result.artifact_eligible
    verified = verify_artifact(result.output_dir / "behavior_model.json")
    assert verified["status"] == "verified"
    runtime_input = {"cow_id": "T01", "timestamp": "2023-08-01T00:00:00+00:00", "deployment_id": "runtime-a", "source_kind": "same_cow_runtime", "sample_rate_hz": 10.0, "acceleration_unit": "m_s2", "gyroscope_unit": "deg_s", "calibration_id": "cal-001", "sample_timestamps": [(datetime(2023, 8, 1, tzinfo=timezone.utc) + timedelta(seconds=index / 10)).isoformat() for index in range(50)], "samples": matrix.tolist()}
    predicted = predict_runtime_windows(
        [runtime_input],
        model_path=result.output_dir / "behavior_model.json",
    )
    assert "samples" not in predicted[0]
    invalid_gap = dict(runtime_input)
    invalid_gap["sample_timestamps"] = list(runtime_input["sample_timestamps"])
    invalid_gap["sample_timestamps"][10] = "2023-08-01T00:00:05+00:00"
    with pytest.raises(BehaviorError, match="FEATURE_CONTRACT"):
        predict_runtime_windows([invalid_gap], model_path=result.output_dir / "behavior_model.json")
    missing_contract_field = dict(runtime_input)
    missing_contract_field.pop("calibration_id")
    with pytest.raises(BehaviorError, match="RUNTIME_CONTEXT_INVALID") as raw_error:
        predict_runtime_windows([missing_contract_field], model_path=result.output_dir / "behavior_model.json")
    assert "samples" not in json.dumps(raw_error.value.details)
    with pytest.raises(BehaviorError, match="OUTPUT_EXISTS"):
        benchmark.run_benchmark(dataset, result.output_dir)
    with (result.output_dir / "behavior_model.json").open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(BehaviorError, match="ARTIFACT_HASH_MISMATCH"):
        verify_artifact(result.output_dir / "behavior_model.json")


def test_no_root_ml_import_or_pickle_loading() -> None:
    package = Path(__file__).parents[2] / "stage1_anomaly_detection" / "behavior_classifier"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "src.ml" not in source
    assert "pickle.load" not in source


def test_cli_uses_json_error_and_refuses_legacy_pickle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from stage1_anomaly_detection.behavior_classifier.cli import main

    pickle_path = tmp_path / "unsafe.pkl"
    pickle_path.write_bytes(b"not executed")
    assert main(["verify-artifact", "--model-path", str(pickle_path)]) == 2
    response = json.loads(capsys.readouterr().err)
    assert response["code"] == "LEGACY_PICKLE_REJECTED"
