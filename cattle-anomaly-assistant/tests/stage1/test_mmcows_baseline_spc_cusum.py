from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage1_anomaly_detection.baseline_spc_cusum.cli import main
from stage1_anomaly_detection.baseline_spc_cusum.config import DetectorConfig
from stage1_anomaly_detection.baseline_spc_cusum.daily import aggregate_daily_features
from stage1_anomaly_detection.baseline_spc_cusum.detector import detect, learn_baselines
from stage1_anomaly_detection.baseline_spc_cusum.errors import DetectorError
from stage1_anomaly_detection.baseline_spc_cusum.injection import apply_injections
from stage1_anomaly_detection.baseline_spc_cusum.mmcows import load_mmcows_inputs
from stage1_anomaly_detection.baseline_spc_cusum.pipeline import publish_output, run_detector

from .conftest import make_mmcows_subset


def _features(root: Path):
    config = DetectorConfig()
    inputs = load_mmcows_inputs(root, config)
    return aggregate_daily_features(inputs, config)


def test_adapter_filters_stationary_tags_converts_time_and_broadcasts_shared_thi(mmcows_subset: Path) -> None:
    features = _features(mmcows_subset)

    assert {feature.cow_id for feature in features} == {"T01"}
    assert features[0].day.isoformat() == "2023-07-22"
    assert all(feature.thi == 72.0 for feature in features)
    assert all(feature.quality["thi"].valid for feature in features)


def test_daily_coverage_marks_missing_windows_invalid_without_imputation(tmp_path: Path) -> None:
    features = _features(make_mmcows_subset(tmp_path / "missing", missing_cbt_day=3))
    missing_day = features[3]

    assert missing_day.cbt_c == pytest.approx(38.3)
    assert missing_day.quality["cbt"].coverage == pytest.approx(0.5)
    assert not missing_day.quality["cbt"].valid
    assert not missing_day.core_valid


def test_require_immu_fails_when_optional_stream_is_not_staged(tmp_path: Path) -> None:
    data_root = make_mmcows_subset(tmp_path / "no-immu", include_immu=False)
    with pytest.raises(DetectorError, match="MISSING_IMMU"):
        load_mmcows_inputs(data_root, DetectorConfig(), require_immu=True)


def test_adapter_accepts_public_per_tag_immu_layout_and_acceleration_units(tmp_path: Path) -> None:
    data_root = make_mmcows_subset(tmp_path / "public-immu", immu_layout="public_per_tag")
    immu_path = data_root / "main_data" / "immu" / "T01" / "T01_0722.csv"
    with immu_path.open("a", encoding="utf-8") as handle:
        handle.write("1690066800,nan,nan,nan\n")

    inputs = load_mmcows_inputs(data_root, DetectorConfig(), require_immu=True)
    features = aggregate_daily_features(inputs, DetectorConfig())

    assert inputs.immu is not None
    assert inputs.available_streams == ["cbt", "ankle", "thi", "immu"]
    assert all(feature.quality["activity_magnitude"].valid for feature in features)


def test_first_seven_consecutive_days_are_immutable_personal_baseline(mmcows_subset: Path) -> None:
    features = _features(mmcows_subset)
    baselines = learn_baselines(features, DetectorConfig())
    cbt = next(baseline for baseline in baselines if baseline.signal == "cbt")

    assert cbt.available
    assert cbt.start_day == features[0].day
    assert cbt.end_day == features[6].day
    assert cbt.mean == pytest.approx(38.3)
    assert cbt.stddev is not None and cbt.stddev > 0


def test_zero_variance_baseline_is_unavailable_not_divided_by_epsilon(tmp_path: Path) -> None:
    features = _features(make_mmcows_subset(tmp_path / "constant", constant_cbt=True))
    result = detect(features, learn_baselines(features, DetectorConfig()), DetectorConfig())
    baseline = next(item for item in result.baselines if item.signal == "cbt")
    cbt_traces = [trace for trace in result.traces if trace.signal == "cbt"]

    assert not baseline.available
    assert baseline.reason == "zero_variance_baseline"
    assert all(trace.state == "unavailable" for trace in cbt_traces)


def test_cusum_injection_crosses_conservative_threshold_on_third_day(mmcows_subset: Path, tmp_path: Path) -> None:
    features = _features(mmcows_subset)
    baselines = learn_baselines(features, DetectorConfig())
    injection_path = tmp_path / "inject.json"
    injection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "scenario_id": "cbt-rise",
                        "cow_id": "T01",
                        "start_date": features[7].day.isoformat(),
                        "duration_days": 3,
                        "changes": [{"signal": "cbt", "standard_deviations": 2.5}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    injected, report = apply_injections(features, baselines, injection_path)
    result = detect(injected, baselines, DetectorConfig())
    cbt_traces = [trace for trace in result.traces if trace.signal == "cbt" and trace.state == "monitoring"]

    assert report is not None and report.modified_windows == 3
    assert [trace.active_cusum for trace in cbt_traces] == pytest.approx([2.0, 4.0, 6.0])
    final_window = result.windows[-1]
    assert final_window["anomaly_flag"]
    assert final_window["driving_signals"] == ["cbt"]
    assert final_window["anomaly_score"] == pytest.approx(0.6)


def test_negative_lying_and_activity_injections_preserve_direction_and_attribution(mmcows_subset: Path, tmp_path: Path) -> None:
    features = _features(mmcows_subset)
    baselines = learn_baselines(features, DetectorConfig())
    injection_path = tmp_path / "combined.json"
    injection_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "scenarios": [
                    {
                        "scenario_id": "combined-decrease",
                        "cow_id": "T01",
                        "start_date": features[7].day.isoformat(),
                        "duration_days": 3,
                        "changes": [
                            {"signal": "lying_time", "standard_deviations": -2.5},
                            {"signal": "activity_magnitude", "standard_deviations": -2.5},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    injected, _ = apply_injections(features, baselines, injection_path)
    result = detect(injected, baselines, DetectorConfig())
    final_window = result.windows[-1]
    final_traces = {trace.signal: trace for trace in result.traces if trace.day == injected[-1].day}

    assert final_window["driving_signals"] == ["lying_time", "activity_magnitude"]
    assert final_traces["lying_time"].direction == "below"
    assert final_traces["activity_magnitude"].direction == "below"
    assert all(0 <= window["anomaly_score"] <= 1 for window in result.windows)


def test_cusum_resets_after_an_invalid_monitoring_day(mmcows_subset: Path) -> None:
    features = _features(mmcows_subset)
    baselines = learn_baselines(features, DetectorConfig())
    # A valid shift before a bad day must not bridge into the next day's CUSUM.
    cbt_baseline = next(baseline for baseline in baselines if baseline.signal == "cbt")
    assert cbt_baseline.stddev is not None
    features[7].cbt_c = cbt_baseline.mean + 2.5 * cbt_baseline.stddev
    features[8].quality["cbt"] = features[8].quality["cbt"].__class__(0, 4, 0.0, False, "fixture_gap")
    features[9].cbt_c = cbt_baseline.mean + 2.5 * cbt_baseline.stddev
    result = detect(features, baselines, DetectorConfig())
    cbt_traces = [trace for trace in result.traces if trace.signal == "cbt" and trace.state != "baseline"]

    assert cbt_traces[-1].active_cusum == pytest.approx(2.0)
    assert cbt_traces[-1].reset_reason == "previous_monitoring_window_invalid"


def test_control_scenario_is_clean_and_invalid_injection_is_rejected(mmcows_subset: Path, tmp_path: Path) -> None:
    features = _features(mmcows_subset)
    baselines = learn_baselines(features, DetectorConfig())
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"schema_version": 1, "scenarios": [{"scenario_id": "control", "cow_id": "T01", "start_date": features[7].day.isoformat(), "duration_days": 3, "changes": []}]}),
        encoding="utf-8",
    )
    unchanged, report = apply_injections(features, baselines, control)
    assert report is not None and report.modified_windows == 0
    assert all(not feature.injected_signals for feature in unchanged)
    assert not any(window["anomaly_flag"] for window in detect(unchanged, baselines, DetectorConfig()).windows)

    invalid = tmp_path / "bad.json"
    invalid.write_text(
        json.dumps({"schema_version": 1, "scenarios": [{"scenario_id": "bad", "cow_id": "T01", "start_date": features[0].day.isoformat(), "duration_days": 1, "changes": [{"signal": "cbt", "standard_deviations": 2.5}]}]}),
        encoding="utf-8",
    )
    with pytest.raises(DetectorError, match="INJECTION_TOUCHES_BASELINE"):
        apply_injections(features, baselines, invalid)


def test_output_is_derived_only_atomic_and_requires_explicit_overwrite(mmcows_subset: Path, tmp_path: Path) -> None:
    config = DetectorConfig()
    result = run_detector(mmcows_subset, config)
    output_dir = tmp_path / "results"
    published = publish_output(result, config, data_root=mmcows_subset, output_dir=output_dir)

    assert published == output_dir
    assert {path.name for path in output_dir.iterdir()} == {"anomaly_windows.jsonl", "baselines.jsonl", "cusum_traces.jsonl", "daily_features.jsonl", "manifest.json"}
    assert '"raw_sensor_rows_persisted": false' in (output_dir / "manifest.json").read_text(encoding="utf-8")
    with pytest.raises(DetectorError, match="OUTPUT_EXISTS"):
        publish_output(result, config, data_root=mmcows_subset, output_dir=output_dir)
    assert not list(tmp_path.glob(".results.tmp-*"))


def test_cli_preflight_and_safe_output_directory(mmcows_subset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--data-root", str(mmcows_subset), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "validated"
    assert main(["--data-root", str(mmcows_subset), "--output-dir", str(mmcows_subset)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "UNSAFE_OUTPUT_DIRECTORY"


def test_cli_rejects_ambiguous_headers_without_guessing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    data_root = tmp_path / "bad-headers"
    csv_path = data_root / "main_data" / "cbt" / "T01.csv"
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("when,mystery\n1690000000,38.5\n", encoding="utf-8")

    assert main(["--data-root", str(data_root), "--validate-only"]) == 2
    error = json.loads(capsys.readouterr().err)
    assert error["code"] == "UNSUPPORTED_HEADERS"
