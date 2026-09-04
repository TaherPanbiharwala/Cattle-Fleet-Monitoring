"""Unit tests for the raw-data boundary and deterministic Phase 3 features."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

np = pytest.importorskip("numpy", reason="Phase 3 feature tests require the optional ml dependencies.")

from dataset_adapters.wasp_lab import (
    DatasetValidationError,
    ImuSample,
    WaspEvent,
    create_dataset_manifest,
    load_wasp_dataset,
)
from ml.benchmark import (
    BenchmarkConfig,
    _aggregate_per_class_metrics,
    _assert_group_integrity,
    _build_run_config,
    _fit_classical_model,
    _fit_cnn_final_model,
    _ml_imports,
)
from ml.features import FEATURE_NAMES, WINDOW_SAMPLES, extract_window_features, segment_event


def _write_event(path, *, rows: int = 50, nonfinite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "Time,MPU9250_AX,MPU9250_AY,MPU9250_AZ,MPU9250_GX,MPU9250_GY,MPU9250_GZ\n"
    values = [header]
    started = datetime(2024, 5, 15, 13, 30, 48)
    for index in range(rows):
        timestamp = started + timedelta(milliseconds=100 * index)
        accel_x = "nan" if nonfinite and index == 3 else str(index / 10)
        values.append(f"{timestamp.isoformat(sep=' ')},{accel_x},2,3,4,5,6\n")
    path.write_text("".join(values), encoding="utf-8")


@pytest.fixture
def wasp_dir(tmp_path):
    dataset = tmp_path / "wasp"
    _write_event(dataset / "Resting" / "1_Resting_cow-a_20240515_133048.csv")
    _write_event(dataset / "Grazing" / "2_Grazing_cow-b_20240515_133048.csv")
    _write_event(dataset / "Walking" / "3_Walking_cow-c_20240515_133048.csv")
    _write_event(dataset / "Miscellaneous behaviors" / "4_Miscellaneous_cow-d_20240515_133048.csv")
    return dataset


def test_load_wasp_dataset_maps_only_safe_behaviour_codes(wasp_dir) -> None:
    events = load_wasp_dataset(wasp_dir)

    assert [event.behaviour_code for event in events] == [1, 5, 0, 3]
    assert {event.behaviour_code for event in events} == {0, 1, 3, 5}
    assert all(len(event.samples) == 50 for event in events)


def test_wasp_manifest_contains_hashes_not_raw_sample_values(wasp_dir) -> None:
    manifest = create_dataset_manifest(wasp_dir)

    assert manifest["raw_samples_embedded"] is False
    assert manifest["file_count"] == 4
    assert len(manifest["combined_sha256"]) == 64
    assert "MPU9250_AX" not in str(manifest)


def test_adapter_rejects_nonfinite_samples(wasp_dir) -> None:
    _write_event(
        wasp_dir / "Grazing" / "99_Grazing_cow-extra_20240515_133048.csv",
        nonfinite=True,
    )

    with pytest.raises(DatasetValidationError, match="non-finite"):
        load_wasp_dataset(wasp_dir)


def test_windowing_never_crosses_a_timestamp_gap() -> None:
    started = datetime(2024, 1, 1)
    first_segment = [
        ImuSample(started + timedelta(milliseconds=100 * index), 1, 2, 3, 4, 5, 6)
        for index in range(WINDOW_SAMPLES)
    ]
    second_start = started + timedelta(seconds=10)
    second_segment = [
        ImuSample(second_start + timedelta(milliseconds=100 * index), 1, 2, 3, 4, 5, 6)
        for index in range(WINDOW_SAMPLES)
    ]
    event = WaspEvent(
        event_id="gap-event",
        cow_id="cow-a",
        behaviour_code=3,
        behaviour_name="Walking",
        source_path=Path("gap.csv"),
        samples=tuple(first_segment + second_segment),
    )

    windows = segment_event(event)

    assert len(windows) == 2
    assert windows[0].start_time == started
    assert windows[1].start_time == second_start


def test_features_are_exactly_112_and_deterministic() -> None:
    time_axis = np.arange(WINDOW_SAMPLES) / 10.0
    samples = np.column_stack(
        (
            np.sin(2 * np.pi * 2 * time_axis),
            np.zeros(WINDOW_SAMPLES),
            np.ones(WINDOW_SAMPLES),
            np.cos(2 * np.pi * 2 * time_axis),
            np.zeros(WINDOW_SAMPLES),
            np.ones(WINDOW_SAMPLES),
        )
    )

    first = extract_window_features(samples)
    second = extract_window_features(samples)

    assert first.shape == (112,)
    assert len(FEATURE_NAMES) == 112
    assert np.array_equal(first, second)
    assert first[FEATURE_NAMES.index("accel_x__dominant_frequency_hz")] == pytest.approx(2.0)


def test_group_integrity_rejects_cow_or_event_leakage() -> None:
    cows = np.asarray(["cow-a", "cow-a", "cow-b", "cow-b"])
    events = np.asarray(["event-1", "event-1", "event-2", "event-2"])

    _assert_group_integrity(cows, events, np.asarray([0, 1]), np.asarray([2, 3]))
    with pytest.raises(RuntimeError, match="Cow-grouped"):
        _assert_group_integrity(cows, events, np.asarray([0, 1]), np.asarray([1, 2]))


def test_fixed_seed_random_forest_training_is_reproducible() -> None:
    imports = _ml_imports()
    rng = np.random.default_rng(42)
    features = rng.normal(size=(32, 112))
    labels = np.asarray([0, 1, 3, 5] * 8, dtype=np.int64)
    groups = np.asarray([f"cow-{cow}" for cow in range(4) for _ in range(8)])

    first, first_params = _fit_classical_model(
        model_name="random_forest",
        features=features,
        labels=labels,
        groups=groups,
        imports=imports,
        seed=42,
        inner_splits=3,
    )
    second, second_params = _fit_classical_model(
        model_name="random_forest",
        features=features,
        labels=labels,
        groups=groups,
        imports=imports,
        seed=42,
        inner_splits=3,
    )

    assert first_params == second_params
    assert np.array_equal(first.predict(features), second.predict(features))


def test_per_class_aggregate_retains_all_required_metrics() -> None:
    matrix = np.asarray(
        [
            [8, 2, 0, 0],
            [1, 9, 0, 0],
            [0, 0, 7, 3],
            [0, 0, 2, 8],
        ]
    )

    metrics = _aggregate_per_class_metrics(matrix)

    assert set(metrics) == {"Resting", "Grazing", "Walking", "Other/Unknown"}
    assert metrics["Resting"] == {
        "code": 0,
        "precision": pytest.approx(8 / 9),
        "recall": pytest.approx(0.8),
        "f1": pytest.approx(16 / 19),
        "support": 10,
    }
    assert metrics["Grazing"]["support"] == 10
    assert metrics["Walking"]["recall"] == pytest.approx(0.7)
    assert metrics["Other/Unknown"]["precision"] == pytest.approx(8 / 11)


def test_run_config_records_all_result_affecting_settings() -> None:
    config = BenchmarkConfig(
        seed=7,
        inner_splits=3,
        cnn_max_epochs=12,
        cnn_patience_epochs=4,
        source_revision="a" * 40,
    )

    run_config = _build_run_config(config, ["random_forest", "cnn_1d"])

    assert run_config["benchmark"] == {
        "seed": 7,
        "data_source_ref": "WASP-lab/db-cow-walking (private Kaggle input)",
        "source_revision": "a" * 40,
        "include_cnn": True,
        "inner_splits": 3,
        "cnn_max_epochs": 12,
        "cnn_patience_epochs": 4,
    }
    assert run_config["model_search"]["random_forest"]["search_grid"]["max_depth"] == [None, 12]
    assert run_config["model_search"]["cnn_1d"]["early_stopping"]["max_epochs"] == 12


def test_final_cnn_refit_normalises_against_all_valid_windows() -> None:
    pytest.importorskip("torch", reason="Final CNN refit requires the optional torch dependency.")
    raw_windows = np.zeros((8, WINDOW_SAMPLES, 6), dtype=np.float32)
    raw_windows[4:] = 10.0
    labels = np.asarray([0, 1, 3, 5, 0, 1, 3, 5], dtype=np.int64)

    bundle = _fit_cnn_final_model(raw_windows=raw_windows, labels=labels, seed=42, epochs=1)

    assert np.array_equal(bundle.mean, raw_windows.mean(axis=(0, 1), keepdims=True))
