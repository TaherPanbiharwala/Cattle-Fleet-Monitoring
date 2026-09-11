"""Historical public-dataset demo records using the active 25-channel XGBoost model.

This is intentionally separate from ``fuse_daily_records``.  It never claims
that WASP and MmCows rows belong to the same cow, date, or deployment.  Its
sole purpose is to let the application display real historical behavior-model
output beside independently-derived historical CUSUM indicators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from shared.schemas import AnomalyRecord, HistoricalBehaviorContext

from .context import CusumDailyResult, _model_rows, adapt_cusum_window
from .errors import fail
from .io import sha256_file

MODEL_SCHEMA_VERSION = 1
WINDOW_SAMPLES = 50
WINDOW_STRIDE_SAMPLES = 25
SAMPLE_RATE_HZ = 10
EXPECTED_SAMPLE_PERIOD_S = 1 / SAMPLE_RATE_HZ
SAMPLE_PERIOD_TOLERANCE_S = 0.02
PREDICTION_BATCH_WINDOWS = 512
SENSOR_COLUMNS = (
    "BNO055_ARX", "BNO055_ARY", "BNO055_ARZ",
    "BNO055_AX", "BNO055_AY", "BNO055_AZ",
    "BNO055_GX", "BNO055_GY", "BNO055_GZ",
    "BNO055_MX", "BNO055_MY", "BNO055_MZ",
    "BNO055_Q0", "BNO055_Q1", "BNO055_Q2", "BNO055_Q3",
    "MPU9250_AX", "MPU9250_AY", "MPU9250_AZ",
    "MPU9250_GX", "MPU9250_GY", "MPU9250_GZ",
    "MPU9250_MX", "MPU9250_MY", "MPU9250_MZ",
)
STATISTIC_NAMES = ("mean", "std", "min", "max", "median", "range", "rms", "mean_abs", "energy")
FEATURE_NAMES = tuple(f"{sensor}__{statistic}" for sensor in SENSOR_COLUMNS for statistic in STATISTIC_NAMES)
CLASS_LABELS = {
    "Walking": 0,
    "Grazing": 1,
    "Resting": 2,
    "Miscellaneous behaviors": 3,
}
INDEX_TO_STATE = {
    0: "walking",
    1: "grazing",
    2: "resting",
    3: "miscellaneous",
}
WASP_CLASS_DIRECTORIES = tuple(CLASS_LABELS)


def _numpy_xgboost() -> tuple[Any, Any]:
    try:
        import numpy
        import xgboost
    except ImportError as exc:  # pragma: no cover - package extra boundary
        fail("DEPENDENCY_MISSING", "Historical behavior-model inference requires the behavior extra.", install="python -m pip install -e '.[behavior,dev]'")
        raise AssertionError from exc
    return numpy, xgboost


def _feature_hash() -> str:
    return hashlib.sha256(json.dumps(list(FEATURE_NAMES), separators=(",", ":")).encode("utf-8")).hexdigest()


def _read_manifest(model_path: Path, manifest_path: Path) -> dict[str, Any]:
    target = model_path.expanduser().resolve()
    manifest_target = manifest_path.expanduser().resolve()
    if target.suffix.casefold() in {".pkl", ".pickle"}:
        fail("LEGACY_PICKLE_REJECTED", "Python pickle models are never loaded. Supply the Kaggle native JSON model.", path=str(target))
    if target.suffix.casefold() != ".json" or not target.is_file():
        fail("HISTORICAL_MODEL_INVALID", "--model-path must be an existing native XGBoost JSON file.", path=str(target))
    try:
        manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("HISTORICAL_MODEL_INVALID", "The historical model manifest is missing or invalid JSON.", path=str(manifest_target), error=str(exc))
        raise AssertionError from exc
    if not isinstance(manifest, dict):
        fail("HISTORICAL_MODEL_INVALID", "The historical model manifest must be a JSON object.", path=str(manifest_target))
    expected_window = {"sample_rate_hz": SAMPLE_RATE_HZ, "window_samples": WINDOW_SAMPLES, "stride_samples": WINDOW_STRIDE_SAMPLES}
    required = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_format": "native_xgboost_json",
        "model_file": target.name,
        "model_sha256": sha256_file(target),
        "feature_names": list(FEATURE_NAMES),
        "feature_sha256": _feature_hash(),
        "sensor_columns": list(SENSOR_COLUMNS),
        "class_labels": CLASS_LABELS,
        "window_settings": expected_window,
    }
    mismatches = [key for key, value in required.items() if manifest.get(key) != value]
    if mismatches:
        fail("HISTORICAL_MODEL_INVALID", "The Kaggle model or manifest does not match the required 25-channel XGBoost contract.", mismatches=mismatches)
    return manifest


def _load_booster(model_path: Path, manifest_path: Path) -> tuple[Any, dict[str, Any]]:
    numpy, xgboost = _numpy_xgboost()
    manifest = _read_manifest(model_path, manifest_path)
    try:
        booster = xgboost.Booster()
        booster.load_model(model_path.expanduser().resolve())
        probabilities = booster.predict(xgboost.DMatrix(numpy.zeros((1, len(FEATURE_NAMES)), dtype=numpy.float32), feature_names=list(FEATURE_NAMES)))
    except Exception as exc:
        fail("HISTORICAL_MODEL_INVALID", "The native XGBoost model could not be loaded and probed.", error=str(exc))
        raise AssertionError from exc
    if booster.num_features() != len(FEATURE_NAMES) or probabilities.shape != (1, 4) or not numpy.isfinite(probabilities).all() or not numpy.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        fail("HISTORICAL_MODEL_INVALID", "The XGBoost model does not emit a finite four-class probability distribution.")
    return booster, manifest


def _event_paths(dataset_dir: Path) -> list[Path]:
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        fail("WASP_LAYOUT_INVALID", "--wasp-dataset-dir does not exist or is not a directory.", dataset_dir=str(root))
    paths: list[Path] = []
    for directory in WASP_CLASS_DIRECTORIES:
        folder = root / directory
        if not folder.is_dir():
            fail("WASP_LAYOUT_INVALID", "Historical WASP dataset is missing a required class folder.", folder=directory, dataset_dir=str(root))
        paths.extend(sorted(folder.glob("*.csv")))
    if not paths:
        fail("WASP_LAYOUT_INVALID", "Historical WASP dataset has no CSV recordings.", dataset_dir=str(root))
    return paths


def _parse_timestamp(value: str | None) -> datetime:
    """Parse a WASP recording-clock timestamp.

    The public files use ISO date-times without an offset (for example
    ``2024-05-15 13:04:17.0``). They are used only for within-file cadence
    validation, never for an MmCows join or output timestamp, so a consistent
    naive recording clock is valid. A file may not mix naive and aware values.
    """

    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed


def _windows_from_segment(segment: list[list[float]]) -> list[list[list[float]]]:
    return [segment[index : index + WINDOW_SAMPLES] for index in range(0, len(segment) - WINDOW_SAMPLES + 1, WINDOW_STRIDE_SAMPLES)]


def _event_windows(path: Path) -> tuple[list[list[list[float]]], dict[str, int]]:
    """Return only complete cadence-valid windows plus derived exclusion counts.

    A malformed/non-monotonic/gapped timestamp ends the current segment.  Its
    source values never reach an output artifact and a later valid segment can
    still be used, which is safer than silently bridging an invalid cadence.
    """

    try:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = [name for name in ("Time", *SENSOR_COLUMNS) if name not in (reader.fieldnames or [])]
            if missing:
                fail("WASP_LAYOUT_INVALID", "Historical WASP CSV is missing required columns.", path=str(path), missing=missing)
            windows: list[list[list[float]]] = []
            segment: list[list[float]] = []
            previous_timestamp: datetime | None = None
            timestamp_awareness: bool | None = None
            source_rows = 0
            valid_rows = 0
            timestamp_breaks = 0
            cadence_breaks = 0
            short_segment_rows = 0

            def flush_segment() -> None:
                nonlocal short_segment_rows
                if segment:
                    windows.extend(_windows_from_segment(segment))
                    if len(segment) < WINDOW_SAMPLES:
                        short_segment_rows += len(segment)
                    segment.clear()

            for row_number, row in enumerate(reader, start=2):
                source_rows += 1
                try:
                    values = [float(row[name]) for name in SENSOR_COLUMNS]
                except (TypeError, ValueError):
                    fail("WASP_LAYOUT_INVALID", "Historical WASP sensor value is not numeric.", path=str(path), row=row_number)
                if not all(math.isfinite(value) for value in values):
                    fail("WASP_LAYOUT_INVALID", "Historical WASP sensor value is not finite.", path=str(path), row=row_number)
                try:
                    timestamp = _parse_timestamp(row.get("Time"))
                except ValueError:
                    flush_segment()
                    previous_timestamp = None
                    timestamp_awareness = None
                    timestamp_breaks += 1
                    continue
                is_aware = timestamp.tzinfo is not None and timestamp.utcoffset() is not None
                if previous_timestamp is not None:
                    if timestamp_awareness != is_aware:
                        flush_segment()
                        timestamp_breaks += 1
                    else:
                        interval_s = (timestamp - previous_timestamp).total_seconds()
                        if interval_s <= 0:
                            flush_segment()
                            timestamp_breaks += 1
                        elif abs(interval_s - EXPECTED_SAMPLE_PERIOD_S) > SAMPLE_PERIOD_TOLERANCE_S:
                            flush_segment()
                            cadence_breaks += 1
                segment.append(values)
                previous_timestamp = timestamp
                timestamp_awareness = is_aware
                valid_rows += 1
            flush_segment()
    except OSError as exc:
        fail("WASP_LAYOUT_INVALID", "Historical WASP CSV could not be read.", path=str(path), error=str(exc))
    return windows, {
        "source_rows": source_rows,
        "valid_rows": valid_rows,
        "timestamp_breaks": timestamp_breaks,
        "cadence_breaks": cadence_breaks,
        "short_segment_rows": short_segment_rows,
    }


def _features(windows: list[list[list[float]]]) -> Any:
    numpy, _ = _numpy_xgboost()
    matrix = numpy.asarray(windows, dtype=numpy.float32)
    if matrix.ndim != 3 or matrix.shape[1:] != (WINDOW_SAMPLES, len(SENSOR_COLUMNS)):
        fail("HISTORICAL_MODEL_INVALID", "Historical feature extraction received an invalid window shape.", shape=list(matrix.shape))
    features = numpy.stack((
        matrix.mean(axis=1), matrix.std(axis=1), matrix.min(axis=1), matrix.max(axis=1), numpy.median(matrix, axis=1),
        matrix.max(axis=1) - matrix.min(axis=1), numpy.sqrt((matrix ** 2).mean(axis=1)), numpy.abs(matrix).mean(axis=1), (matrix ** 2).mean(axis=1),
    ), axis=2).reshape(len(matrix), len(FEATURE_NAMES)).astype(numpy.float32)
    if not numpy.isfinite(features).all():
        fail("HISTORICAL_MODEL_INVALID", "Historical feature extraction produced a non-finite value.")
    return features


def _source_sha256(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def historical_behavior_summary(*, wasp_dataset_dir: Path, model_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Classify the staged public WASP data with the user's active Kaggle model."""

    numpy, xgboost = _numpy_xgboost()
    booster, manifest = _load_booster(model_path, manifest_path)
    paths = _event_paths(wasp_dataset_dir)
    root = wasp_dataset_dir.expanduser().resolve()
    counts: Counter[str] = Counter()
    confidence_sums: Counter[str] = Counter()
    window_count = 0
    exclusion_counts: Counter[str] = Counter()
    for path in paths:
        windows, diagnostics = _event_windows(path)
        exclusion_counts.update(diagnostics)
        for start in range(0, len(windows), PREDICTION_BATCH_WINDOWS):
            batch = windows[start : start + PREDICTION_BATCH_WINDOWS]
            probabilities = booster.predict(xgboost.DMatrix(_features(batch), feature_names=list(FEATURE_NAMES)))
            if probabilities.shape != (len(batch), 4) or not numpy.isfinite(probabilities).all() or not numpy.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
                fail("HISTORICAL_MODEL_INVALID", "Historical behavior prediction output has an invalid probability shape or sum.")
            indexes = probabilities.argmax(axis=1)
            for row, index in enumerate(indexes):
                state = INDEX_TO_STATE[int(index)]
                counts[state] += 1
                confidence_sums[state] += float(probabilities[row, index])
            window_count += len(batch)
    if not window_count:
        fail("WASP_LAYOUT_INVALID", "Historical WASP recordings contain no complete 50-sample windows.")
    highest = max(counts.values())
    state = min(name for name, count in counts.items() if count == highest)
    return {
        "schema_version": 1,
        "source_kind": "wasp_public_historical_dataset",
        "context_relation": "cross_dataset_historical_demo",
        "model_file": model_path.expanduser().resolve().name,
        "model_sha256": manifest["model_sha256"],
        "manifest_sha256": sha256_file(manifest_path.expanduser().resolve()),
        "behavior_state": state,
        "behavior_state_confidence": confidence_sums[state] / counts[state],
        "behavior_state_distribution": {name: counts[name] / window_count for name in ("walking", "grazing", "resting", "miscellaneous")},
        "observed_windows": window_count,
        "source_csv_files": len(paths),
        "source_sha256": _source_sha256(paths, root),
        "window_exclusions": {
            "source_rows": exclusion_counts["source_rows"],
            "valid_rows": exclusion_counts["valid_rows"],
            "timestamp_breaks": exclusion_counts["timestamp_breaks"],
            "cadence_breaks": exclusion_counts["cadence_breaks"],
            "short_segment_rows": exclusion_counts["short_segment_rows"],
        },
        "prediction_batch_windows": PREDICTION_BATCH_WINDOWS,
        "confidence_kind": "uncalibrated_max_class_probability",
        "raw_sensor_rows_persisted": False,
        "interpretation": "Historical WASP behavior summary from the active XGBoost model. It is displayed beside, not joined to, independent MmCows anomaly indicators.",
    }


def historical_demo_anomaly_records(
    cusum_window_records: list[dict[str, Any]],
    *,
    deployment_id: str,
    timezone: str,
    behavior_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    """Attach one labelled historical behavior summary to independently-derived CUSUM records."""

    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        fail("RUNTIME_CONTEXT_INVALID", "--timezone must be a valid IANA timezone.", timezone=timezone)
    normalized = [adapt_cusum_window(row, deployment_id=deployment_id, source_kind="mmcows_public", timezone=timezone) for row in cusum_window_records]
    cusums = _model_rows(normalized, CusumDailyResult, error_code="RUNTIME_CONTEXT_INVALID")
    if not cusums:
        fail("NO_JOINABLE_RECORDS", "No CUSUM daily windows were supplied.")
    model_hash = behavior_summary.get("model_sha256")
    manifest_hash = behavior_summary.get("manifest_sha256")
    if not isinstance(model_hash, str) or not isinstance(manifest_hash, str):
        fail("HISTORICAL_MODEL_INVALID", "Historical behavior summary does not meet the AnomalyRecord behavior-context contract.")
    try:
        historical_context = HistoricalBehaviorContext(
            source_kind=behavior_summary.get("source_kind"),
            context_relation=behavior_summary.get("context_relation"),
            model_sha256=model_hash,
            manifest_sha256=manifest_hash,
            behavior_state=behavior_summary.get("behavior_state"),
            behavior_state_confidence=behavior_summary.get("behavior_state_confidence"),
            behavior_state_distribution=behavior_summary.get("behavior_state_distribution"),
            observed_windows=behavior_summary.get("observed_windows"),
            source_csv_files=behavior_summary.get("source_csv_files"),
            confidence_kind=behavior_summary.get("confidence_kind"),
        )
    except Exception as exc:
        fail("HISTORICAL_MODEL_INVALID", "Historical behavior summary does not meet the AnomalyRecord historical-context contract.", error=str(exc))
    history_by_cow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for cusum in sorted(cusums, key=lambda row: (row.cow_id, row.local_date, row.window_id)):
        if not cusum.core_quality_valid:
            fail("RUNTIME_CONTEXT_INVALID", "CUSUM daily context lacks required valid physiology values.", window_id=cusum.window_id)
        if "behavior_state" in cusum.driving_signals:
            fail("RUNTIME_CONTEXT_INVALID", "Behavior remains context-only and cannot be a CUSUM driver.", window_id=cusum.window_id)
        timestamp = datetime.combine(cusum.local_date + timedelta(days=1), time.min, tzinfo=zone).astimezone(ZoneInfo("UTC"))
        record = AnomalyRecord(
            cow_id=cusum.cow_id,
            timestamp=timestamp,
            window_id=cusum.window_id,
            behavior_state=None,
            behavior_state_confidence=None,
            behavior_state_distribution_24h=None,
            behavior_context_source="wasp_public_historical_dataset",
            behavior_context_relation="cross_dataset_historical_demo",
            behavior_model_sha256=model_hash,
            historical_behavior_context=historical_context,
            cbt_c=cusum.cbt_c,
            cbt_deviation_sigma=cusum.cbt_deviation_sigma,
            cbt_cusum_value=cusum.cbt_cusum_value,
            lying_time_pct_24h=cusum.lying_time_pct_24h,
            lying_time_deviation_sigma=cusum.lying_time_deviation_sigma,
            thi=cusum.thi,
            activity_magnitude_deviation_sigma=cusum.activity_magnitude_deviation_sigma,
            herd_isolation_score=None,
            anomaly_flag=cusum.anomaly_flag,
            anomaly_score=cusum.anomaly_score,
            driving_signals=list(cusum.driving_signals),
            recent_anomaly_history=history_by_cow[cusum.cow_id][-3:],
        )
        # Cross-dataset records deliberately omit the ordinary cow-day
        # behaviour field names altogether. A JSON ``null`` still invites a
        # downstream consumer to treat them as a missing measurement, while
        # absence makes the public-data boundary unambiguous.
        dumped = record.model_dump(mode="json", exclude_none=True)
        output.append(dumped)
        history_by_cow[cusum.cow_id].append({"timestamp": dumped["timestamp"], "flag": dumped["anomaly_flag"], "driving_signals": dumped["driving_signals"]})
    return output
