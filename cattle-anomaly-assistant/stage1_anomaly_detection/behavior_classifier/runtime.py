"""Prediction of future runtime six-axis IMU windows without persisting raw data."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, Field, ValidationError

from shared.schemas._base import StrictModel

from .artifact import load_verified_booster
from .context import BehaviorPrediction
from .errors import fail
from .features import extract_features


class RuntimeWindow(StrictModel):
    """Input-only runtime row. ``samples`` is intentionally never emitted."""

    cow_id: str = Field(min_length=1)
    timestamp: AwareDatetime
    deployment_id: str = Field(min_length=1)
    source_kind: str
    sample_rate_hz: float
    acceleration_unit: Literal["m_s2"]
    gyroscope_unit: Literal["deg_s"]
    calibration_id: str = Field(min_length=1)
    sample_timestamps: list[AwareDatetime]
    samples: list[list[float]]


def predict_runtime_windows(records: list[dict[str, Any]], *, model_path: Path, manifest_path: Path | None = None) -> list[dict[str, Any]]:
    booster, manifest = load_verified_booster(model_path, manifest_path)
    if not records:
        fail("RUNTIME_CONTEXT_INVALID", "No runtime IMU windows were supplied for prediction.")
    try:
        import numpy as np
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover - verified artifact normally catches this first
        fail("DEPENDENCY_MISSING", "Prediction requires the behavior extra.")
        raise AssertionError from exc
    output: list[dict[str, Any]] = []
    labels = manifest["internal_class_order"]
    for row_number, row in enumerate(records, start=1):
        try:
            window = RuntimeWindow.model_validate(row)
        except ValidationError as exc:
            # Pydantic's detailed errors can echo the input ``samples`` value.
            # Keep raw IMU rows out of every output, including error JSON.
            fail("RUNTIME_CONTEXT_INVALID", "Runtime IMU input does not match the required window contract.", row=row_number, validation_error_count=len(exc.errors()))
        if window.source_kind != "same_cow_runtime":
            fail("CONTEXT_SOURCE_NOT_RUNTIME", "Prediction accepts only same-cow runtime input, not public benchmark data.", row=row_number, source_kind=window.source_kind)
        if abs(window.sample_rate_hz - 10.0) > 1e-9:
            fail("FEATURE_CONTRACT", "Runtime windows must declare the 10 Hz sampling rate used for the model.", row=row_number, sample_rate_hz=window.sample_rate_hz)
        if len(window.sample_timestamps) != 50 or len(window.samples) != 50:
            fail("FEATURE_CONTRACT", "Runtime input must provide 50 timestamped six-axis samples.", row=row_number)
        if window.timestamp != window.sample_timestamps[0]:
            fail("FEATURE_CONTRACT", "Runtime window timestamp must equal its first sample timestamp.", row=row_number)
        for previous, current in zip(window.sample_timestamps, window.sample_timestamps[1:], strict=False):
            if abs((current - previous).total_seconds() - 0.1) > 0.02:
                fail("FEATURE_CONTRACT", "Runtime IMU samples contain a timing gap or incompatible cadence.", row=row_number)
        features = extract_features(np.asarray(window.samples, dtype=np.float64))
        probabilities = booster.predict(xgb.DMatrix(features.reshape(1, -1)))
        if probabilities.shape != (1, 4) or not np.isfinite(probabilities).all() or abs(float(probabilities[0].sum()) - 1.0) > 1e-5:
            fail("ARTIFACT_CONTRACT_MISMATCH", "Verified model did not return a valid four-state probability distribution.")
        index = int(np.argmax(probabilities[0]))
        prediction = BehaviorPrediction(
            cow_id=window.cow_id,
            timestamp=window.timestamp,
            deployment_id=window.deployment_id,
            source_kind="same_cow_runtime",
            model_sha256=str(manifest["model_sha256"]),
            behavior_state=labels[index],
            behavior_state_confidence=float(probabilities[0, index]),
        )
        output.append(prediction.model_dump(mode="json"))
    return output
