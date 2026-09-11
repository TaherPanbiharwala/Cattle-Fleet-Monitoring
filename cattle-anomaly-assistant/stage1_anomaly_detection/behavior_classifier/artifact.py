"""Native XGBoost JSON artifact creation and verification; never pickle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import fail
from .features import FEATURE_NAMES, SAMPLE_RATE_HZ, WINDOW_SAMPLES, WINDOW_STRIDE_SAMPLES
from .io import sha256_file, write_json
from .wasp import CLASS_NAMES, PLATFORM_CODES

ARTIFACT_SCHEMA_VERSION = 1
MAX_MODEL_BYTES = 64 * 1024 * 1024


def _xgb() -> Any:
    try:
        import xgboost
    except ImportError as exc:  # pragma: no cover - optional-install boundary
        fail("DEPENDENCY_MISSING", "XGBoost is required to verify or use behavior artifacts.", install="python -m pip install -e '.[behavior,dev]'")
        raise AssertionError from exc
    return xgboost


def feature_hash() -> str:
    return hashlib.sha256("\n".join(FEATURE_NAMES).encode("utf-8")).hexdigest()


def legacy_pickle_provenance(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    target = path.expanduser().resolve()
    if not target.is_file():
        fail("LEGACY_PICKLE_REJECTED", "Legacy pickle reference does not exist and cannot be inspected.", path=str(target))
    if target.suffix.casefold() not in {".pkl", ".pickle"}:
        fail("LEGACY_PICKLE_REJECTED", "Only a legacy pickle reference may be recorded here; it will not be loaded.", path=str(target))
    return {"path_name": target.name, "sha256": sha256_file(target), "loaded": False, "status": "unsupported_python_pickle"}


def save_model_artifact(
    model: Any,
    output_dir: Path,
    *,
    dataset_provenance: dict[str, object],
    selected_params: dict[str, object],
    seed: int,
    legacy_pickle: Path | None = None,
) -> dict[str, object]:
    """Persist a native JSON booster and a complete derived-only manifest."""

    xgboost = _xgb()
    model_path = output_dir / "behavior_model.json"
    model.save_model(model_path)
    model_hash = sha256_file(model_path)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact": "wasp_six_axis_xgboost_behavior_benchmark",
        "benchmark_only": True,
        "field_deployment_validated": False,
        "confidence": "uncalibrated_max_class_probability",
        "model_file": model_path.name,
        "model_sha256": model_hash,
        "model_size_bytes": model_path.stat().st_size,
        "xgboost_version": xgboost.__version__,
        "objective": "multi:softprob",
        "num_class": 4,
        "internal_class_order": list(CLASS_NAMES),
        "platform_behavior_codes": PLATFORM_CODES,
        "feature_count": len(FEATURE_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "feature_sha256": feature_hash(),
        "window": {"sample_rate_hz": SAMPLE_RATE_HZ, "window_samples": WINDOW_SAMPLES, "stride_samples": WINDOW_STRIDE_SAMPLES},
        "selected_params": selected_params,
        "seed": seed,
        "dataset_provenance": dataset_provenance,
        "legacy_pickle_provenance": legacy_pickle_provenance(legacy_pickle),
        "raw_sensor_rows_persisted": False,
        "interpretation": "Public WASP benchmark artifact only; it is not validated for the project collar, farm, diagnosis, or treatment.",
    }
    write_json(output_dir / "behavior_model.manifest.json", manifest)
    return manifest


def verify_artifact(model_path: Path, manifest_path: Path | None = None) -> dict[str, object]:
    """Fail safely when a model has changed or violates the prediction contract."""

    model_path = model_path.expanduser().resolve()
    if model_path.suffix.casefold() in {".pkl", ".pickle"}:
        fail("LEGACY_PICKLE_REJECTED", "Python pickle artifacts are unsupported and are never deserialized. Train or verify a native JSON artifact.", path=str(model_path))
    if model_path.suffix.casefold() != ".json":
        fail("ARTIFACT_CONTRACT_MISMATCH", "Behavior model must use the native .json XGBoost format.", path=str(model_path))
    if not model_path.is_file():
        fail("ARTIFACT_INTEGRITY", "Behavior model JSON does not exist.", path=str(model_path))
    if model_path.stat().st_size > MAX_MODEL_BYTES:
        fail("ARTIFACT_INTEGRITY", "Behavior model exceeds the safe size limit.", size_bytes=model_path.stat().st_size, limit_bytes=MAX_MODEL_BYTES)
    manifest_path = (manifest_path or model_path.with_name("behavior_model.manifest.json")).expanduser().resolve()
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("ARTIFACT_INTEGRITY", "Artifact manifest is missing or invalid JSON.", path=str(manifest_path))
        raise AssertionError from exc
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        fail("ARTIFACT_CONTRACT_MISMATCH", "Artifact schema is unsupported; retrain the model instead of migrating it automatically.", found=manifest.get("schema_version"), supported=ARTIFACT_SCHEMA_VERSION)
    if manifest.get("model_file") != model_path.name or manifest.get("model_sha256") != sha256_file(model_path):
        fail("ARTIFACT_HASH_MISMATCH", "Model hash does not match its manifest.", model_path=str(model_path))
    if manifest.get("feature_count") != len(FEATURE_NAMES) or manifest.get("feature_sha256") != feature_hash() or manifest.get("feature_names") != list(FEATURE_NAMES):
        fail("ARTIFACT_CONTRACT_MISMATCH", "Model feature contract differs from this runtime; retrain with the current feature version.")
    if manifest.get("internal_class_order") != list(CLASS_NAMES) or manifest.get("platform_behavior_codes") != PLATFORM_CODES:
        fail("ARTIFACT_CONTRACT_MISMATCH", "Model class order or safe platform behavior mapping is invalid.")
    if manifest.get("objective") != "multi:softprob" or manifest.get("num_class") != 4:
        fail("ARTIFACT_CONTRACT_MISMATCH", "Model must be a four-class multi:softprob classifier.")
    xgboost = _xgb()
    try:
        booster = xgboost.Booster()
        booster.load_model(model_path)
        import numpy as np

        probabilities = booster.predict(xgboost.DMatrix(np.zeros((1, len(FEATURE_NAMES)), dtype=np.float64)))
    except Exception as exc:
        fail("ARTIFACT_INTEGRITY", "Native XGBoost model could not be loaded and probed safely.", error=str(exc))
    if probabilities.shape != (1, 4) or not np.isfinite(probabilities).all() or abs(float(probabilities[0].sum()) - 1.0) > 1e-5:
        fail("ARTIFACT_CONTRACT_MISMATCH", "Model probability output is not a finite four-class probability distribution.")
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "verified",
        "model_path": str(model_path),
        "manifest_path": str(manifest_path),
        "model_sha256": manifest["model_sha256"],
        "feature_sha256": manifest["feature_sha256"],
        "benchmark_only": True,
        "confidence": manifest["confidence"],
        "raw_sensor_rows_persisted": False,
    }


def load_verified_booster(model_path: Path, manifest_path: Path | None = None) -> tuple[Any, dict[str, object]]:
    verify_artifact(model_path, manifest_path)
    manifest_file = (manifest_path or model_path.with_name("behavior_model.manifest.json")).expanduser().resolve()
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    xgboost = _xgb()
    booster = xgboost.Booster()
    booster.load_model(model_path.expanduser().resolve())
    return booster, manifest
