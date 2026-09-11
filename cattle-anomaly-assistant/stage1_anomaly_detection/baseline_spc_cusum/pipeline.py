"""Orchestration and atomic derived-artifact publication for M1b/M1c."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import DetectorConfig
from .daily import aggregate_daily_features
from .detector import DetectionResult, detect, learn_baselines
from .errors import fail
from .injection import InjectionReport, apply_injections
from .mmcows import MmCowsInputs, load_mmcows_inputs
from .models import DailyFeature


@dataclass(frozen=True)
class RunResult:
    inputs: MmCowsInputs
    features: list[DailyFeature]
    detection: DetectionResult
    injection: InjectionReport | None

    def summary(self) -> dict[str, object]:
        usable_baselines = sum(1 for baseline in self.detection.baselines if baseline.available)
        anomalies = sum(1 for window in self.detection.windows if window["anomaly_flag"])
        return {
            "schema_version": 1,
            "available_streams": self.inputs.available_streams,
            "cow_count": len({feature.cow_id for feature in self.features}),
            "daily_feature_windows": len(self.features),
            "usable_baselines": usable_baselines,
            "anomaly_windows": anomalies,
            "injection_scenarios": self.injection.scenario_ids if self.injection else [],
        }


def run_detector(
    data_root: Path,
    config: DetectorConfig,
    *,
    require_immu: bool = False,
    injection_path: Path | None = None,
) -> RunResult:
    """Validate, aggregate, optionally inject, and detect without writing raw data."""

    inputs = load_mmcows_inputs(data_root, config, require_immu=require_immu)
    features = aggregate_daily_features(inputs, config)
    baselines = learn_baselines(features, config)
    processed_features, injection = apply_injections(features, baselines, injection_path)
    detection = detect(processed_features, baselines, config)
    return RunResult(inputs=inputs, features=processed_features, detection=detection, injection=injection)


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False))
            handle.write("\n")


def _assert_safe_output_location(data_root: Path, output_dir: Path) -> tuple[Path, Path]:
    resolved_input = data_root.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()
    if resolved_output == resolved_input or resolved_input in resolved_output.parents or resolved_output in resolved_input.parents:
        fail(
            "UNSAFE_OUTPUT_DIRECTORY",
            "--output-dir must be a separate sibling location, never the input directory or one of its parents/children.",
            data_root=str(resolved_input),
            output_dir=str(resolved_output),
        )
    return resolved_input, resolved_output


def publish_output(
    result: RunResult,
    config: DetectorConfig,
    *,
    data_root: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    """Publish only derived artifacts through a sibling temporary directory."""

    _, final_dir = _assert_safe_output_location(data_root, output_dir)
    if final_dir.exists() and not overwrite:
        fail("OUTPUT_EXISTS", "The output directory already exists; choose a new path or pass --overwrite.", output_dir=str(final_dir))
    if final_dir.exists() and not final_dir.is_dir():
        fail("OUTPUT_NOT_DIRECTORY", "The output path exists but is not a directory.", output_dir=str(final_dir))
    try:
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=final_dir.parent))
    except OSError as exc:
        fail("OUTPUT_WRITE_ERROR", "Could not create the separate output directory.", output_dir=str(final_dir), error=str(exc))
    try:
        _write_jsonl(temporary_dir / "daily_features.jsonl", (feature.as_dict() for feature in result.features))
        _write_jsonl(temporary_dir / "baselines.jsonl", (baseline.as_dict() for baseline in result.detection.baselines))
        _write_jsonl(temporary_dir / "cusum_traces.jsonl", (trace.as_dict() for trace in result.detection.traces))
        _write_jsonl(temporary_dir / "anomaly_windows.jsonl", result.detection.windows)
        manifest = {
            "schema_version": 1,
            "artifact": "mmcows_daily_personal_baseline_spc_cusum",
            "summary": result.summary(),
            "config": config.as_dict(),
            "input": {
                "data_root": str(result.inputs.root),
                "available_streams": result.inputs.available_streams,
                "raw_sensor_rows_persisted": False,
            },
            "reference_baseline_verified_healthy": False,
            "interpretation": "Derived anomaly indicators only. Not a clinical or veterinary diagnosis.",
        }
        (temporary_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if final_dir.exists():
            # --overwrite is explicit user authorization.  The preflight above
            # constrains this deletion to the exact output directory.
            shutil.rmtree(final_dir)
        os.replace(temporary_dir, final_dir)
    except OSError as exc:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        fail("OUTPUT_WRITE_ERROR", "Could not publish derived output artifacts.", output_dir=str(final_dir), error=str(exc))
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    return final_dir
