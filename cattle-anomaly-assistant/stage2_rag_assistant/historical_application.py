"""Public-data historical AnomalyRecord-to-RAG application command.

This is the project’s intentional public-data bridge.  It combines a
dataset-level WASP model summary with independently-derived MmCows CUSUM
records, but never treats that summary as a measurement of an MmCows cow/day.
The CUSUM result remains the sole anomaly signal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import urllib.error
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from shared.schemas import AnomalyExplanationResponse
from stage1_anomaly_detection.behavior_classifier.errors import BehaviorError
from stage1_anomaly_detection.behavior_classifier.historical_demo import (
    historical_behavior_summary,
    historical_demo_anomaly_records,
)
from stage1_anomaly_detection.behavior_classifier.io import atomic_output_dir, read_jsonl, sha256_file, write_json, write_jsonl
from stage2_rag_assistant.config.settings import Stage2Config, load_config
from stage2_rag_assistant.llm.client import LLMClient, build_llm_client
from stage2_rag_assistant.llm.providers.openrouter_provider import OpenRouterError
from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

SCHEMA_VERSION = 1
DEFAULT_DEPLOYMENT_ID = "mmcows-public-2023"
DEFAULT_TIMEZONE = "America/Chicago"
EXPECTED_MODEL_SHA256 = "79bfdde322a1602a2df10293a254c6c3242ad50cc5fbf6566303cb0bcdc29028"
EXPECTED_MANIFEST_SHA256 = "29b8874ec94daaf5f7f50b23661ffe81382355baa38379323be6e54ef9aad92b"
EXPECTED_METRICS_SHA256 = "c2472c3c5b943238b0fb521670839ed8a7ff21f11c7e545919d6eb5039aae42a"
RECORD_COUNT = 24
FLAGGED_RECORD_COUNT = 11
CONTROL_RECORD_COUNT = 13

_ROOT = Path(__file__).resolve().parents[1]
_REFERENCE_DIR = _ROOT / "stage1_anomaly_detection" / "behavior_classifier" / "reference_model"
PINNED_MODEL_PATH = _REFERENCE_DIR / "cow_behavior_xgboost_reference.json"
PINNED_MANIFEST_PATH = _REFERENCE_DIR / "cow_behavior_xgboost_reference.manifest.json"
PINNED_METRICS_PATH = _REFERENCE_DIR / "cow_behavior_xgboost_reference.metrics.json"
DEFAULT_PIPELINE_CONFIG = Path(__file__).parent / "config" / "openrouter.yaml"


@dataclass(frozen=True)
class HistoricalApplicationError(Exception):
    code: str
    message: str
    details: dict[str, Any]

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


def _fail(code: str, message: str, **details: Any) -> None:
    raise HistoricalApplicationError(code=code, message=message, details=details)


def _stable_hash(value: Any) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _repository_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_ROOT.parent, check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _validate_pinned_model() -> dict[str, str]:
    expected_names = {
        "model": PINNED_MODEL_PATH.name,
        "manifest": PINNED_MANIFEST_PATH.name,
        "metrics": PINNED_METRICS_PATH.name,
    }
    if not all(path.is_file() for path in (PINNED_MODEL_PATH, PINNED_MANIFEST_PATH, PINNED_METRICS_PATH)):
        _fail("PINNED_MODEL_MISMATCH", "The checked-in approved historical XGBoost reference files are missing.", files=expected_names)
    model_hash = sha256_file(PINNED_MODEL_PATH)
    if model_hash != EXPECTED_MODEL_SHA256:
        _fail(
            "PINNED_MODEL_MISMATCH",
            "The checked-in historical XGBoost model does not match the approved friend-model hash.",
            expected_sha256=EXPECTED_MODEL_SHA256,
            actual_sha256=model_hash,
        )
    manifest_hash = sha256_file(PINNED_MANIFEST_PATH)
    metrics_hash = sha256_file(PINNED_METRICS_PATH)
    if manifest_hash != EXPECTED_MANIFEST_SHA256 or metrics_hash != EXPECTED_METRICS_SHA256:
        _fail(
            "PINNED_MODEL_MISMATCH",
            "The checked-in historical model manifest or metrics file does not match the approved friend-model provenance.",
            expected_manifest_sha256=EXPECTED_MANIFEST_SHA256,
            actual_manifest_sha256=manifest_hash,
            expected_metrics_sha256=EXPECTED_METRICS_SHA256,
            actual_metrics_sha256=metrics_hash,
        )
    try:
        manifest = json.loads(PINNED_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("PINNED_MODEL_MISMATCH", "The checked-in historical model manifest is unreadable.", error=str(exc))
    if (
        not isinstance(manifest, dict)
        or manifest.get("model_file") != PINNED_MODEL_PATH.name
        or manifest.get("model_sha256") != EXPECTED_MODEL_SHA256
        or manifest.get("model_format") != "native_xgboost_json"
    ):
        _fail("PINNED_MODEL_MISMATCH", "The checked-in historical model manifest does not match the approved friend model.")
    return {
        "model_sha256": model_hash,
        "manifest_sha256": manifest_hash,
        "metrics_sha256": metrics_hash,
    }


def _load_cusum_input(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    resolved = path.expanduser().resolve()
    rows = read_jsonl(resolved)
    manifest_path = resolved.parent / "manifest.json"
    if not manifest_path.is_file():
        _fail("CUSUM_PROVENANCE_INVALID", "CUSUM input must be accompanied by the detector manifest.json.", manifest_path=str(manifest_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _fail("CUSUM_PROVENANCE_INVALID", "CUSUM detector manifest is invalid JSON.", manifest_path=str(manifest_path), error=str(exc))
    config = manifest.get("config") if isinstance(manifest, dict) else None
    required = {
        "artifact": "mmcows_daily_personal_baseline_spc_cusum",
        "schema_version": 1,
    }
    if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required.items()):
        _fail("CUSUM_PROVENANCE_INVALID", "CUSUM input is not a supported Stage 1 detector artifact.", manifest_path=str(manifest_path))
    expected_config = {"timezone": DEFAULT_TIMEZONE, "baseline_days": 7, "coverage_minimum": 0.75, "cusum_k": 0.5, "cusum_h": 5.0}
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in expected_config.items()):
        _fail("CUSUM_PROVENANCE_INVALID", "CUSUM manifest does not prove the required personal-baseline detector configuration.")
    return rows, manifest, sha256_file(resolved)


def _is_valid_monitoring_row(row: dict[str, Any]) -> bool:
    return (
        row.get("detector_state") == "monitoring"
        and isinstance(row.get("cow_id"), str)
        and isinstance(row.get("window_id"), str)
        and isinstance(row.get("anomaly_flag"), bool)
        and isinstance(row.get("driving_signals"), list)
        and all(row.get(field) is not None for field in ("cbt_c", "cbt_deviation_sigma", "cbt_cusum_value", "lying_time_pct_24h", "lying_time_deviation_sigma", "thi"))
    )


def select_historical_cusum_windows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 11 flagged rows and 13 cow-distributed monitoring controls."""

    monitoring = sorted((row for row in rows if _is_valid_monitoring_row(row)), key=lambda row: (str(row["cow_id"]), str(row["window_id"])))
    flagged = [row for row in monitoring if row["anomaly_flag"]]
    controls = [row for row in monitoring if not row["anomaly_flag"]]
    if len(flagged) != FLAGGED_RECORD_COUNT:
        _fail("CUSUM_SELECTION_INVALID", "Expected exactly 11 flagged valid monitoring CUSUM windows for the historical application.", found=len(flagged))
    by_cow: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in controls:
        by_cow[str(row["cow_id"])].append(row)
    selected_controls: list[dict[str, Any]] = []
    while len(selected_controls) < CONTROL_RECORD_COUNT and any(by_cow.values()):
        for cow_id in sorted(by_cow):
            if by_cow[cow_id] and len(selected_controls) < CONTROL_RECORD_COUNT:
                selected_controls.append(by_cow[cow_id].popleft())
    if len(selected_controls) != CONTROL_RECORD_COUNT:
        _fail("CUSUM_SELECTION_INVALID", "Fewer than 13 valid non-flag monitoring controls are available.", found=len(selected_controls))
    selected = sorted([*flagged, *selected_controls], key=lambda row: (str(row["cow_id"]), str(row["window_id"])))
    if len(selected) != RECORD_COUNT:  # defensive invariant
        _fail("CUSUM_SELECTION_INVALID", "Historical CUSUM selection did not produce 24 records.", found=len(selected))
    return selected


def _validate_kb(config: Stage2Config) -> str:
    path = Path(config.kb.db_path).expanduser()
    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    if not path.is_file():
        _fail(
            "KB_NOT_READY",
            "The grounded knowledge-base database is missing. Build it before the historical application run.",
            command=".venv/bin/python -m stage2_rag_assistant.kb.build_kb --overwrite",
            db_path=str(path),
        )
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
            tables = {
                str(row[0])
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            expected_tables = {"shift_categories", "shift_thresholds", "literature_links"}
            if not expected_tables.issubset(tables):
                _fail(
                    "KB_NOT_READY",
                    "The grounded knowledge-base database is missing required tables. Rebuild it before the historical application run.",
                    missing_tables=sorted(expected_tables - tables),
                    db_path=str(path),
                )
            category_count = int(connection.execute("SELECT COUNT(*) FROM shift_categories").fetchone()[0])
            if category_count == 0:
                _fail(
                    "KB_NOT_READY",
                    "The grounded knowledge-base database has no seeded categories. Rebuild it before the historical application run.",
                    db_path=str(path),
                )
    except sqlite3.Error as exc:
        _fail("KB_NOT_READY", "The grounded knowledge-base database could not be read safely.", db_path=str(path), error=str(exc))
    return sha256_file(path)


def preflight(
    *,
    wasp_dataset_dir: Path,
    cusum_windows_jsonl: Path,
    deployment_id: str = DEFAULT_DEPLOYMENT_ID,
    timezone_name: str = DEFAULT_TIMEZONE,
    pipeline_config: Stage2Config | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Validate and derive the read-only inputs; never calls an LLM or writes."""

    if timezone_name != DEFAULT_TIMEZONE:
        _fail("RUNTIME_CONTEXT_INVALID", "Historical MmCows CUSUM records must use America/Chicago.", timezone=timezone_name)
    if not deployment_id.strip():
        _fail("RUNTIME_CONTEXT_INVALID", "--deployment-id must be non-empty.")
    config = pipeline_config or load_config(DEFAULT_PIPELINE_CONFIG)
    if config.llm.provider != "openrouter":
        _fail("PIPELINE_CONFIG_INVALID", "historical-anomaly-rag requires an OpenRouter pipeline configuration.", provider=config.llm.provider)
    pinned = _validate_pinned_model()
    kb_hash = _validate_kb(config)
    raw_rows, detector_manifest, cusum_hash = _load_cusum_input(cusum_windows_jsonl)
    selected = select_historical_cusum_windows(raw_rows)
    behavior_summary = historical_behavior_summary(
        wasp_dataset_dir=wasp_dataset_dir,
        model_path=PINNED_MODEL_PATH,
        manifest_path=PINNED_MANIFEST_PATH,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "deployment_id": deployment_id,
        "timezone": timezone_name,
        "selected_record_count": len(selected),
        "flagged_record_count": sum(bool(row["anomaly_flag"]) for row in selected),
        "control_record_count": sum(not bool(row["anomaly_flag"]) for row in selected),
        "expected_openrouter_calls": len(selected),  # system-auto queries bypass the router
        "openrouter_key_present": bool(os.environ.get("OPENROUTER_API_KEY")),
        "model": pinned,
        "wasp": {
            "source_sha256": behavior_summary["source_sha256"],
            "observed_windows": behavior_summary["observed_windows"],
            "source_csv_files": behavior_summary["source_csv_files"],
            "window_exclusions": behavior_summary["window_exclusions"],
        },
        "cusum": {"source_sha256": cusum_hash, "manifest_sha256": sha256_file(cusum_windows_jsonl.expanduser().resolve().parent / "manifest.json")},
        "knowledge_base_sha256": kb_hash,
        "detector_manifest_schema_version": detector_manifest["schema_version"],
        "interpretation": "Mechanical anomaly indicators only. The WASP result is separate historical context, not a measurement of an MmCows cow/day or evidence for an individual anomaly.",
    }
    return summary, selected, behavior_summary


def _provider_error(exc: Exception) -> HistoricalApplicationError:
    message = str(exc)
    if "OPENROUTER_API_KEY" in message:
        return HistoricalApplicationError("OPENROUTER_KEY_MISSING", "OPENROUTER_API_KEY is required for the live historical application run.", {})
    if "HTTP 401" in message or "HTTP 403" in message:
        return HistoricalApplicationError("OPENROUTER_AUTH_FAILED", "OpenRouter rejected the configured credentials.", {})
    if isinstance(exc, (urllib.error.URLError, TimeoutError, OSError)):
        return HistoricalApplicationError("OPENROUTER_UNAVAILABLE", "OpenRouter could not be reached. Retry later; no output was published.", {})
    return HistoricalApplicationError("OPENROUTER_RESPONSE_INVALID", "OpenRouter returned an invalid response. No output was published.", {})


def run(
    *,
    wasp_dataset_dir: Path,
    cusum_windows_jsonl: Path,
    output_dir: Path,
    deployment_id: str = DEFAULT_DEPLOYMENT_ID,
    timezone_name: str = DEFAULT_TIMEZONE,
    pipeline_config: Stage2Config | None = None,
    llm_client: LLMClient | None = None,
) -> dict[str, Any]:
    """Run the paid OpenRouter application and atomically publish five artifacts."""

    resolved_output = output_dir.expanduser().resolve()
    if resolved_output.exists():
        _fail("OUTPUT_EXISTS", "--output-dir must name a new, nonexistent directory.", output_dir=str(resolved_output))
    for input_path in (wasp_dataset_dir.expanduser().resolve(), cusum_windows_jsonl.expanduser().resolve()):
        if resolved_output == input_path or resolved_output in input_path.parents or input_path in resolved_output.parents:
            _fail(
                "UNSAFE_OUTPUT_DIRECTORY",
                "--output-dir must be separate from every staged input path.",
                output_dir=str(resolved_output),
                input_path=str(input_path),
            )
    config = pipeline_config or load_config(DEFAULT_PIPELINE_CONFIG)
    preflight_summary, selected, behavior_summary = preflight(
        wasp_dataset_dir=wasp_dataset_dir,
        cusum_windows_jsonl=cusum_windows_jsonl,
        deployment_id=deployment_id,
        timezone_name=timezone_name,
        pipeline_config=config,
    )
    try:
        client = llm_client or build_llm_client(config.llm)
        records = historical_demo_anomaly_records(
            selected,
            deployment_id=deployment_id,
            timezone=timezone_name,
            behavior_summary=behavior_summary,
        )
        explanations: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        for record in records:
            response = run_pipeline(record, None, config=config, llm_client=client)
            AnomalyExplanationResponse.model_validate(response)
            response_dump = response.model_dump(mode="json")
            explanations.append(response_dump)
            audit_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "query_id": response.query_id,
                    "window_id": record["window_id"],
                    "cow_id": record["cow_id"],
                    "path_taken": response.path_taken,
                    "latency_ms": response.latency_ms,
                    "record_sha256": _stable_hash(record),
                    "response_sha256": _stable_hash(response_dump),
                    "model_sha256": behavior_summary["model_sha256"],
                }
            )
    except (OpenRouterError, urllib.error.URLError, TimeoutError, OSError) as exc:
        raise _provider_error(exc) from exc

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact": "historical_anomaly_rag_application",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repository_revision": _repository_revision(),
        "pipeline_config_sha256": _stable_hash(config.model_dump(mode="json")),
        "tau": config.fallback.tau,
        "tau_policy": "balanced operational demo gate; not clinical calibration",
        "record_count": len(records),
        "explanation_count": len(explanations),
        "path_counts": {path: sum(item["path_taken"] == path for item in explanations) for path in sorted({item["path_taken"] for item in explanations})},
        "inputs": preflight_summary,
        "behavior_summary_sha256": _stable_hash(behavior_summary),
        "anomaly_records_sha256": _stable_hash(records),
        "explanations_sha256": _stable_hash(explanations),
        "audit_sha256": _stable_hash(audit_rows),
        "raw_sensor_rows_persisted": False,
        "interpretation": "Mechanical anomaly indicators and grounded explanatory assistance only; never a clinical or veterinary diagnosis.",
    }

    def writer(temp_dir: Path) -> None:
        write_jsonl(temp_dir / "anomaly_records.jsonl", records)
        write_jsonl(temp_dir / "explanations.jsonl", explanations)
        write_json(temp_dir / "historical_behavior_summary.json", behavior_summary)
        write_jsonl(temp_dir / "audit.jsonl", audit_rows)
        write_json(temp_dir / "run_manifest.json", manifest)

    published = atomic_output_dir(output_dir, writer)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "output_dir": str(published),
        "record_count": len(records),
        "path_counts": manifest["path_counts"],
        "tau": config.fallback.tau,
        "model_sha256": behavior_summary["model_sha256"],
        "interpretation": manifest["interpretation"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="historical-anomaly-rag",
        description="Run the pinned public WASP historical context beside MmCows CUSUM anomaly indicators. Never diagnoses.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("preflight", "Validate staged derived inputs, model pin, cadence, and KB without writing or calling OpenRouter."),
        ("run", "Make explicit paid OpenRouter calls and atomically publish the historical application artifacts."),
    ):
        subparser = commands.add_parser(command, help=help_text)
        subparser.add_argument("--wasp-dataset-dir", required=True, type=Path)
        subparser.add_argument("--cusum-windows-jsonl", required=True, type=Path)
        subparser.add_argument("--deployment-id", default=DEFAULT_DEPLOYMENT_ID)
        subparser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
        subparser.add_argument("--pipeline-config", type=Path, default=DEFAULT_PIPELINE_CONFIG)
        if command == "run":
            subparser.add_argument("--output-dir", required=True, type=Path, help="A new, nonexistent output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.pipeline_config)
        if args.command == "preflight":
            summary, _, _ = preflight(
                wasp_dataset_dir=args.wasp_dataset_dir,
                cusum_windows_jsonl=args.cusum_windows_jsonl,
                deployment_id=args.deployment_id,
                timezone_name=args.timezone,
                pipeline_config=config,
            )
        else:
            summary = run(
                wasp_dataset_dir=args.wasp_dataset_dir,
                cusum_windows_jsonl=args.cusum_windows_jsonl,
                output_dir=args.output_dir,
                deployment_id=args.deployment_id,
                timezone_name=args.timezone,
                pipeline_config=config,
            )
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except (HistoricalApplicationError, BehaviorError) as exc:
        print(json.dumps({"code": exc.code, "message": exc.message, "details": exc.details}, sort_keys=True, allow_nan=False), file=sys.stderr)
        # `run()` maps provider failures before this boundary so no vendor
        # body, key, endpoint, or raw response can leak through the CLI.
        return 3 if exc.code.startswith("OPENROUTER_") else 2
    except (OpenRouterError, urllib.error.URLError, TimeoutError, OSError) as exc:
        mapped = _provider_error(exc)
        print(json.dumps({"code": mapped.code, "message": mapped.message, "details": mapped.details}, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 3
    except ValidationError as exc:
        print(json.dumps({"code": "OPENROUTER_RESPONSE_INVALID", "message": "The provider output did not match the required response contract.", "details": {}}, sort_keys=True), file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
