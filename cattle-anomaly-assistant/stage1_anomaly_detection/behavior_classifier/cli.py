"""Discoverable, JSON-summary CLI for M1a behavior and M1d fusion work."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .artifact import legacy_pickle_provenance, verify_artifact
from .benchmark import preflight, run_benchmark
from .context import adapt_cusum_window, aggregate_daily_context, fuse_daily_records
from .errors import BehaviorError, fail
from .io import atomic_output_dir, read_jsonl, write_json, write_jsonl
from .runtime import predict_runtime_windows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="behavior-classifier", description="WASP public benchmark and future same-cow runtime behavior context tools. Never diagnoses.")
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight", help="Validate public WASP source data; writes no output.")
    preflight_parser.add_argument("--dataset-dir", required=True, type=Path, help="Public db-cow-walking directory with the four label folders.")
    preflight_parser.add_argument("--legacy-pickle", type=Path, help="Inspection-only legacy .pkl reference; it is hashed but never loaded.")
    benchmark_parser = commands.add_parser("benchmark", help="Run nested LOCO on public WASP data and write derived benchmark artifacts.")
    benchmark_parser.add_argument("--dataset-dir", required=True, type=Path)
    benchmark_parser.add_argument("--output-dir", required=True, type=Path, help="New separate output directory; no overwrite is available.")
    benchmark_parser.add_argument("--legacy-pickle", type=Path, help="Optional inspection-only legacy .pkl reference recorded by hash if an artifact qualifies.")
    verify_parser = commands.add_parser("verify-artifact", help="Verify a native JSON benchmark artifact; writes no output.")
    verify_parser.add_argument("--model-path", required=True, type=Path)
    verify_parser.add_argument("--manifest-path", type=Path)
    predict_parser = commands.add_parser("predict", help="Read same-cow runtime IMU JSONL and write derived behavior predictions only.")
    predict_parser.add_argument("--model-path", required=True, type=Path)
    predict_parser.add_argument("--manifest-path", type=Path)
    predict_parser.add_argument("--input-jsonl", required=True, type=Path, help="Input-only raw runtime 50x6 windows; never copied to output.")
    predict_parser.add_argument("--output-dir", required=True, type=Path)
    aggregate_parser = commands.add_parser("aggregate-context", help="Aggregate derived runtime predictions into daily behavior contexts.")
    aggregate_parser.add_argument("--predictions-jsonl", required=True, type=Path)
    aggregate_parser.add_argument("--timezone", required=True, help="IANA local timezone for daily grouping.")
    aggregate_parser.add_argument("--expected-windows", required=True, type=int, help="Expected runtime prediction windows for each cow-day.")
    aggregate_parser.add_argument("--output-dir", required=True, type=Path)
    fusion_parser = commands.add_parser("fuse-daily", help="Fail-closed join of same-cow runtime behavior contexts and CUSUM results into AnomalyRecords.")
    fusion_parser.add_argument("--behavior-context-jsonl", required=True, type=Path)
    fusion_parser.add_argument("--cusum-windows-jsonl", required=True, type=Path, help="Typed CUSUM context or existing derived anomaly_windows.jsonl.")
    fusion_parser.add_argument("--deployment-id", required=True, help="Shared deployment identity required for every CUSUM row.")
    fusion_parser.add_argument("--timezone", required=True, help="IANA local timezone for legacy CUSUM rows.")
    fusion_parser.add_argument("--cusum-source-kind", choices=("same_cow_runtime", "mmcows_public"), default="mmcows_public", help="Current MmCows output is public and intentionally cannot be fused.")
    fusion_parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def _publish_records(output_dir: Path, filename: str, records: list[dict[str, Any]], summary: dict[str, Any]) -> Path:
    def writer(temp_dir: Path) -> None:
        write_jsonl(temp_dir / filename, records)
        write_json(temp_dir / "summary.json", summary)
    return atomic_output_dir(output_dir, writer)


def _typed_or_legacy_cusum(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        if "local_date" in row and "source_kind" in row:
            normalized.append(row)
        else:
            normalized.append(adapt_cusum_window(row, deployment_id=args.deployment_id, source_kind=args.cusum_source_kind, timezone=args.timezone))
    return normalized


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            summary = preflight(args.dataset_dir)
            summary["legacy_pickle_provenance"] = legacy_pickle_provenance(args.legacy_pickle)
        elif args.command == "benchmark":
            result = run_benchmark(args.dataset_dir, args.output_dir, legacy_pickle=args.legacy_pickle)
            summary = {"status": "complete", "output_dir": str(result.output_dir), "artifact_eligible": result.artifact_eligible, "primary_mean_outer_fold_macro_f1": result.primary_mean_macro_f1, "pooled_out_of_fold_macro_f1": result.pooled_oof_macro_f1, "benchmark_only": True}
        elif args.command == "verify-artifact":
            summary = verify_artifact(args.model_path, args.manifest_path)
        elif args.command == "predict":
            records = predict_runtime_windows(read_jsonl(args.input_jsonl), model_path=args.model_path, manifest_path=args.manifest_path)
            published = _publish_records(args.output_dir, "behavior_predictions.jsonl", records, {"schema_version": 1, "prediction_count": len(records), "raw_sensor_rows_persisted": False, "benchmark_only": True})
            summary = {"status": "complete", "output_dir": str(published), "prediction_count": len(records), "raw_sensor_rows_persisted": False}
        elif args.command == "aggregate-context":
            contexts = aggregate_daily_context(read_jsonl(args.predictions_jsonl), timezone=args.timezone, expected_windows=args.expected_windows)
            published = _publish_records(args.output_dir, "behavior_daily_contexts.jsonl", contexts, {"schema_version": 1, "context_count": len(contexts), "raw_sensor_rows_persisted": False, "benchmark_only": True})
            summary = {"status": "complete", "output_dir": str(published), "context_count": len(contexts), "raw_sensor_rows_persisted": False}
        elif args.command == "fuse-daily":
            contexts = read_jsonl(args.behavior_context_jsonl)
            cusums = _typed_or_legacy_cusum(read_jsonl(args.cusum_windows_jsonl), args)
            records = fuse_daily_records(contexts, cusums)
            published = _publish_records(args.output_dir, "anomaly_records.jsonl", records, {"schema_version": 1, "record_count": len(records), "behavior_context_only": True, "interpretation": "Mechanical anomaly indicators only; not a clinical or veterinary diagnosis.", "raw_sensor_rows_persisted": False})
            summary = {"status": "complete", "output_dir": str(published), "record_count": len(records), "behavior_context_only": True}
        else:  # pragma: no cover - argparse enforces the command set
            fail("INVALID_ARGUMENTS", "Unknown behavior-classifier command.")
        print(json.dumps(summary, sort_keys=True, allow_nan=False))
        return 0
    except BehaviorError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message, "details": exc.details}, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
