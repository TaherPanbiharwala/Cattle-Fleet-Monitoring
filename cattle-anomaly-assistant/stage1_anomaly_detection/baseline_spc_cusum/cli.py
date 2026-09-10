"""Command-line interface for the MmCows baseline detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .errors import DetectorError, fail
from .pipeline import publish_output, run_detector


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baseline-spc-cusum",
        description="Build daily MmCows per-cow baseline and CUSUM anomaly indicators. Never outputs a diagnosis.",
    )
    parser.add_argument("--data-root", required=True, type=Path, help="Extracted MmCows root or its main_data directory.")
    parser.add_argument("--output-dir", type=Path, help="Separate directory for derived JSONL artifacts.")
    parser.add_argument("--config", type=Path, help="Optional detector JSON configuration.")
    parser.add_argument("--injection-config", type=Path, help="Optional M1c monitoring-feature injection JSON.")
    parser.add_argument("--validate-only", action="store_true", help="Validate inputs and calculate readiness without writing output.")
    parser.add_argument("--require-immu", action="store_true", help="Treat missing IMMU acceleration as a validation failure.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output directory after a successful run.")
    parser.add_argument("--verbose", action="store_true", help="Print a derived-only completion summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.validate_only and args.output_dir is None:
            fail("OUTPUT_REQUIRED", "--output-dir is required unless --validate-only is used.")
        if args.validate_only and args.overwrite:
            fail("INVALID_ARGUMENTS", "--overwrite cannot be used with --validate-only.")
        config = load_config(args.config)
        result = run_detector(
            args.data_root,
            config,
            require_immu=args.require_immu,
            injection_path=args.injection_config,
        )
        summary = result.summary()
        if args.verbose:
            summary["baseline_availability"] = [
                {
                    "cow_id": baseline.cow_id,
                    "signal": baseline.signal,
                    "available": baseline.available,
                    "reason": baseline.reason,
                }
                for baseline in result.detection.baselines
            ]
        if args.validate_only:
            print(json.dumps({"status": "validated", **summary}, sort_keys=True))
            return 0
        assert args.output_dir is not None
        published = publish_output(result, config, data_root=args.data_root, output_dir=args.output_dir, overwrite=args.overwrite)
        print(json.dumps({"status": "complete", "output_dir": str(published), **summary}, sort_keys=True))
        return 0
    except DetectorError as exc:
        print(json.dumps({"code": exc.code, "message": exc.message, "details": exc.details}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - exercised through __main__
    raise SystemExit(main())
