"""Command-line entry point for the Phase 3 public-data benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import BenchmarkConfig, run_benchmark
from .wandb_tracking import WandbSettings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ml.train",
        description="Run the cow-grouped WASP behaviour benchmark (not field validation).",
    )
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Attached private Kaggle WASP dataset path.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for aggregate benchmark artifacts.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="disabled")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-project", default="cattle-fleet-phase3")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--data-source-ref",
        default="WASP-lab/db-cow-walking (private Kaggle input)",
        help="Non-secret Kaggle dataset slug/version or other provenance label saved with the run.",
    )
    parser.add_argument(
        "--source-revision",
        default=None,
        help="Immutable Git commit used for this run; saved with aggregate artifacts and W&B metadata.",
    )
    parser.add_argument("--without-cnn", action="store_true", help="Skip only the experimental 1D CNN.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        seed=args.seed,
        include_cnn=not args.without_cnn,
        data_source_ref=args.data_source_ref,
        source_revision=args.source_revision,
        wandb=WandbSettings(
            mode=args.wandb_mode,
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=args.wandb_group or f"wasp-loso-seed-{args.seed}",
        ),
    )
    result = run_benchmark(args.dataset_dir, args.output_dir, config)
    print(
        json.dumps(
            {
                "selected_model": result.selected_model_name,
                "release_gate_passed": result.release_gate_passed,
                "mean_macro_f1": result.mean_macro_f1,
                "walking_recall": result.walking_recall,
                "unknown_recall": result.unknown_recall,
                "output_dir": str(result.output_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    raise SystemExit(main())
