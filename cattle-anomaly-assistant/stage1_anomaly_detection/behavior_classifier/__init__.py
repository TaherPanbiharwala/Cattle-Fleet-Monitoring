"""Leakage-safe WASP behaviour benchmarking and same-cow daily context tools.

This package intentionally has no dependency on the repository's legacy
``src/ml`` package and never deserialises Python pickle model artifacts.
"""

from .artifact import verify_artifact
from .benchmark import run_benchmark
from .context import aggregate_daily_context, fuse_daily_records

__all__ = ["aggregate_daily_context", "fuse_daily_records", "run_benchmark", "verify_artifact"]
