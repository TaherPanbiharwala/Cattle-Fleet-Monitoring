"""Shared loading for the derived historical evaluation corpus.

The default corpus is deliberately external to Git because it contains
provenance tied to a user's staged derived Stage 1 outputs.  Callers either
pass its three JSONL files explicitly or set ``HISTORICAL_GOLDEN_DIR`` after
building it.  Missing files fail loudly rather than silently becoming an
empty evaluation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

from shared.schemas import AnomalyExplanationQuery, GoldenCase

_GOLDEN_DIR = Path(__file__).parent / "eval" / "golden"


def default_golden_files() -> list[Path]:
    """Return the user-built 24 real + 24 injected + 7 adversarial corpus."""

    root = Path(os.environ.get("HISTORICAL_GOLDEN_DIR", _GOLDEN_DIR / "historical-built"))
    return [root / "real_cases.jsonl", root / "injected_cases.jsonl", root / "adversarial_cases.jsonl"]


# Kept as a constant import surface for existing CLI modules. Environment
# selection intentionally happens only when the process imports this module.
DEFAULT_GOLDEN_FILES = default_golden_files()


def load_cases(paths: list[Path]) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. Build the derived 24 real + 24 injected + 7 adversarial corpus with "
                "stage2_rag_assistant.eval.build_historical_golden_cases, then pass all three files with "
                "--golden-file or set HISTORICAL_GOLDEN_DIR."
            )
        with path.open() as f:
            cases.extend(GoldenCase.model_validate_json(line) for line in f if line.strip())
    return cases


def query_for(case: GoldenCase) -> AnomalyExplanationQuery | None:
    if case.query_text is None:
        return None
    return AnomalyExplanationQuery(
        query_id=f"eval-{case.case_id}",
        cow_id=case.input_record.cow_id,
        raw_text=case.query_text,
        submitted_by="vet",
        timestamp=datetime.now(timezone.utc),
    )
