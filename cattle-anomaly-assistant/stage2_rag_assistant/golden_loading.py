"""Shared golden-set loading, used by both stage2_rag_assistant/eval/ and
stage2_rag_assistant/calibration/ — lives here, not nested under either,
since both depend on it equally and neither should own it.

real_cases.jsonl / injected_cases.jsonl (PRD Section 13's proposed tree)
are deliberately not created — they need real Stage 1 output that doesn't
exist yet (PRD Section 14's Integration Point). Pointing this at a
nonexistent file fails loudly rather than silently reporting "0 cases" as
if that were a legitimate result.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from shared.schemas import AnomalyExplanationQuery, GoldenCase

_GOLDEN_DIR = Path(__file__).parent / "eval" / "golden"
DEFAULT_GOLDEN_FILES = [_GOLDEN_DIR / "mock_derived_cases.jsonl", _GOLDEN_DIR / "adversarial_cases.jsonl"]


def load_cases(paths: list[Path]) -> list[GoldenCase]:
    cases: list[GoldenCase] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(
                f"{path} does not exist. real_cases.jsonl/injected_cases.jsonl need real Stage 1 output "
                "(PRD Section 14 Integration Point) and are deliberately not created yet — see "
                "LLM_ASSISTANT_STATUS.md before assuming this is a bug."
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
