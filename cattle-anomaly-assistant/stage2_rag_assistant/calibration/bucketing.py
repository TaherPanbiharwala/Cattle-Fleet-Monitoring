"""Pure computation for the historical scenario-grouped workflow report.

This module splits whole scenario families, buckets uncalibrated LLM
confidence, and reports a held-out workflow slice. It never computes a
clinical calibration or a CUSUM detection-accuracy result.
"""

from __future__ import annotations

from collections import defaultdict

from shared.schemas import AnomalyExplanationResponse, GoldenCase
from shared.schemas._base import StrictModel

NUM_BUCKETS = 10


def split_calibration_test(cases: list[GoldenCase]) -> tuple[list[GoldenCase], list[GoldenCase]]:
    """Split whole injection scenarios deterministically, never their days.

    Repeated days from a sustained injected shift share a scenario and would
    leak the same constructed pattern across calibration/test if split by
    individual case.  Legacy fixtures without a scenario id remain readable
    by treating each old case as its own scenario.
    """
    groups: dict[str, list[GoldenCase]] = defaultdict(list)
    for case in cases:
        groups[case.scenario_id or f"legacy:{case.case_id}"].append(case)
    ordered_groups = [groups[key] for key in sorted(groups)]
    calibration = [case for group in ordered_groups[0::2] for case in group]
    test = [case for group in ordered_groups[1::2] for case in group]
    return sorted(calibration, key=lambda case: case.case_id), sorted(test, key=lambda case: case.case_id)


def bucket_label(confidence: float) -> str:
    """confidence is trusted to already be in [0,1] — AnomalyExplanationResponse's
    own Field(ge=0, le=1) guarantees this; this function does not re-validate.

    Rounds to the nearest integer percent before bucketing to dodge float
    noise at exact decile boundaries (e.g. 0.3 * 10 == 2.9999999999999996
    in raw float arithmetic).
    """
    index = min(round(confidence * 100) // 10, NUM_BUCKETS - 1)
    lower, upper = index / NUM_BUCKETS, (index + 1) / NUM_BUCKETS
    return f"{lower:.1f}-{upper:.1f}"


def reconstruct_llm_asserts_anomaly(response: AnomalyExplanationResponse) -> bool:
    """Algebraic inverse of fallback_gate.py's own
    `disagreement = llm_asserts_anomaly != stage1_anomaly_flag`, which
    flows unchanged into response.disagreement_flag. Only meaningful when
    path_taken == "llm_grounded" — callers must check that first (see
    evaluate_case below).

    Reads stage1_output.anomaly_flag only to fold it into this XOR.
    Nothing downstream may reinterpret this as a CUSUM accuracy measurement:
    the historical real cases are mechanical workflow indicators and the
    injection cases are deterministic detector checks, not health truth.
    """
    return response.stage1_output.anomaly_flag != response.disagreement_flag


class CalibrationCaseResult(StrictModel):
    """Internal-only, mirrors retriever.py's RetrievalResult precedent —
    not part of shared/schemas/. Deliberately has no stage1_output-shaped
    field; see reconstruct_llm_asserts_anomaly's docstring for why.
    """

    case_id: str
    scenario_id: str
    bucket: str
    confidence: float
    llm_asserts_anomaly: bool
    gold_anomaly_flag: bool
    correct: bool


def evaluate_case(case: GoldenCase, response: AnomalyExplanationResponse) -> CalibrationCaseResult | None:
    """None means "exclude" — the caller counts this by path_taken instead.
    Not data loss: only llm_grounded responses ever carry a stated
    confidence to bucket in the first place (response_builder.py sets
    confidence=None on every fallback path).
    """
    if response.path_taken != "llm_grounded":
        return None
    llm_asserts_anomaly = reconstruct_llm_asserts_anomaly(response)
    return CalibrationCaseResult(
        case_id=case.case_id,
        scenario_id=case.scenario_id or f"legacy:{case.case_id}",
        bucket=bucket_label(response.confidence),
        confidence=response.confidence,
        llm_asserts_anomaly=llm_asserts_anomaly,
        gold_anomaly_flag=case.gold_anomaly_flag,
        correct=llm_asserts_anomaly == case.gold_anomaly_flag,
    )


class BucketStats(StrictModel):
    bucket: str
    lower: float
    upper: float
    count: int
    correct_count: int
    accuracy: float | None  # None, not 0.0, when count == 0


def aggregate_by_bucket(results: list[CalibrationCaseResult]) -> list[BucketStats]:
    """Always emits all NUM_BUCKETS rows, even empty ones — an empty
    bucket on a small golden set is itself a signal worth showing, not
    hiding.
    """
    by_bucket: dict[str, list[CalibrationCaseResult]] = {}
    for result in results:
        by_bucket.setdefault(result.bucket, []).append(result)

    stats = []
    for index in range(NUM_BUCKETS):
        lower, upper = index / NUM_BUCKETS, (index + 1) / NUM_BUCKETS
        label = f"{lower:.1f}-{upper:.1f}"
        bucket_results = by_bucket.get(label, [])
        count = len(bucket_results)
        correct_count = sum(1 for r in bucket_results if r.correct)
        stats.append(
            BucketStats(
                bucket=label,
                lower=lower,
                upper=upper,
                count=count,
                correct_count=correct_count,
                accuracy=(correct_count / count) if count else None,
            )
        )
    return stats
