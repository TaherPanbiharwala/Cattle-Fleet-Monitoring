"""Pure computation for PRD Section 10's calibration steps 1-3: split,
bucket, reconstruct, aggregate. Zero I/O, zero pipeline execution — see
tune_threshold.py for the runner that actually drives the pipeline.

Steps 4-6 (Stage 1's own accuracy, setting tau, re-validating on the test
split) are explicitly out of scope here and need real Stage 1 output —
see tune_threshold.py's module docstring for why step 4 isn't computed
even though it's technically possible on the current mock golden set.
"""

from __future__ import annotations

from shared.schemas import AnomalyExplanationResponse, GoldenCase
from shared.schemas._base import StrictModel

NUM_BUCKETS = 10


def split_calibration_test(cases: list[GoldenCase]) -> tuple[list[GoldenCase], list[GoldenCase]]:
    """Deterministic, no RNG — case order carries no meaning worth
    preserving via randomness. Sorts by case_id, then interleaves by
    parity (even index -> calibration, odd -> test) rather than a
    contiguous head/tail split: the real golden set's case_ids cluster
    alphabetically by category ("adv-..." block, then "mock-<category>-
    00/01/02" triplets), so a contiguous split would dump whole
    categories on one side by alphabetical accident. Interleaving
    distributes each cluster close to evenly instead.
    """
    ordered = sorted(cases, key=lambda c: c.case_id)
    return ordered[0::2], ordered[1::2]


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
    Nothing downstream may separately compare stage1_output.anomaly_flag
    against a case's gold_anomaly_flag — that's PRD Section 10 step 4
    ("Stage 1's own accuracy"), explicitly out of scope here.
    """
    return response.stage1_output.anomaly_flag != response.disagreement_flag


class CalibrationCaseResult(StrictModel):
    """Internal-only, mirrors retriever.py's RetrievalResult precedent —
    not part of shared/schemas/. Deliberately has no stage1_output-shaped
    field; see reconstruct_llm_asserts_anomaly's docstring for why.
    """

    case_id: str
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
