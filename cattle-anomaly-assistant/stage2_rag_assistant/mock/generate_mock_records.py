"""Stage 2's unblock mechanism. See LLM_Diagnostic_Assistant_PRD.md Section 6.2.

Not statistically sophisticated by design — the PRD is explicit that this
generator's only job is producing schema-valid `AnomalyRecord`s to develop
Stage 2 against before Stage 1 exists. Swap it for
`stage1_anomaly_detection/to_anomaly_record.py`'s real output once that lands
(PRD Section 14's integration point); nothing downstream in Stage 2 should
need to change when that swap happens.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from shared.schemas import (
    AnomalyHistoryEntry,
    AnomalyRecord,
    BehaviorState,
    BehaviorStateDistribution24h,
)

_BEHAVIOR_STATES: tuple[BehaviorState, ...] = ("walking", "grazing", "resting", "miscellaneous")
_DRIVING_SIGNAL_POOL = ["cbt", "lying_time", "activity_magnitude", "behavior_state", "herd_isolation"]
_MOCK_EPOCH = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _random_behavior_distribution(rng: random.Random, dominant: BehaviorState) -> BehaviorStateDistribution24h:
    weights = {state: rng.uniform(0.05, 0.2) for state in _BEHAVIOR_STATES}
    weights[dominant] += 1.0
    total = sum(weights.values())
    return BehaviorStateDistribution24h(**{state: value / total for state, value in weights.items()})


def _random_history(rng: random.Random, timestamp: datetime) -> list[AnomalyHistoryEntry]:
    """A minority of records carry 1-3 prior-day entries, so code built against
    this mock actually exercises `recent_anomaly_history` non-empty, not just
    the always-valid empty-list default.
    """
    if rng.random() > 0.2:
        return []
    return [
        AnomalyHistoryEntry(
            timestamp=timestamp - timedelta(days=days_ago),
            flag=rng.random() < 0.5,
            driving_signals=rng.sample(_DRIVING_SIGNAL_POOL, k=1) if rng.random() < 0.5 else [],
        )
        for days_ago in range(1, rng.randint(2, 4))
    ]


def generate_mock_record(
    cow_id: str,
    *,
    timestamp: datetime | None = None,
    anomaly_flag: bool | None = None,
    rng: random.Random | None = None,
) -> AnomalyRecord:
    """Produce one schema-valid AnomalyRecord with realistic-looking noise."""
    rng = rng or random.Random()
    is_anomaly = anomaly_flag if anomaly_flag is not None else rng.random() < 0.15
    if timestamp is None:
        timestamp = _MOCK_EPOCH + timedelta(minutes=rng.randint(0, 60 * 24 * 30))

    dominant_behavior: BehaviorState = rng.choice(_BEHAVIOR_STATES)
    driving_signals: list[str] = rng.sample(_DRIVING_SIGNAL_POOL, k=rng.choice([1, 2])) if is_anomaly else []

    def deviation(signal: str) -> float:
        if signal not in driving_signals:
            return rng.uniform(-1.0, 1.0)
        magnitude = rng.uniform(1.8, 3.5)
        return magnitude if rng.random() < 0.5 else -magnitude

    cbt_deviation = deviation("cbt")
    lying_deviation = deviation("lying_time")
    activity_deviation = deviation("activity_magnitude")
    herd_isolation = rng.uniform(0.6, 0.95) if "herd_isolation" in driving_signals else rng.uniform(0.0, 0.3)

    return AnomalyRecord(
        cow_id=cow_id,
        timestamp=timestamp,
        window_id=f"{cow_id}-{rng.getrandbits(32):08x}",
        behavior_state=dominant_behavior,
        behavior_state_confidence=round(rng.uniform(0.6, 0.99), 2),
        behavior_state_distribution_24h=_random_behavior_distribution(rng, dominant_behavior),
        cbt_c=round(38.6 + cbt_deviation * 0.3, 2),
        cbt_deviation_sigma=round(cbt_deviation, 2),
        cbt_cusum_value=round(max(0.0, cbt_deviation) * rng.uniform(0.5, 1.5), 2),
        lying_time_pct_24h=round(max(0.0, min(100.0, 45.0 + lying_deviation * 5)), 1),
        lying_time_deviation_sigma=round(lying_deviation, 2),
        thi=round(rng.uniform(60.0, 78.0), 1),
        activity_magnitude_deviation_sigma=round(activity_deviation, 2) if rng.random() > 0.2 else None,
        herd_isolation_score=round(herd_isolation, 2) if rng.random() > 0.3 else None,
        anomaly_flag=is_anomaly,
        anomaly_score=round(rng.uniform(0.6, 0.95) if is_anomaly else rng.uniform(0.0, 0.3), 2),
        driving_signals=driving_signals,
        recent_anomaly_history=_random_history(rng, timestamp),
    )


def generate_mock_batch(
    n: int,
    *,
    cow_ids: list[str] | None = None,
    anomaly_rate: float = 0.15,
    seed: int | None = 42,
) -> list[AnomalyRecord]:
    """Reproducible for a given seed: same seed/n/cow_ids -> byte-identical records."""
    rng = random.Random(seed)
    cow_ids = cow_ids or [f"cow-{i:03d}" for i in range(1, 11)]
    return [
        generate_mock_record(rng.choice(cow_ids), anomaly_flag=rng.random() < anomaly_rate, rng=rng)
        for _ in range(n)
    ]


if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    for record in generate_mock_batch(count):
        print(record.model_dump_json())
