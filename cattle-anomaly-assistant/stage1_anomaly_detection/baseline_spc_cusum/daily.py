"""Derived daily features and explicit stream-quality accounting."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from statistics import fmean

from .config import DetectorConfig
from .mmcows import AccelerationData, MmCowsInputs, ScalarSample, StreamData, _SHARED_COW_ID
from .models import DailyFeature, Quality


def _expected_count(cadence_seconds: float | None) -> int:
    if cadence_seconds is None or cadence_seconds <= 0:
        return 0
    return max(1, round(86_400 / cadence_seconds))


def _daily_scalar(samples: list[ScalarSample], cadence_seconds: float | None, config: DetectorConfig) -> dict[date, tuple[float | None, Quality]]:
    """Collapse samples to local calendar days without imputing a single value."""

    values_by_day: dict[date, dict[float, float]] = defaultdict(dict)
    for sample in samples:
        # Duplicate sensor timestamps count once for coverage.  Keeping the
        # final value is deterministic and avoids a retransmission inflating
        # a day's apparent completeness.
        values_by_day[sample.timestamp.date()][sample.timestamp.timestamp()] = sample.value
    expected = _expected_count(cadence_seconds)
    result: dict[date, tuple[float | None, Quality]] = {}
    for day, timestamp_values in values_by_day.items():
        values = list(timestamp_values.values())
        observed = len(values)
        coverage = observed / expected if expected else 0.0
        if not expected:
            quality = Quality(observed, 0, 0.0, False, "cadence_unavailable")
        elif coverage < config.coverage_minimum:
            quality = Quality(observed, expected, coverage, False, "coverage_below_minimum")
        else:
            quality = Quality(observed, expected, coverage, True)
        result[day] = (fmean(values) if values else None, quality)
    return result


def _daily_activity(data: AccelerationData, cow_id: str, config: DetectorConfig) -> dict[date, tuple[float | None, Quality]]:
    expected = _expected_count(data.cadence_seconds(cow_id))
    result: dict[date, tuple[float | None, Quality]] = {}
    for day, stats in data.daily.get(cow_id, {}).items():
        coverage = stats.count / expected if expected else 0.0
        if not expected:
            quality = Quality(stats.count, 0, 0.0, False, "cadence_unavailable")
        elif coverage < config.coverage_minimum:
            quality = Quality(stats.count, expected, coverage, False, "coverage_below_minimum")
        else:
            quality = Quality(stats.count, expected, coverage, True)
        result[day] = (stats.gravity_centered_rms(), quality)
    return result


def _missing_quality(cadence_seconds: float | None) -> Quality:
    return Quality(0, _expected_count(cadence_seconds), 0.0, False, "no_observed_samples")


def aggregate_daily_features(inputs: MmCowsInputs, config: DetectorConfig) -> list[DailyFeature]:
    """Create one derived-only feature row per cow/day.

    THI may be supplied as a shared environmental stream.  It is duplicated
    into each cow's *derived* feature row as context, never treated as an SPC
    driver.  No raw row reaches output or Stage 2.
    """

    shared_thi = _daily_scalar(inputs.thi.samples.get(_SHARED_COW_ID, []), inputs.thi.cadence_seconds(_SHARED_COW_ID), config)
    shared_thi_missing_quality = _missing_quality(inputs.thi.cadence_seconds(_SHARED_COW_ID))
    features: list[DailyFeature] = []
    cow_ids = sorted(set(inputs.cbt.samples) & set(inputs.ankle.samples))
    for cow_id in cow_ids:
        cbt_cadence = inputs.cbt.cadence_seconds(cow_id)
        ankle_cadence = inputs.ankle.cadence_seconds(cow_id)
        individual_thi_cadence = inputs.thi.cadence_seconds(cow_id)
        cbt = _daily_scalar(inputs.cbt.samples[cow_id], cbt_cadence, config)
        ankle = _daily_scalar(inputs.ankle.samples[cow_id], ankle_cadence, config)
        individual_thi = _daily_scalar(inputs.thi.samples.get(cow_id, []), individual_thi_cadence, config)
        thi = individual_thi or shared_thi
        thi_missing_quality = _missing_quality(individual_thi_cadence) if individual_thi else shared_thi_missing_quality
        activity = _daily_activity(inputs.immu, cow_id, config) if inputs.immu else {}
        cbt_missing_quality = _missing_quality(cbt_cadence)
        ankle_missing_quality = _missing_quality(ankle_cadence)
        activity_missing_quality = _missing_quality(inputs.immu.cadence_seconds(cow_id)) if inputs.immu else Quality(0, 0, 0.0, False, "immu_unavailable")
        all_days = sorted(set(cbt) | set(ankle) | set(thi) | set(activity))
        for day in all_days:
            cbt_value, cbt_quality = cbt[day] if day in cbt else (None, cbt_missing_quality)
            ankle_value, ankle_quality = ankle[day] if day in ankle else (None, ankle_missing_quality)
            thi_value, thi_quality = thi[day] if day in thi else (None, thi_missing_quality)
            activity_value, activity_quality = activity[day] if day in activity else (None, activity_missing_quality)
            features.append(
                DailyFeature(
                    cow_id=cow_id,
                    day=day,
                    cbt_c=cbt_value,
                    lying_time_pct_24h=ankle_value * 100 if ankle_value is not None else None,
                    thi=thi_value,
                    activity_magnitude=activity_value,
                    quality={
                        "cbt": cbt_quality,
                        "lying_time": ankle_quality,
                        "thi": thi_quality,
                        "activity_magnitude": activity_quality,
                    },
                )
            )
    return features
