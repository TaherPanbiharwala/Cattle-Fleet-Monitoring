"""Internal, derived-only data structures for the baseline detector."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal


SignalName = Literal["cbt", "lying_time", "activity_magnitude"]
CORE_SIGNALS: tuple[SignalName, ...] = ("cbt", "lying_time")
ALL_SIGNALS: tuple[SignalName, ...] = ("cbt", "lying_time", "activity_magnitude")


@dataclass(frozen=True)
class Quality:
    observed_samples: int
    expected_samples: int
    coverage: float
    valid: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "observed_samples": self.observed_samples,
            "expected_samples": self.expected_samples,
            "coverage": round(self.coverage, 6),
            "valid": self.valid,
            "reason": self.reason,
        }


@dataclass
class DailyFeature:
    cow_id: str
    day: date
    cbt_c: float | None
    lying_time_pct_24h: float | None
    thi: float | None
    activity_magnitude: float | None
    quality: dict[str, Quality]
    injected_signals: list[str] = field(default_factory=list)

    @property
    def core_valid(self) -> bool:
        return all(self.quality[name].valid for name in ("cbt", "lying_time", "thi"))

    def value_for(self, signal: SignalName) -> float | None:
        if signal == "cbt":
            return self.cbt_c
        if signal == "lying_time":
            return self.lying_time_pct_24h
        return self.activity_magnitude

    def quality_for(self, signal: SignalName) -> Quality:
        if signal == "lying_time":
            return self.quality["lying_time"]
        return self.quality[signal]

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cow_id": self.cow_id,
            "window_id": f"{self.cow_id}:{self.day.isoformat()}",
            "date": self.day.isoformat(),
            "cbt_c": self.cbt_c,
            "lying_time_pct_24h": self.lying_time_pct_24h,
            "thi": self.thi,
            "activity_magnitude": self.activity_magnitude,
            "core_valid": self.core_valid,
            "quality": {name: quality.as_dict() for name, quality in sorted(self.quality.items())},
            "injected_signals": self.injected_signals,
        }


@dataclass(frozen=True)
class SignalBaseline:
    cow_id: str
    signal: SignalName
    start_day: date | None
    end_day: date | None
    mean: float | None
    stddev: float | None
    available: bool
    reason: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cow_id": self.cow_id,
            "signal": self.signal,
            "start_date": self.start_day.isoformat() if self.start_day else None,
            "end_date": self.end_day.isoformat() if self.end_day else None,
            "mean": self.mean,
            "sample_stddev": self.stddev,
            "available": self.available,
            "reason": self.reason,
            "reference_baseline_verified_healthy": False,
        }


@dataclass(frozen=True)
class CusumTrace:
    cow_id: str
    day: date
    signal: SignalName
    state: Literal["baseline", "monitoring", "unavailable"]
    value: float | None
    deviation_sigma: float | None
    positive_cusum: float | None
    negative_cusum: float | None
    active_cusum: float | None
    direction: Literal["above", "below", "none"] | None
    alarm: bool
    reset_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "cow_id": self.cow_id,
            "date": self.day.isoformat(),
            "signal": self.signal,
            "state": self.state,
            "value": self.value,
            "deviation_sigma": self.deviation_sigma,
            "positive_cusum": self.positive_cusum,
            "negative_cusum": self.negative_cusum,
            "active_cusum": self.active_cusum,
            "direction": self.direction,
            "alarm": self.alarm,
            "reset_reason": self.reset_reason,
        }
