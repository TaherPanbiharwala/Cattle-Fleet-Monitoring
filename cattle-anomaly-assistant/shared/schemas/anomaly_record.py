"""Stage 1's output; Stage 2's primary input. See LLM_Diagnostic_Assistant_PRD.md Section 6.1."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from ._base import StrictModel

BehaviorState = Literal["walking", "grazing", "resting", "miscellaneous"]
BehaviorContextSource = Literal["same_cow_runtime", "wasp_public_historical_dataset"]
BehaviorContextRelation = Literal["same_cow_day", "cross_dataset_historical_demo"]

_DISTRIBUTION_SUM_TOLERANCE = 0.01


class BehaviorStateDistribution24h(StrictModel):
    walking: float = Field(ge=0, le=1)
    grazing: float = Field(ge=0, le=1)
    resting: float = Field(ge=0, le=1)
    miscellaneous: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _proportions_sum_to_one(self) -> "BehaviorStateDistribution24h":
        total = self.walking + self.grazing + self.resting + self.miscellaneous
        if abs(total - 1.0) > _DISTRIBUTION_SUM_TOLERANCE:
            raise ValueError(f"behavior_state_distribution_24h must sum to ~1.0, got {total:.4f}")
        return self


class AnomalyHistoryEntry(StrictModel):
    timestamp: AwareDatetime
    flag: bool
    driving_signals: list[str] = Field(default_factory=list)


class AnomalyRecord(StrictModel):
    cow_id: str
    timestamp: AwareDatetime
    window_id: str

    behavior_state: BehaviorState
    behavior_state_confidence: float = Field(ge=0, le=1)
    behavior_state_distribution_24h: BehaviorStateDistribution24h
    behavior_context_source: BehaviorContextSource = "same_cow_runtime"
    behavior_context_relation: BehaviorContextRelation = "same_cow_day"
    behavior_model_sha256: str | None = Field(default=None, min_length=16)

    cbt_c: float
    cbt_deviation_sigma: float
    cbt_cusum_value: float

    lying_time_pct_24h: float = Field(ge=0, le=100)
    lying_time_deviation_sigma: float

    thi: float

    activity_magnitude_deviation_sigma: float | None = None
    herd_isolation_score: float | None = Field(default=None, ge=0, le=1)

    anomaly_flag: bool
    anomaly_score: float = Field(ge=0, le=1)
    driving_signals: list[str] = Field(default_factory=list)
    recent_anomaly_history: list[AnomalyHistoryEntry] = Field(default_factory=list)
