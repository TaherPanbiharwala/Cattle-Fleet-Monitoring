"""Stage 1's output; Stage 2's primary input. See LLM_Diagnostic_Assistant_PRD.md Section 6.1."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from ._base import StrictModel

BehaviorState = Literal["walking", "grazing", "resting", "miscellaneous"]
BehaviorContextSource = Literal["same_cow_runtime", "wasp_public_historical_dataset"]
BehaviorContextRelation = Literal["same_cow_day", "cross_dataset_historical_demo"]

_DISTRIBUTION_SUM_TOLERANCE = 0.01
HISTORICAL_BEHAVIOR_DISCLAIMER = (
    "Historical aggregate from the separate WASP public dataset; it is not a measurement of this "
    "MmCows cow or day and is not evidence for this anomaly indicator."
)


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


class HistoricalBehaviorContext(StrictModel):
    """A labelled aggregate from a separate public behaviour dataset.

    This is deliberately not a cow-day measurement.  The explicit relation
    and fixed disclaimer make it impossible to accidentally reuse it as the
    runtime ``behavior_state_*`` contract.
    """

    schema_version: int = 1
    source_kind: Literal["wasp_public_historical_dataset"] = "wasp_public_historical_dataset"
    context_relation: Literal["cross_dataset_historical_demo"] = "cross_dataset_historical_demo"
    model_sha256: str = Field(min_length=64, max_length=64)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    behavior_state: BehaviorState
    behavior_state_confidence: float = Field(ge=0, le=1)
    behavior_state_distribution: BehaviorStateDistribution24h
    observed_windows: int = Field(gt=0)
    source_csv_files: int = Field(gt=0)
    confidence_kind: Literal["uncalibrated_max_class_probability"] = "uncalibrated_max_class_probability"
    disclaimer: str = HISTORICAL_BEHAVIOR_DISCLAIMER

    @model_validator(mode="after")
    def _requires_fixed_disclaimer(self) -> "HistoricalBehaviorContext":
        if self.disclaimer != HISTORICAL_BEHAVIOR_DISCLAIMER:
            raise ValueError("historical_behavior_context disclaimer must use the fixed public-data boundary")
        return self


class AnomalyHistoryEntry(StrictModel):
    timestamp: AwareDatetime
    flag: bool
    driving_signals: list[str] = Field(default_factory=list)


class AnomalyRecord(StrictModel):
    cow_id: str
    timestamp: AwareDatetime
    window_id: str

    behavior_state: BehaviorState | None = None
    behavior_state_confidence: float | None = Field(default=None, ge=0, le=1)
    behavior_state_distribution_24h: BehaviorStateDistribution24h | None = None
    behavior_context_source: BehaviorContextSource = "same_cow_runtime"
    behavior_context_relation: BehaviorContextRelation = "same_cow_day"
    behavior_model_sha256: str | None = Field(default=None, min_length=16)
    historical_behavior_context: HistoricalBehaviorContext | None = None

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

    @model_validator(mode="after")
    def _validate_behavior_context(self) -> "AnomalyRecord":
        daily_values = (
            self.behavior_state,
            self.behavior_state_confidence,
            self.behavior_state_distribution_24h,
        )
        if self.behavior_context_relation == "same_cow_day":
            if self.behavior_context_source != "same_cow_runtime":
                raise ValueError("same_cow_day behavior context must use same_cow_runtime provenance")
            if any(value is None for value in daily_values):
                raise ValueError("same_cow_day records require all behavior_state daily fields")
            if self.historical_behavior_context is not None:
                raise ValueError("same_cow_day records may not include historical_behavior_context")
            return self

        if self.behavior_context_source != "wasp_public_historical_dataset":
            raise ValueError("cross_dataset_historical_demo records require WASP public provenance")
        if any(value is not None for value in daily_values):
            raise ValueError("cross_dataset_historical_demo records may not use cow-day behavior_state fields")
        if self.historical_behavior_context is None:
            raise ValueError("cross_dataset_historical_demo records require historical_behavior_context")
        if self.behavior_model_sha256 != self.historical_behavior_context.model_sha256:
            raise ValueError("behavior_model_sha256 must match historical_behavior_context.model_sha256")
        return self
