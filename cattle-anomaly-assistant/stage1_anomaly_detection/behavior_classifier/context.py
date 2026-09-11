"""Derived daily behavior context and fail-closed M1d AnomalyRecord fusion."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from typing import Any, Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import AwareDatetime, Field, ValidationError, field_validator

from shared.schemas import AnomalyRecord, BehaviorStateDistribution24h
from shared.schemas._base import StrictModel

from .errors import fail

SOURCE_KIND = Literal["same_cow_runtime", "wasp_public", "mmcows_public"]
STATE = Literal["walking", "grazing", "resting", "miscellaneous"]
SCHEMA_VERSION = 1


class BehaviorPrediction(StrictModel):
    schema_version: int = SCHEMA_VERSION
    cow_id: str = Field(min_length=1)
    timestamp: AwareDatetime
    deployment_id: str = Field(min_length=1)
    source_kind: SOURCE_KIND
    model_sha256: str = Field(min_length=16)
    behavior_state: STATE
    behavior_state_confidence: float = Field(ge=0, le=1)
    confidence_kind: Literal["uncalibrated_max_class_probability"] = "uncalibrated_max_class_probability"


class BehaviorDailyContext(StrictModel):
    schema_version: int = SCHEMA_VERSION
    cow_id: str = Field(min_length=1)
    local_date: date
    timezone: str = Field(min_length=1)
    deployment_id: str = Field(min_length=1)
    source_kind: SOURCE_KIND
    model_sha256: str = Field(min_length=16)
    behavior_state: STATE
    behavior_state_confidence: float = Field(ge=0, le=1)
    behavior_state_distribution_24h: BehaviorStateDistribution24h
    observed_windows: int = Field(ge=0)
    expected_windows: int = Field(gt=0)
    coverage: float = Field(ge=0, le=1)
    quality_status: Literal["valid"] = "valid"
    confidence_kind: Literal["uncalibrated_max_class_probability"] = "uncalibrated_max_class_probability"

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be an IANA timezone") from exc
        return value


class CusumDailyResult(StrictModel):
    schema_version: int = SCHEMA_VERSION
    cow_id: str = Field(min_length=1)
    local_date: date
    timezone: str = Field(min_length=1)
    deployment_id: str = Field(min_length=1)
    source_kind: SOURCE_KIND
    window_id: str = Field(min_length=1)
    cbt_c: float
    cbt_deviation_sigma: float
    cbt_cusum_value: float
    lying_time_pct_24h: float = Field(ge=0, le=100)
    lying_time_deviation_sigma: float
    thi: float
    activity_magnitude_deviation_sigma: float | None = None
    anomaly_flag: bool
    anomaly_score: float = Field(ge=0, le=1)
    driving_signals: list[str] = Field(default_factory=list)
    core_quality_valid: bool = True

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError("timezone must be an IANA timezone") from exc
        return value


def _model_rows(records: list[dict[str, Any]], model_type: Any, *, error_code: str) -> list[Any]:
    parsed: list[Any] = []
    for row_number, record in enumerate(records, start=1):
        try:
            parsed.append(model_type.model_validate(record))
        except ValidationError as exc:
            fail(error_code, "A derived context record does not match the versioned contract.", row=row_number, errors=exc.errors(include_url=False))
    return parsed


def aggregate_daily_context(prediction_records: list[dict[str, Any]], *, timezone: str, expected_windows: int) -> list[dict[str, Any]]:
    """Aggregate only derived predictions; it never writes source windows or samples."""

    if expected_windows < 1:
        fail("RUNTIME_CONTEXT_INVALID", "--expected-windows must be positive.", expected_windows=expected_windows)
    try:
        zone = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        fail("RUNTIME_CONTEXT_INVALID", "--timezone must be a valid IANA timezone.", timezone=timezone)
    predictions = _model_rows(prediction_records, BehaviorPrediction, error_code="RUNTIME_CONTEXT_INVALID")
    groups: dict[tuple[str, date, str], list[BehaviorPrediction]] = defaultdict(list)
    for prediction in predictions:
        groups[(prediction.cow_id, prediction.timestamp.astimezone(zone).date(), prediction.deployment_id)].append(prediction)
    contexts: list[dict[str, Any]] = []
    for (cow_id, day, deployment_id), group in sorted(groups.items()):
        source_kinds = {prediction.source_kind for prediction in group}
        model_hashes = {prediction.model_sha256 for prediction in group}
        if len(source_kinds) != 1 or len(model_hashes) != 1:
            fail("RUNTIME_CONTEXT_INVALID", "A daily context may not mix behavior sources or model artifacts.", cow_id=cow_id, local_date=day.isoformat())
        coverage = len(group) / expected_windows
        if coverage < 0.75:
            fail("CONTEXT_COVERAGE_INSUFFICIENT", "Behavior context has less than 75% observed window coverage.", cow_id=cow_id, local_date=day.isoformat(), observed_windows=len(group), expected_windows=expected_windows, coverage=coverage)
        counts = {state: sum(prediction.behavior_state == state for prediction in group) for state in ("walking", "grazing", "resting", "miscellaneous")}
        highest = max(counts.values())
        # Lexical order is deliberate and stable when daily counts tie.
        state = min(name for name, count in counts.items() if count == highest)
        winners = [prediction.behavior_state_confidence for prediction in group if prediction.behavior_state == state]
        context = BehaviorDailyContext(
            cow_id=cow_id,
            local_date=day,
            timezone=timezone,
            deployment_id=deployment_id,
            source_kind=next(iter(source_kinds)),
            model_sha256=next(iter(model_hashes)),
            behavior_state=state,
            behavior_state_confidence=sum(winners) / len(winners),
            behavior_state_distribution_24h={name: count / len(group) for name, count in counts.items()},
            observed_windows=len(group),
            expected_windows=expected_windows,
            coverage=coverage,
        )
        contexts.append(context.model_dump(mode="json"))
    if not contexts:
        fail("RUNTIME_CONTEXT_INVALID", "No behavior predictions were supplied for aggregation.")
    return contexts


def adapt_cusum_window(record: dict[str, Any], *, deployment_id: str, source_kind: str, timezone: str) -> dict[str, Any]:
    """Normalize existing derived CUSUM JSONL; public MmCows remains ineligible for fusion."""

    if source_kind not in {"same_cow_runtime", "mmcows_public"}:
        fail("RUNTIME_CONTEXT_INVALID", "CUSUM source kind is unsupported.", source_kind=source_kind)
    date_value = record.get("date")
    if not isinstance(date_value, str):
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, str):
            fail("RUNTIME_CONTEXT_INVALID", "CUSUM rows must provide a local date or an ISO-8601 timestamp.")
        try:
            parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed_timestamp.tzinfo is None:
                raise ValueError("naive timestamp")
            date_value = parsed_timestamp.astimezone(ZoneInfo(timezone)).date().isoformat()
        except (ValueError, ZoneInfoNotFoundError):
            fail("RUNTIME_CONTEXT_INVALID", "CUSUM timestamp or requested timezone is invalid.", timestamp=timestamp, timezone=timezone)
    return {
        "schema_version": SCHEMA_VERSION,
        "cow_id": record.get("cow_id"),
        "local_date": date_value,
        "timezone": timezone,
        "deployment_id": deployment_id,
        "source_kind": source_kind,
        "window_id": record.get("window_id"),
        "cbt_c": record.get("cbt_c"),
        "cbt_deviation_sigma": record.get("cbt_deviation_sigma"),
        "cbt_cusum_value": record.get("cbt_cusum_value"),
        "lying_time_pct_24h": record.get("lying_time_pct_24h"),
        "lying_time_deviation_sigma": record.get("lying_time_deviation_sigma"),
        "thi": record.get("thi"),
        "activity_magnitude_deviation_sigma": record.get("activity_magnitude_deviation_sigma"),
        "anomaly_flag": record.get("anomaly_flag"),
        "anomaly_score": record.get("anomaly_score"),
        "driving_signals": record.get("driving_signals", []),
        "core_quality_valid": all(record.get(key) is not None for key in ("cbt_c", "cbt_deviation_sigma", "cbt_cusum_value", "lying_time_pct_24h", "lying_time_deviation_sigma", "thi")),
    }


def _join_key(row: Any) -> tuple[str, date, str, str]:
    return row.cow_id, row.local_date, row.timezone, row.deployment_id


def fuse_daily_records(behavior_context_records: list[dict[str, Any]], cusum_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build schema-valid Stage 2 records from *aligned runtime* daily context only."""

    contexts = _model_rows(behavior_context_records, BehaviorDailyContext, error_code="RUNTIME_CONTEXT_INVALID")
    cusums = _model_rows(cusum_records, CusumDailyResult, error_code="RUNTIME_CONTEXT_INVALID")
    if not contexts or not cusums:
        fail("NO_JOINABLE_RECORDS", "Both behavior and CUSUM daily context inputs must contain records.")
    for row in [*contexts, *cusums]:
        if row.source_kind != "same_cow_runtime":
            fail("CONTEXT_SOURCE_NOT_RUNTIME", "Public WASP and MmCows data cannot be fused. Supply aligned same-cow runtime context instead.", source_kind=row.source_kind)
    context_by_key: dict[tuple[str, date, str, str], BehaviorDailyContext] = {}
    cusum_by_key: dict[tuple[str, date, str, str], CusumDailyResult] = {}
    for context in contexts:
        key = _join_key(context)
        if key in context_by_key:
            fail("CONTEXT_ALIGNMENT_FAILED", "Duplicate behavior daily context prevents an exact join.", key=[str(part) for part in key])
        if context.coverage < 0.75 or context.quality_status != "valid":
            fail("CONTEXT_COVERAGE_INSUFFICIENT", "Behavior daily context is not valid enough for fusion.", key=[str(part) for part in key])
        context_by_key[key] = context
    for cusum in cusums:
        key = _join_key(cusum)
        if key in cusum_by_key:
            fail("CONTEXT_ALIGNMENT_FAILED", "Duplicate CUSUM daily context prevents an exact join.", key=[str(part) for part in key])
        if not cusum.core_quality_valid:
            fail("CONTEXT_ALIGNMENT_FAILED", "CUSUM daily context lacks required valid physiology values.", key=[str(part) for part in key])
        if "behavior_state" in cusum.driving_signals:
            fail("CONTEXT_ALIGNMENT_FAILED", "Behavior is context-only in this release and may not be a CUSUM driver.", key=[str(part) for part in key])
        cusum_by_key[key] = cusum
    if set(context_by_key) != set(cusum_by_key):
        fail("CONTEXT_ALIGNMENT_FAILED", "Behavior and CUSUM daily keys do not match exactly.", behavior_only=len(set(context_by_key) - set(cusum_by_key)), cusum_only=len(set(cusum_by_key) - set(context_by_key)))
    history_by_cow: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output: list[dict[str, Any]] = []
    for key in sorted(context_by_key):
        behavior = context_by_key[key]
        cusum = cusum_by_key[key]
        zone = ZoneInfo(behavior.timezone)
        timestamp = datetime.combine(behavior.local_date + timedelta(days=1), time.min, tzinfo=zone).astimezone(ZoneInfo("UTC"))
        record = AnomalyRecord(
            cow_id=cusum.cow_id,
            timestamp=timestamp,
            window_id=cusum.window_id,
            behavior_state=behavior.behavior_state,
            behavior_state_confidence=behavior.behavior_state_confidence,
            behavior_state_distribution_24h=behavior.behavior_state_distribution_24h,
            cbt_c=cusum.cbt_c,
            cbt_deviation_sigma=cusum.cbt_deviation_sigma,
            cbt_cusum_value=cusum.cbt_cusum_value,
            lying_time_pct_24h=cusum.lying_time_pct_24h,
            lying_time_deviation_sigma=cusum.lying_time_deviation_sigma,
            thi=cusum.thi,
            activity_magnitude_deviation_sigma=cusum.activity_magnitude_deviation_sigma,
            herd_isolation_score=None,
            anomaly_flag=cusum.anomaly_flag,
            anomaly_score=cusum.anomaly_score,
            driving_signals=list(cusum.driving_signals),
            recent_anomaly_history=history_by_cow[cusum.cow_id][-3:],
        )
        dumped = record.model_dump(mode="json")
        output.append(dumped)
        history_by_cow[cusum.cow_id].append({"timestamp": dumped["timestamp"], "flag": dumped["anomaly_flag"], "driving_signals": dumped["driving_signals"]})
    return output
