"""Per-cow immutable baselines and conservative two-sided daily CUSUM."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from statistics import fmean, stdev
from zoneinfo import ZoneInfo

from .config import DetectorConfig
from .models import ALL_SIGNALS, CusumTrace, DailyFeature, SignalBaseline, SignalName


@dataclass(frozen=True)
class DetectionResult:
    baselines: list[SignalBaseline]
    traces: list[CusumTrace]
    windows: list[dict[str, object]]


def _consecutive_run(features: list[DailyFeature], signal: SignalName, count: int) -> list[DailyFeature] | None:
    """Return the earliest run of valid, adjacent daily windows for a signal."""

    run: list[DailyFeature] = []
    previous_day = None
    for feature in features:
        value = feature.value_for(signal)
        valid = feature.core_valid and feature.quality_for(signal).valid and value is not None
        if valid and (previous_day is None or feature.day == previous_day + timedelta(days=1)):
            run.append(feature)
        elif valid:
            run = [feature]
        else:
            run = []
        previous_day = feature.day
        if len(run) == count:
            return run
    return None


def learn_baselines(features: list[DailyFeature], config: DetectorConfig) -> list[SignalBaseline]:
    """Learn immutable per-cow reference distributions from observed days only."""

    by_cow: dict[str, list[DailyFeature]] = defaultdict(list)
    for feature in features:
        by_cow[feature.cow_id].append(feature)
    baselines: list[SignalBaseline] = []
    for cow_id, cow_features in sorted(by_cow.items()):
        cow_features.sort(key=lambda item: item.day)
        for signal in ALL_SIGNALS:
            run = _consecutive_run(cow_features, signal, config.baseline_days)
            if run is None:
                baselines.append(
                    SignalBaseline(cow_id, signal, None, None, None, None, False, "insufficient_consecutive_valid_days")
                )
                continue
            values = [feature.value_for(signal) for feature in run]
            assert all(value is not None for value in values)
            numeric_values = [float(value) for value in values if value is not None]
            sample_stddev = stdev(numeric_values)
            if sample_stddev == 0:
                baselines.append(
                    SignalBaseline(
                        cow_id,
                        signal,
                        run[0].day,
                        run[-1].day,
                        fmean(numeric_values),
                        0.0,
                        False,
                        "zero_variance_baseline",
                    )
                )
                continue
            baselines.append(
                SignalBaseline(
                    cow_id,
                    signal,
                    run[0].day,
                    run[-1].day,
                    fmean(numeric_values),
                    sample_stddev,
                    True,
                )
            )
    return baselines


def _trace_signal(features: list[DailyFeature], baseline: SignalBaseline, config: DetectorConfig) -> list[CusumTrace]:
    traces: list[CusumTrace] = []
    if not baseline.available or baseline.start_day is None or baseline.end_day is None or baseline.mean is None or baseline.stddev is None:
        return [
            CusumTrace(
                feature.cow_id,
                feature.day,
                baseline.signal,
                "unavailable",
                feature.value_for(baseline.signal),
                None,
                None,
                None,
                None,
                None,
                False,
                baseline.reason,
            )
            for feature in features
        ]

    positive = 0.0
    negative = 0.0
    previous_monitoring_day = None
    previous_monitoring_valid = False
    for feature in features:
        value = feature.value_for(baseline.signal)
        if feature.day < baseline.start_day:
            traces.append(
                CusumTrace(feature.cow_id, feature.day, baseline.signal, "unavailable", value, None, None, None, None, None, False, "before_baseline")
            )
            continue
        if baseline.start_day <= feature.day <= baseline.end_day:
            traces.append(
                CusumTrace(feature.cow_id, feature.day, baseline.signal, "baseline", value, None, 0.0, 0.0, 0.0, "none", False, None)
            )
            continue

        reset_reason = None
        if previous_monitoring_day is not None and feature.day != previous_monitoring_day + timedelta(days=1):
            reset_reason = "gap_between_monitoring_days"
        elif previous_monitoring_day is not None and not previous_monitoring_valid:
            reset_reason = "previous_monitoring_window_invalid"
        if reset_reason:
            positive = 0.0
            negative = 0.0

        valid = feature.core_valid and feature.quality_for(baseline.signal).valid and value is not None
        if not valid:
            positive = 0.0
            negative = 0.0
            traces.append(
                CusumTrace(
                    feature.cow_id,
                    feature.day,
                    baseline.signal,
                    "unavailable",
                    value,
                    None,
                    0.0,
                    0.0,
                    0.0,
                    "none",
                    False,
                    "invalid_monitoring_window" if reset_reason is None else reset_reason,
                )
            )
            previous_monitoring_day = feature.day
            previous_monitoring_valid = False
            continue

        deviation = (value - baseline.mean) / baseline.stddev
        positive = max(0.0, positive + deviation - config.cusum_k)
        negative = max(0.0, negative - deviation - config.cusum_k)
        active = max(positive, negative)
        direction = "above" if positive > negative else "below" if negative > positive else "none"
        traces.append(
            CusumTrace(
                feature.cow_id,
                feature.day,
                baseline.signal,
                "monitoring",
                value,
                deviation,
                positive,
                negative,
                active,
                direction,
                active >= config.cusum_h,
                reset_reason,
            )
        )
        previous_monitoring_day = feature.day
        previous_monitoring_valid = True
    return traces


def _window_records(features: list[DailyFeature], traces: list[CusumTrace], config: DetectorConfig) -> list[dict[str, object]]:
    by_window: dict[tuple[str, object], dict[str, CusumTrace]] = defaultdict(dict)
    for trace in traces:
        by_window[(trace.cow_id, trace.day)][trace.signal] = trace
    records: list[dict[str, object]] = []
    for feature in sorted(features, key=lambda item: (item.cow_id, item.day)):
        signal_traces = by_window[(feature.cow_id, feature.day)]
        active_values = [trace.active_cusum for trace in signal_traces.values() if trace.active_cusum is not None]
        max_active = max(active_values, default=0.0)
        drivers = [signal for signal in ALL_SIGNALS if signal_traces[signal].alarm]
        cbt_trace = signal_traces["cbt"]
        records.append(
            {
                "schema_version": 1,
                "cow_id": feature.cow_id,
                "window_id": f"{feature.cow_id}:{feature.day.isoformat()}",
                "timestamp": datetime.combine(feature.day, time.min, tzinfo=ZoneInfo(config.timezone)).isoformat(),
                "detector_state": "monitoring" if any(trace.state == "monitoring" for trace in signal_traces.values()) else "baseline_or_unavailable",
                "cbt_c": feature.cbt_c,
                "cbt_deviation_sigma": cbt_trace.deviation_sigma,
                "cbt_cusum_value": cbt_trace.active_cusum,
                "lying_time_pct_24h": feature.lying_time_pct_24h,
                "lying_time_deviation_sigma": signal_traces["lying_time"].deviation_sigma,
                "thi": feature.thi,
                "activity_magnitude_deviation_sigma": signal_traces["activity_magnitude"].deviation_sigma,
                "anomaly_flag": bool(drivers),
                "anomaly_score": min(max_active / (2 * config.cusum_h), 1.0),
                "driving_signals": drivers,
                "injected_signals": feature.injected_signals,
                "reference_baseline_verified_healthy": False,
            }
        )
    return records


def detect(features: list[DailyFeature], baselines: list[SignalBaseline], config: DetectorConfig) -> DetectionResult:
    """Run CUSUM against fixed baselines; no later window can alter a baseline."""

    by_cow_features: dict[str, list[DailyFeature]] = defaultdict(list)
    for feature in features:
        by_cow_features[feature.cow_id].append(feature)
    baseline_by_key = {(baseline.cow_id, baseline.signal): baseline for baseline in baselines}
    traces: list[CusumTrace] = []
    for cow_id, cow_features in sorted(by_cow_features.items()):
        cow_features.sort(key=lambda item: item.day)
        for signal in ALL_SIGNALS:
            traces.extend(_trace_signal(cow_features, baseline_by_key[(cow_id, signal)], config))
    return DetectionResult(baselines=baselines, traces=traces, windows=_window_records(features, traces, config))
