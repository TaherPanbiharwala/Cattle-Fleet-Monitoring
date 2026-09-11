"""Deterministic monitoring-feature injections for M1c validation.

This module intentionally receives derived daily features, never raw MmCows
sensor rows.  Baseline windows are immutable: a scenario can only perturb a
monitoring window after that signal's baseline has ended.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .errors import fail
from .models import DailyFeature, SignalBaseline

_SIGNAL_FIELDS = {
    "cbt": "cbt_c",
    "lying_time": "lying_time_pct_24h",
    "activity_magnitude": "activity_magnitude",
}


@dataclass(frozen=True)
class InjectionReport:
    scenario_ids: list[str]
    modified_windows: int


def _load_payload(path: Path) -> dict[str, Any]:
    if not path.is_file():
        fail("INJECTION_CONFIG_NOT_FOUND", "The injection config file does not exist.", path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail("INVALID_INJECTION_CONFIG", "The injection config must be valid JSON.", path=str(path), error=str(exc))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or not isinstance(payload.get("scenarios"), list):
        fail("INVALID_INJECTION_CONFIG", "Expected {'schema_version': 1, 'scenarios': [...]}.", path=str(path))
    return payload


def _parse_date(value: object, *, scenario_id: str) -> date:
    if not isinstance(value, str):
        fail("INVALID_INJECTION_CONFIG", "start_date must be YYYY-MM-DD.", scenario_id=scenario_id)
    try:
        return date.fromisoformat(value)
    except ValueError:
        fail("INVALID_INJECTION_CONFIG", "start_date must be YYYY-MM-DD.", scenario_id=scenario_id, start_date=value)


def apply_injections(
    features: list[DailyFeature], baselines: list[SignalBaseline], injection_path: Path | None
) -> tuple[list[DailyFeature], InjectionReport | None]:
    """Return a deep-copied monitoring table with declared shifts applied."""

    if injection_path is None:
        return features, None
    payload = _load_payload(injection_path)
    copied = copy.deepcopy(features)
    by_window = {(feature.cow_id, feature.day): feature for feature in copied}
    baseline_by_key = {(baseline.cow_id, baseline.signal): baseline for baseline in baselines}
    seen_ids: set[str] = set()
    modified: set[tuple[str, date]] = set()

    for scenario in payload["scenarios"]:
        if not isinstance(scenario, dict):
            fail("INVALID_INJECTION_CONFIG", "Each injection scenario must be an object.")
        scenario_id = scenario.get("scenario_id")
        cow_id = scenario.get("cow_id")
        duration_days = scenario.get("duration_days")
        changes = scenario.get("changes")
        if not isinstance(scenario_id, str) or not scenario_id or scenario_id in seen_ids:
            fail("INVALID_INJECTION_CONFIG", "scenario_id must be a unique non-empty string.", scenario_id=scenario_id)
        seen_ids.add(scenario_id)
        if not isinstance(cow_id, str) or not cow_id:
            fail("INVALID_INJECTION_CONFIG", "cow_id must be a non-empty string.", scenario_id=scenario_id)
        start_day = _parse_date(scenario.get("start_date"), scenario_id=scenario_id)
        if not isinstance(duration_days, int) or duration_days <= 0:
            fail("INVALID_INJECTION_CONFIG", "duration_days must be a positive integer.", scenario_id=scenario_id)
        if not isinstance(changes, list):
            fail("INVALID_INJECTION_CONFIG", "changes must be a list; use [] for an explicit control.", scenario_id=scenario_id)
        parsed_changes: list[tuple[str, float]] = []
        for change in changes:
            if not isinstance(change, dict):
                fail("INVALID_INJECTION_CONFIG", "Each change must be an object.", scenario_id=scenario_id)
            signal = change.get("signal")
            sigma = change.get("standard_deviations")
            if signal not in _SIGNAL_FIELDS:
                fail("INVALID_INJECTION_CONFIG", "Injection signals must use the Stage 1 short vocabulary.", scenario_id=scenario_id, signal=signal, supported=sorted(_SIGNAL_FIELDS))
            if not isinstance(sigma, (int, float)) or isinstance(sigma, bool) or sigma == 0:
                fail("INVALID_INJECTION_CONFIG", "standard_deviations must be a non-zero number.", scenario_id=scenario_id, signal=signal)
            parsed_changes.append((signal, float(sigma)))

        for offset in range(duration_days):
            day = start_day + timedelta(days=offset)
            feature = by_window.get((cow_id, day))
            if feature is None:
                fail("INJECTION_WINDOW_NOT_FOUND", "An injection day is not present in the derived monitoring table.", scenario_id=scenario_id, cow_id=cow_id, date=day.isoformat())
            for signal, standard_deviations in parsed_changes:
                baseline = baseline_by_key.get((cow_id, signal))
                if baseline is None or not baseline.available or baseline.stddev is None or baseline.end_day is None:
                    fail("INJECTION_BASELINE_UNAVAILABLE", "The requested signal has no usable personal baseline.", scenario_id=scenario_id, cow_id=cow_id, signal=signal)
                if day <= baseline.end_day:
                    fail("INJECTION_TOUCHES_BASELINE", "Injections may only alter monitoring windows after the immutable baseline.", scenario_id=scenario_id, cow_id=cow_id, signal=signal, date=day.isoformat(), baseline_end=baseline.end_day.isoformat())
                field_name = _SIGNAL_FIELDS[signal]
                current = getattr(feature, field_name)
                if current is None or not feature.quality_for(signal).valid:
                    fail("INJECTION_WINDOW_INVALID", "An injection cannot fabricate a missing or low-coverage measurement.", scenario_id=scenario_id, cow_id=cow_id, signal=signal, date=day.isoformat())
                # A scenario declares the intended baseline-relative daily
                # value, not an offset from an arbitrary observed value.
                # Thus a sustained +/-2.5 sigma injection produces three
                # CUSUM increments of +/-2.0 and deterministically crosses
                # the conservative h=5 threshold on its third day.
                assert baseline.mean is not None
                updated = baseline.mean + standard_deviations * baseline.stddev
                if signal == "lying_time" and not 0 <= updated <= 100:
                    fail("INJECTION_OUT_OF_RANGE", "The requested lying-time injection would leave the physical 0-100% range.", scenario_id=scenario_id, date=day.isoformat(), updated_value=updated)
                setattr(feature, field_name, updated)
                if signal not in feature.injected_signals:
                    feature.injected_signals.append(signal)
                modified.add((cow_id, day))

    for feature in copied:
        feature.injected_signals.sort()
    return copied, InjectionReport(scenario_ids=sorted(seen_ids), modified_windows=len(modified))
