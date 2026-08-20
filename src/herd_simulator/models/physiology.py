"""
physiology.py — Body temperature and ambient weather modeling.

Body temperature = per-animal baseline + diurnal curve + bounded Gaussian
noise + an optional fever offset (onset -> plateau -> recovery ramp).
Ambient temperature/humidity are shared across the whole herd — one
pasture, one weather (HerdSimulator PRD A7) — and follow the same diurnal
curve shape as body temperature.

THI is deliberately NOT computed here: it is delegated to
`herd_simulator.utils.geo.compute_thi`, the single canonical formula
(ADR-007), so there is exactly one implementation to keep in parity with
the (future) firmware.

Fever/heat events carry their own timing (onset/plateau/recovery seconds,
peak offset) as explicit parameters rather than reading them from global
config — that timing is scenario-event data (Master PRD "Scenario
contract"), owned by the not-yet-built scenario engine. This module only
knows how to shape a ramp given those numbers.

References:
  Master PRD: "Physiology"
  HerdSimulator PRD §6.4 (FR-10..FR-12), §9.3
  ADR-007: THI formula
"""

from __future__ import annotations

import math
import random

_HOURS_PER_DAY = 24.0
_PEAK_HOUR = 14.0  # ambient/body temperature peaks at ~14:00 (config comment)

# Physically plausible bounds. Values outside these are rejected rather than
# silently clamped — an out-of-bounds reading means a bug upstream, not a
# real physiological state (Master PRD: "Values outside configured physical
# bounds are rejected").
BODY_TEMP_MIN_C = 35.0
BODY_TEMP_MAX_C = 43.0
AMBIENT_TEMP_MIN_C = -10.0
AMBIENT_TEMP_MAX_C = 55.0


class PhysiologyBoundsError(ValueError):
    """A computed physiological value fell outside plausible bounds."""


def diurnal_fraction(hour_of_day: float, peak_hour: float = _PEAK_HOUR) -> float:
    """A value in [-1, 1] tracing a smooth day/night cosine curve, peaking
    at `peak_hour` and troughing 12 hours later."""
    phase = 2 * math.pi * (hour_of_day - peak_hour) / _HOURS_PER_DAY
    return math.cos(phase)


def ambient_temperature_c(hour_of_day: float, day_peak_c: float, night_trough_c: float) -> float:
    """Shared ambient temperature (°C) at a given hour, per the configured
    diurnal model (FR-12 default: synthesized curve)."""
    mid = (day_peak_c + night_trough_c) / 2.0
    amplitude = (day_peak_c - night_trough_c) / 2.0
    return mid + amplitude * diurnal_fraction(hour_of_day)


def ambient_humidity_pct(
    hour_of_day: float,
    mean_pct: float,
    std_pct: float,
    rng: random.Random,
) -> float:
    """Shared relative humidity (%): mean value with bounded Gaussian noise.
    Not modeled as diurnal (A7: one pasture, one weather — kept simple)."""
    value = rng.gauss(mean_pct, std_pct)
    return _clamp(value, 0.0, 100.0)


def body_temperature_c(
    baseline_c: float,
    hour_of_day: float,
    diurnal_amplitude_c: float,
    noise_std_c: float,
    rng: random.Random,
    fever_offset_c: float = 0.0,
) -> float:
    """One tick's body temperature: baseline + diurnal swing + noise +
    fever offset (FR-10). `fever_offset_c` should come from
    `fever_ramp_offset_c`, precomputed by the caller for this tick's
    elapsed time since fever onset."""
    diurnal = (diurnal_amplitude_c / 2.0) * diurnal_fraction(hour_of_day)
    noise = rng.gauss(0.0, noise_std_c)
    return baseline_c + diurnal + noise + fever_offset_c


def fever_ramp_offset_c(
    elapsed_s: float,
    onset_s: float,
    plateau_s: float,
    recovery_s: float,
    peak_offset_c: float,
) -> float:
    """Fever temperature offset (°C, >= 0) at `elapsed_s` since the fever
    event started.

    Three phases (Master PRD: "Fever injection has onset, plateau, and
    recovery"):
      - Onset    [0, onset_s):                        linear 0 -> peak_offset_c
      - Plateau  [onset_s, onset_s+plateau_s):         held at peak_offset_c
      - Recovery [onset_s+plateau_s, ...+recovery_s):  linear peak_offset_c -> 0
      - Before the event starts or after it fully ends: 0.0 (no fever).
    """
    if elapsed_s < 0:
        return 0.0

    if elapsed_s < onset_s:
        return peak_offset_c * (elapsed_s / onset_s) if onset_s > 0 else peak_offset_c

    elapsed_s -= onset_s
    if elapsed_s < plateau_s:
        return peak_offset_c

    elapsed_s -= plateau_s
    if elapsed_s < recovery_s:
        return peak_offset_c * (1.0 - elapsed_s / recovery_s) if recovery_s > 0 else 0.0

    return 0.0


def heat_stress_temperature_c(base_ambient_c: float, heat_injection_offset_c: float) -> float:
    """A heat-stress event changes *ambient* conditions, never THI directly
    (Master PRD: "Heat injection changes ambient conditions rather than
    THI directly"). Feed the returned value into
    `herd_simulator.utils.geo.compute_thi` alongside humidity."""
    return base_ambient_c + heat_injection_offset_c


def validate_body_temp(value_c: float) -> float:
    """Return `value_c` unchanged, or raise PhysiologyBoundsError."""
    if not (BODY_TEMP_MIN_C <= value_c <= BODY_TEMP_MAX_C):
        raise PhysiologyBoundsError(
            f"body_temp {value_c:.2f}°C outside plausible bounds "
            f"[{BODY_TEMP_MIN_C}, {BODY_TEMP_MAX_C}]"
        )
    return value_c


def validate_ambient_temp(value_c: float) -> float:
    """Return `value_c` unchanged, or raise PhysiologyBoundsError."""
    if not (AMBIENT_TEMP_MIN_C <= value_c <= AMBIENT_TEMP_MAX_C):
        raise PhysiologyBoundsError(
            f"ambient_temp {value_c:.2f}°C outside plausible bounds "
            f"[{AMBIENT_TEMP_MIN_C}, {AMBIENT_TEMP_MAX_C}]"
        )
    return value_c


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
