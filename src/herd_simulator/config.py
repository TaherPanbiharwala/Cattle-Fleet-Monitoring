"""
config.py — Strict YAML schema validation with fail-fast error reporting.

Loads and validates the simulator configuration from a YAML file.
Reports the exact field name, current value, and expected format
on validation failure (DX requirement from autoplan).

References:
  - ADR-004 (ThingSpeak rate limits)
  - ADR-007 (risk score formula constants)
  - ADR-008 (geofence polygon)
  - ADR-009 (battery — no solar)
  - AGENTS.md §3 (non-negotiable rules)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ConfigError(Exception):
    """Raised when a config value fails validation."""

    def __init__(self, field_path: str, value: Any, expected: str) -> None:
        self.field_path = field_path
        self.value = value
        self.expected = expected
        super().__init__(
            f"Config validation failed:\n"
            f"  Field:    {field_path}\n"
            f"  Got:      {value!r}\n"
            f"  Expected: {expected}"
        )


def _require(raw: dict, key: str, parent: str = "") -> Any:
    """Return raw[key] or raise ConfigError if missing."""
    path = f"{parent}.{key}" if parent else key
    if key not in raw:
        raise ConfigError(path, "<missing>", "key must be present")
    return raw[key]


def _require_type(raw: dict, key: str, typ: type, parent: str = "") -> Any:
    val = _require(raw, key, parent)
    path = f"{parent}.{key}" if parent else key
    if not isinstance(val, typ):
        raise ConfigError(path, val, f"type {typ.__name__}")
    return val


def _require_positive(raw: dict, key: str, parent: str = "") -> float:
    val = _require(raw, key, parent)
    path = f"{parent}.{key}" if parent else key
    if not isinstance(val, (int, float)) or val <= 0:
        raise ConfigError(path, val, "positive number (> 0)")
    return float(val)


def _require_non_negative(raw: dict, key: str, parent: str = "") -> float:
    val = _require(raw, key, parent)
    path = f"{parent}.{key}" if parent else key
    if not isinstance(val, (int, float)) or val < 0:
        raise ConfigError(path, val, "non-negative number (>= 0)")
    return float(val)


def _require_int_range(raw: dict, key: str, lo: int, hi: int, parent: str = "") -> int:
    val = _require(raw, key, parent)
    path = f"{parent}.{key}" if parent else key
    if not isinstance(val, int) or not (lo <= val <= hi):
        raise ConfigError(path, val, f"integer in [{lo}, {hi}]")
    return val


# ---------------------------------------------------------------------------
# Dataclasses — typed config sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HerdConfig:
    n_sim: int
    n_total: int
    physical_collar_id: int


@dataclass(frozen=True)
class PhysiologyConfig:
    baseline_temp_mean: float
    baseline_temp_std: float
    diurnal_amplitude: float
    noise_std: float


@dataclass(frozen=True)
class WeatherConfig:
    ambient_temp_day: float
    ambient_temp_night: float
    humidity_mean: float
    humidity_std: float


@dataclass(frozen=True)
class BatteryConfig:
    initial_level: float
    base_drain_per_hour: float
    alert_drain_multiplier: float


@dataclass(frozen=True)
class MovementConfig:
    centroid_speed_m_per_s: float
    individual_offset_max_m: float
    flocking_strength: float


@dataclass(frozen=True)
class BehaviourTransitions:
    resting: dict[str, float]
    grazing: dict[str, float]
    ruminating: dict[str, float]
    walking: dict[str, float]
    restless: dict[str, float]


@dataclass(frozen=True)
class BehaviourConfig:
    transition_interval_s: int
    base_transitions: BehaviourTransitions


@dataclass(frozen=True)
class SeverityConfig:
    temp_offset_low: float
    temp_offset_high: float
    thi_low: float
    thi_high: float
    restless: float
    geo_warn: float
    geo_breach: float
    social_isolation: float
    collar_tamper: float


@dataclass(frozen=True)
class AlertBands:
    green_max: int
    yellow_max: int


@dataclass(frozen=True)
class RiskConfig:
    severity: SeverityConfig
    alert_bands: AlertBands


@dataclass(frozen=True)
class ThingSpeakConfig:
    channel_2_id: str
    write_cadence_s: int
    breach_cadence_s: int
    min_interval_s: int
    channel_1_sniff_interval_s: int
    channel_1_stale_threshold_s: int
    backoff_base_s: int
    backoff_max_s: int


@dataclass(frozen=True)
class LoggingConfig:
    log_dir: str
    ground_truth_enabled: bool
    telemetry_csv_enabled: bool
    events_jsonl_enabled: bool
    buffer_size: int


@dataclass(frozen=True)
class HudConfig:
    host: str
    port: int
    poll_interval_ms: int


@dataclass(frozen=True)
class SimulatorConfig:
    """Top-level validated configuration for the herd simulator."""

    herd: HerdConfig
    seed: int
    pasture_polygon: list[tuple[float, float]]
    physiology: PhysiologyConfig
    weather: WeatherConfig
    battery: BatteryConfig
    movement: MovementConfig
    behaviour: BehaviourConfig
    risk: RiskConfig
    thingspeak: ThingSpeakConfig
    logging: LoggingConfig
    hud: HudConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _parse_polygon(raw_list: list, parent: str) -> list[tuple[float, float]]:
    """Parse [[lat, lon], ...] into list of (lat, lon) tuples."""
    if not isinstance(raw_list, list) or len(raw_list) < 3:
        raise ConfigError(parent, raw_list, "list of ≥3 [lat, lon] coordinate pairs")
    result: list[tuple[float, float]] = []
    for i, pair in enumerate(raw_list):
        path = f"{parent}[{i}]"
        if not isinstance(pair, list) or len(pair) != 2:
            raise ConfigError(path, pair, "[lat, lon] pair (list of 2 floats)")
        lat, lon = pair
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            raise ConfigError(path, pair, "[float, float]")
        result.append((float(lat), float(lon)))
    return result


def _parse_transitions(raw: dict, parent: str) -> BehaviourTransitions:
    """Parse and validate behaviour transition probabilities."""
    states = ["resting", "grazing", "ruminating", "walking", "restless"]
    parsed: dict[str, dict[str, float]] = {}
    for state in states:
        path = f"{parent}.{state}"
        trans = raw.get(state, {})
        if not isinstance(trans, dict):
            raise ConfigError(path, trans, "dict of {target_state: probability}")
        total = sum(trans.values())
        if total > 1.0 + 1e-9:
            raise ConfigError(path, trans, f"probabilities sum ≤ 1.0 (got {total:.4f})")
        # Validate target states are known
        valid_targets = set(states)
        for target in trans:
            if target not in valid_targets:
                raise ConfigError(f"{path}.{target}", target, f"one of {states}")
        parsed[state] = {k: float(v) for k, v in trans.items()}
    return BehaviourTransitions(**parsed)


def load_config(path: str | Path) -> SimulatorConfig:
    """Load and validate a YAML config file. Raises ConfigError on any issue."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigError("<root>", type(raw).__name__, "YAML mapping (dict)")

    # --- herd ---
    herd_raw = _require_type(raw, "herd", dict)
    herd = HerdConfig(
        n_sim=_require_int_range(herd_raw, "n_sim", 1, 100, "herd"),
        n_total=_require_int_range(herd_raw, "n_total", 2, 101, "herd"),
        physical_collar_id=_require_int_range(herd_raw, "physical_collar_id", 1, 100, "herd"),
    )
    if herd.n_total != herd.n_sim + 1:
        raise ConfigError("herd.n_total", herd.n_total, f"n_sim + 1 = {herd.n_sim + 1}")

    # --- seed ---
    seed = _require_type(raw, "seed", int)

    # --- pasture polygon ---
    poly_raw = _require(raw, "pasture_polygon")
    polygon = _parse_polygon(poly_raw, "pasture_polygon")

    # --- physiology ---
    phys_raw = _require_type(raw, "physiology", dict)
    physiology = PhysiologyConfig(
        baseline_temp_mean=_require_positive(phys_raw, "baseline_temp_mean", "physiology"),
        baseline_temp_std=_require_non_negative(phys_raw, "baseline_temp_std", "physiology"),
        diurnal_amplitude=_require_non_negative(phys_raw, "diurnal_amplitude", "physiology"),
        noise_std=_require_non_negative(phys_raw, "noise_std", "physiology"),
    )

    # --- weather ---
    wx_raw = _require_type(raw, "weather", dict)
    weather = WeatherConfig(
        ambient_temp_day=float(_require(wx_raw, "ambient_temp_day", "weather")),
        ambient_temp_night=float(_require(wx_raw, "ambient_temp_night", "weather")),
        humidity_mean=float(_require(wx_raw, "humidity_mean", "weather")),
        humidity_std=_require_non_negative(wx_raw, "humidity_std", "weather"),
    )

    # --- battery ---
    bat_raw = _require_type(raw, "battery", dict)
    battery = BatteryConfig(
        initial_level=_require_positive(bat_raw, "initial_level", "battery"),
        base_drain_per_hour=_require_positive(bat_raw, "base_drain_per_hour", "battery"),
        alert_drain_multiplier=_require_positive(bat_raw, "alert_drain_multiplier", "battery"),
    )

    # --- movement ---
    mov_raw = _require_type(raw, "movement", dict)
    movement = MovementConfig(
        centroid_speed_m_per_s=_require_non_negative(mov_raw, "centroid_speed_m_per_s", "movement"),
        individual_offset_max_m=_require_positive(mov_raw, "individual_offset_max_m", "movement"),
        flocking_strength=float(_require(mov_raw, "flocking_strength", "movement")),
    )

    # --- behaviour ---
    beh_raw = _require_type(raw, "behaviour", dict)
    trans_raw = _require_type(beh_raw, "base_transitions", dict, "behaviour")
    behaviour = BehaviourConfig(
        transition_interval_s=_require_int_range(beh_raw, "transition_interval_s", 1, 3600, "behaviour"),
        base_transitions=_parse_transitions(trans_raw, "behaviour.base_transitions"),
    )

    # --- risk ---
    risk_raw = _require_type(raw, "risk", dict)
    sev_raw = _require_type(risk_raw, "severity", dict, "risk")
    severity = SeverityConfig(
        temp_offset_low=float(_require(sev_raw, "temp_offset_low", "risk.severity")),
        temp_offset_high=float(_require(sev_raw, "temp_offset_high", "risk.severity")),
        thi_low=float(_require(sev_raw, "thi_low", "risk.severity")),
        thi_high=float(_require(sev_raw, "thi_high", "risk.severity")),
        restless=float(_require(sev_raw, "restless", "risk.severity")),
        geo_warn=float(_require(sev_raw, "geo_warn", "risk.severity")),
        geo_breach=float(_require(sev_raw, "geo_breach", "risk.severity")),
        social_isolation=float(_require(sev_raw, "social_isolation", "risk.severity")),
        collar_tamper=float(_require(sev_raw, "collar_tamper", "risk.severity")),
    )
    ab_raw = _require_type(risk_raw, "alert_bands", dict, "risk")
    alert_bands = AlertBands(
        green_max=_require_int_range(ab_raw, "green_max", 0, 100, "risk.alert_bands"),
        yellow_max=_require_int_range(ab_raw, "yellow_max", 0, 100, "risk.alert_bands"),
    )
    risk_cfg = RiskConfig(severity=severity, alert_bands=alert_bands)

    # --- thingspeak ---
    ts_raw = _require_type(raw, "thingspeak", dict)
    thingspeak = ThingSpeakConfig(
        channel_2_id=str(ts_raw.get("channel_2_id", "")),
        write_cadence_s=_require_int_range(ts_raw, "write_cadence_s", 15, 3600, "thingspeak"),
        breach_cadence_s=_require_int_range(ts_raw, "breach_cadence_s", 15, 3600, "thingspeak"),
        min_interval_s=_require_int_range(ts_raw, "min_interval_s", 15, 3600, "thingspeak"),
        channel_1_sniff_interval_s=_require_int_range(ts_raw, "channel_1_sniff_interval_s", 15, 3600, "thingspeak"),
        channel_1_stale_threshold_s=_require_int_range(ts_raw, "channel_1_stale_threshold_s", 30, 7200, "thingspeak"),
        backoff_base_s=_require_int_range(ts_raw, "backoff_base_s", 1, 60, "thingspeak"),
        backoff_max_s=_require_int_range(ts_raw, "backoff_max_s", 1, 300, "thingspeak"),
    )
    # Enforce 15s floor invariant (ADR-004)
    if thingspeak.min_interval_s < 15:
        raise ConfigError("thingspeak.min_interval_s", thingspeak.min_interval_s, ">= 15 (ThingSpeak floor)")

    # --- logging ---
    log_raw = _require_type(raw, "logging", dict)
    logging_cfg = LoggingConfig(
        log_dir=str(_require(log_raw, "log_dir", "logging")),
        ground_truth_enabled=bool(log_raw.get("ground_truth_enabled", True)),
        telemetry_csv_enabled=bool(log_raw.get("telemetry_csv_enabled", True)),
        events_jsonl_enabled=bool(log_raw.get("events_jsonl_enabled", True)),
        buffer_size=_require_int_range(log_raw, "buffer_size", 1, 10000, "logging"),
    )

    # --- hud ---
    hud_raw = _require_type(raw, "hud", dict)
    hud = HudConfig(
        host=str(_require(hud_raw, "host", "hud")),
        port=_require_int_range(hud_raw, "port", 1024, 65535, "hud"),
        poll_interval_ms=_require_int_range(hud_raw, "poll_interval_ms", 500, 30000, "hud"),
    )

    return SimulatorConfig(
        herd=herd,
        seed=seed,
        pasture_polygon=polygon,
        physiology=physiology,
        weather=weather,
        battery=battery,
        movement=movement,
        behaviour=behaviour,
        risk=risk_cfg,
        thingspeak=thingspeak,
        logging=logging_cfg,
        hud=hud,
    )


def load_env_credentials() -> dict[str, str]:
    """Load ThingSpeak API keys from environment variables.

    Returns a dict of credential keys. Missing keys are empty strings
    (valid for dry-run/offline modes). Never raises — callers check
    at the point of use.
    """
    keys = [
        "THINGSPEAK_WRITE_API_KEY",
        "THINGSPEAK_READ_API_KEY",
        "THINGSPEAK_CONFIG_READ_API_KEY",
    ]
    creds: dict[str, str] = {}
    for k in keys:
        val = os.environ.get(k, "")
        # Paranoia: strip whitespace that might have been pasted
        creds[k] = val.strip()
    return creds
