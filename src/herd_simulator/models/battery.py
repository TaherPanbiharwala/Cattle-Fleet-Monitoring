"""
battery.py — Activity-aware battery drainage (ADR-009: no solar recharge).

Battery is strictly monotonically decreasing — there is no recharge path
anywhere in this module. Drain is computed per *simulated* second from a
configured hourly rate, so dry-run (fast-forward, no sleeping) and
real-time execution produce the identical trajectory for the same number
of simulated seconds (Master PRD: "Dry-run and real-time modes produce
the same trajectory").

References:
  ADR-009: Activity-Aware Battery Drainage (Strictly No Solar Recharge)
  Master PRD: "Battery"
  HerdSimulator PRD §6.4 (FR-30), §9.3
"""

from __future__ import annotations

from dataclasses import dataclass

from herd_simulator.config import BatteryConfig

DROPOUT_THRESHOLD_PCT = 0.0


@dataclass
class BatteryState:
    level_pct: float
    dropped_out: bool = False


def new_battery_state(cfg: BatteryConfig) -> BatteryState:
    """Initial battery state at the start of a run (always the configured
    `initial_level`, normally 100%)."""
    return BatteryState(
        level_pct=cfg.initial_level,
        dropped_out=cfg.initial_level <= DROPOUT_THRESHOLD_PCT,
    )


def drain_for_elapsed(cfg: BatteryConfig, elapsed_s: float, alert_active: bool) -> float:
    """Battery percentage consumed over `elapsed_s` simulated seconds
    (always >= 0 — this is drain, never recharge).

    Rate is `base_drain_per_hour`, multiplied by `alert_drain_multiplier`
    while `alert_active` is True (breach/warning/active-anomaly 15s-burst
    mode, ADR-009).
    """
    if elapsed_s < 0:
        raise ValueError(f"elapsed_s must be >= 0, got {elapsed_s}")
    multiplier = cfg.alert_drain_multiplier if alert_active else 1.0
    rate_per_s = (cfg.base_drain_per_hour * multiplier) / 3600.0
    return rate_per_s * elapsed_s


def step(state: BatteryState, cfg: BatteryConfig, elapsed_s: float, alert_active: bool) -> BatteryState:
    """Advance the battery by one step and return the new state.

    Never recharges; clamps at exactly 0.0 and latches `dropped_out`
    permanently once triggered — a dead collar stays dead for the rest of
    the run (ADR-009: "When battery reaches 0%, the device triggers
    collar_dropout and ceases transmission").
    """
    if state.dropped_out:
        return state

    drained = drain_for_elapsed(cfg, elapsed_s, alert_active)
    new_level = max(0.0, state.level_pct - drained)
    return BatteryState(level_pct=new_level, dropped_out=new_level <= DROPOUT_THRESHOLD_PCT)
