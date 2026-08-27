"""
risk.py — Deterministic product-rule composite risk scoring engine.

Implements the exact risk formula from ADR-007 so that given identical
inputs, the Python simulator and future C++ firmware produce identical
scores (verified via golden test vectors).

Formula:
    Compute severity components S_i in [0, 1]:
        S_temp    = clamp((body_temp - baseline - 0.5) / 1.5, 0, 1)
        S_THI     = clamp((THI - 68) / 16, 0, 1)
        S_restless = 0.35  (if behaviour == Restless)
        S_geo_warn = 0.50  (if geofence == Warning)
        S_geo_breach = 0.90 (if geofence == Breach)
        S_isol    = 0.70  (if socially isolated)
        S_tamper  = 0.90  (if collar tampered)

    Composite:
        risk = round(100 * (1 - product(1 - S_i))) ∈ [0, 100]

Alert bands:
    0–39  = Green  (Normal)
    40–69 = Yellow (Warning)
    70–100 = Red   (Critical Alert)

References:
    ADR-007: Deterministic Product-Rule Risk Scoring Formula
    AGENTS.md §4.2: Mathematical Formulas
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from herd_simulator.config import SeverityConfig


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a value to [lo, hi]."""
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@dataclass
class RiskInputs:
    """All inputs required to compute a single animal's composite risk score.

    Each field maps to one severity component in the product-rule formula.
    Boolean flags are False by default (no contribution to risk).
    """

    body_temp: float = 38.6       # Current body temperature (°C)
    baseline_temp: float = 38.6   # This animal's baseline body temp (°C)
    thi: float = 65.0             # Current Temperature-Humidity Index
    is_restless: bool = False     # Behaviour state == Restless (code 4)
    geofence_status: int = 0      # 0=Inside, 1=Warning, 2=Breach
    is_isolated: bool = False     # Social isolation flag
    is_tampered: bool = False     # Collar tamper flag


def compute_severity_components(
    inputs: RiskInputs,
    cfg: SeverityConfig,
) -> dict[str, float]:
    """Compute each severity component S_i in [0, 1].

    Returns a dict mapping component name to its severity value.
    Components that are inactive (e.g., geofence=Inside) contribute 0.0.
    """
    components: dict[str, float] = {}

    # S_temp: temperature deviation from baseline
    temp_offset = inputs.body_temp - inputs.baseline_temp
    components["temp"] = clamp(
        (temp_offset - cfg.temp_offset_low) / (cfg.temp_offset_high - cfg.temp_offset_low),
        0.0, 1.0,
    )

    # S_THI: heat stress index
    components["thi"] = clamp(
        (inputs.thi - cfg.thi_low) / (cfg.thi_high - cfg.thi_low),
        0.0, 1.0,
    )

    # S_restless: behaviour = Restless
    components["restless"] = cfg.restless if inputs.is_restless else 0.0

    # S_geo: geofence status (mutually exclusive: warning OR breach, not both)
    if inputs.geofence_status == 2:
        components["geofence"] = cfg.geo_breach
    elif inputs.geofence_status == 1:
        components["geofence"] = cfg.geo_warn
    else:
        components["geofence"] = 0.0

    # S_isol: social isolation
    components["isolation"] = cfg.social_isolation if inputs.is_isolated else 0.0

    # S_tamper: collar tamper
    components["tamper"] = cfg.collar_tamper if inputs.is_tampered else 0.0

    return components


def compute_risk_score(
    inputs: RiskInputs,
    cfg: SeverityConfig,
) -> int:
    """Compute the composite risk score ∈ [0, 100] using the product rule.

    Formula: risk = round(100 * (1 - ∏(1 - S_i)))

    This is the canonical implementation. The C++ firmware must produce
    identical output for identical inputs (verified by golden vectors).
    """
    components = compute_severity_components(inputs, cfg)

    # Product rule: 1 - product(1 - S_i)
    product = 1.0
    for s in components.values():
        product *= (1.0 - s)

    risk = 100.0 * (1.0 - product)
    return round(risk)


def classify_alert(risk_score: int, green_max: int = 39, yellow_max: int = 69) -> str:
    """Classify a risk score into an alert band.

    Returns:
        'green'  for 0–green_max
        'yellow' for green_max+1–yellow_max
        'red'    for yellow_max+1–100
    """
    if risk_score <= green_max:
        return "green"
    elif risk_score <= yellow_max:
        return "yellow"
    else:
        return "red"
