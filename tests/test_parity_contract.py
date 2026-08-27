"""Regression checks for the tracked Python/ESP32 telemetry parity contract."""

from __future__ import annotations

import json
from pathlib import Path

from herd_simulator.config import SeverityConfig, load_config
from herd_simulator.utils.geo import classify_geofence, compute_thi
from herd_simulator.utils.risk import RiskInputs, classify_alert, compute_risk_score


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads(
    (ROOT / "contracts" / "telemetry_parity_v1.json").read_text(encoding="utf-8")
)


def _severity_from_contract() -> SeverityConfig:
    risk = CONTRACT["risk"]
    return SeverityConfig(
        temp_offset_low=risk["temp_offset_low"],
        temp_offset_high=risk["temp_offset_high"],
        thi_low=risk["thi_low"],
        thi_high=risk["thi_high"],
        restless=risk["restless"],
        geo_warn=risk["geo_warn"],
        geo_breach=risk["geo_breach"],
        social_isolation=risk["social_isolation"],
        collar_tamper=risk["collar_tamper"],
    )


def test_default_simulator_config_matches_parity_contract() -> None:
    cfg = load_config(ROOT / "config" / "default_config.yaml")
    risk = CONTRACT["risk"]
    assert cfg.herd.physical_collar_id == CONTRACT["telemetry"]["physical_collar_id"]
    assert list(cfg.pasture_polygon) == [tuple(point) for point in CONTRACT["geofence"]["polygon"]]
    assert cfg.risk.severity.temp_offset_low == risk["temp_offset_low"]
    assert cfg.risk.severity.temp_offset_high == risk["temp_offset_high"]
    assert cfg.risk.severity.thi_low == risk["thi_low"]
    assert cfg.risk.severity.thi_high == risk["thi_high"]
    assert cfg.risk.alert_bands.green_max == risk["green_max"]
    assert cfg.risk.alert_bands.yellow_max == risk["yellow_max"]


def test_python_thi_vectors_match_parity_contract() -> None:
    for vector in CONTRACT["thi_vectors"]:
        assert abs(compute_thi(vector["ambient_temp_c"], vector["humidity_pct"]) - vector["expected_thi"]) < 0.001


def test_python_geofence_vectors_match_parity_contract() -> None:
    polygon = [tuple(point) for point in CONTRACT["geofence"]["polygon"]]
    for vector in CONTRACT["geofence"]["vectors"]:
        assert classify_geofence((vector["latitude"], vector["longitude"]), polygon) == vector["expected_status"]


def test_python_risk_vectors_and_alert_bands_match_parity_contract() -> None:
    risk = CONTRACT["risk"]
    severity = _severity_from_contract()
    for vector in risk["vectors"]:
        score = compute_risk_score(
            RiskInputs(
                body_temp=vector["body_temp_c"],
                baseline_temp=risk["baseline_temp_c"],
                thi=vector["thi"],
                is_restless=vector["restless"],
                geofence_status=vector["geofence_status"],
                is_isolated=vector["isolated"],
                is_tampered=vector["tampered"],
            ),
            severity,
        )
        assert score == vector["expected_score"]
    assert classify_alert(risk["green_max"]) == "green"
    assert classify_alert(risk["green_max"] + 1) == "yellow"
    assert classify_alert(risk["yellow_max"] + 1) == "red"
