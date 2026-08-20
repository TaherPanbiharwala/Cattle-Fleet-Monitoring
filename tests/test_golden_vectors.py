"""
test_golden_vectors.py — Golden vector parity tests for math utilities.

These tests define the canonical input→output pairs that BOTH the Python
simulator AND the future C++ ESP32 firmware must pass identically.
They are the cross-language parity contract (PRD §9.5, ADR-007).

Each test uses hand-verified arithmetic so failures pinpoint formula bugs,
not test bugs.
"""

from __future__ import annotations

import math
import sys
import os

import pytest

# Add src/ to path so we can import herd_simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.utils.geo import (
    haversine_m,
    point_in_polygon,
    classify_geofence,
    min_distance_to_polygon_m,
    compute_thi,
    GEOFENCE_INSIDE,
    GEOFENCE_WARNING,
    GEOFENCE_BREACH,
)
from herd_simulator.utils.risk import (
    RiskInputs,
    compute_risk_score,
    compute_severity_components,
    classify_alert,
    clamp,
)
from herd_simulator.config import SeverityConfig


# ===================================================================
# Shared fixtures
# ===================================================================

# Default severity config matching ADR-007 / default_config.yaml
DEFAULT_SEVERITY = SeverityConfig(
    temp_offset_low=0.5,
    temp_offset_high=2.0,
    thi_low=68,
    thi_high=84,
    restless=0.35,
    geo_warn=0.50,
    geo_breach=0.90,
    social_isolation=0.70,
    collar_tamper=0.90,
)

# Simple rectangular pasture polygon (default from config)
PASTURE_RECT = [
    (12.9720, 79.1585),
    (12.9720, 79.1605),
    (12.9705, 79.1605),
    (12.9705, 79.1585),
]


# ===================================================================
# 1. THI Formula Golden Vectors
# ===================================================================

class TestTHI:
    """THI = (1.8*T + 32) - (0.55 - 0.0055*RH) * (1.8*T - 26)"""

    def test_thi_standard_conditions(self):
        """T=30°C, RH=60% → hand-calculated THI."""
        # (1.8*30 + 32) = 86
        # (0.55 - 0.0055*60) = 0.55 - 0.33 = 0.22
        # (1.8*30 - 26) = 28
        # THI = 86 - 0.22 * 28 = 86 - 6.16 = 79.84
        result = compute_thi(30.0, 60.0)
        assert abs(result - 79.84) < 0.01, f"Expected 79.84, got {result}"

    def test_thi_hot_humid(self):
        """T=40°C, RH=80% → extreme heat stress."""
        # (1.8*40 + 32) = 104
        # (0.55 - 0.0055*80) = 0.55 - 0.44 = 0.11
        # (1.8*40 - 26) = 46
        # THI = 104 - 0.11 * 46 = 104 - 5.06 = 98.94
        result = compute_thi(40.0, 80.0)
        assert abs(result - 98.94) < 0.01, f"Expected 98.94, got {result}"

    def test_thi_cool_dry(self):
        """T=15°C, RH=30% → no heat stress."""
        # (1.8*15 + 32) = 59
        # (0.55 - 0.0055*30) = 0.55 - 0.165 = 0.385
        # (1.8*15 - 26) = 1
        # THI = 59 - 0.385 * 1 = 58.615
        result = compute_thi(15.0, 30.0)
        assert abs(result - 58.615) < 0.01, f"Expected 58.615, got {result}"

    def test_thi_zero_humidity(self):
        """T=25°C, RH=0% → dry conditions."""
        # (1.8*25 + 32) = 77
        # (0.55 - 0.0055*0) = 0.55
        # (1.8*25 - 26) = 19
        # THI = 77 - 0.55 * 19 = 77 - 10.45 = 66.55
        result = compute_thi(25.0, 0.0)
        assert abs(result - 66.55) < 0.01, f"Expected 66.55, got {result}"


# ===================================================================
# 2. Haversine Distance Golden Vectors
# ===================================================================

class TestHaversine:
    """Haversine formula with Earth radius = 6,371,000 m."""

    def test_same_point(self):
        """Distance between a point and itself is 0."""
        d = haversine_m((12.9716, 79.1589), (12.9716, 79.1589))
        assert d == 0.0

    def test_known_short_distance(self):
        """Two points ~166m apart (hand-verified with online calculator)."""
        # VIT Vellore campus area: ~0.0015 degrees latitude ≈ 166.7m
        p1 = (12.9700, 79.1589)
        p2 = (12.9715, 79.1589)
        d = haversine_m(p1, p2)
        assert abs(d - 166.7) < 1.0, f"Expected ~166.7m, got {d:.1f}m"

    def test_known_longitude_distance(self):
        """Two points separated only by longitude at lat ~13°N."""
        # At lat 13°, 0.001° longitude ≈ 108.5m (cos(13°) * 111320 * 0.001)
        p1 = (13.0, 79.1580)
        p2 = (13.0, 79.1590)
        d = haversine_m(p1, p2)
        expected = math.cos(math.radians(13.0)) * 111_320 * 0.001
        assert abs(d - expected) < 1.0, f"Expected ~{expected:.1f}m, got {d:.1f}m"

    def test_symmetry(self):
        """haversine(A, B) == haversine(B, A)."""
        p1 = (12.9720, 79.1585)
        p2 = (12.9705, 79.1605)
        assert abs(haversine_m(p1, p2) - haversine_m(p2, p1)) < 1e-9


# ===================================================================
# 3. Ray-Casting Point-in-Polygon Golden Vectors
# ===================================================================

class TestPointInPolygon:
    """Ray-casting algorithm tests on the default rectangular pasture."""

    def test_center_is_inside(self):
        """Center of the rectangle is inside."""
        center = (12.97125, 79.1595)
        assert point_in_polygon(center, PASTURE_RECT) is True

    def test_outside_north(self):
        """Point north of the rectangle is outside."""
        north = (12.9730, 79.1595)
        assert point_in_polygon(north, PASTURE_RECT) is False

    def test_outside_south(self):
        """Point south of the rectangle is outside."""
        south = (12.9695, 79.1595)
        assert point_in_polygon(south, PASTURE_RECT) is False

    def test_outside_east(self):
        """Point east of the rectangle is outside."""
        east = (12.97125, 79.1615)
        assert point_in_polygon(east, PASTURE_RECT) is False

    def test_outside_west(self):
        """Point west of the rectangle is outside."""
        west = (12.97125, 79.1575)
        assert point_in_polygon(west, PASTURE_RECT) is False

    def test_triangle(self):
        """Point-in-polygon works for a triangle."""
        triangle = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0)]
        assert point_in_polygon((0.1, 0.1), triangle) is True
        assert point_in_polygon((0.8, 0.8), triangle) is False


# ===================================================================
# 4. Geofence Classification Golden Vectors
# ===================================================================

class TestGeofenceClassification:
    """Geofence status: Inside (0), Warning (1), Breach (2)."""

    def test_center_is_inside(self):
        """Center of a large-enough polygon is INSIDE (not warning)."""
        center = (12.97125, 79.1595)
        status = classify_geofence(center, PASTURE_RECT)
        assert status == GEOFENCE_INSIDE

    def test_outside_is_breach(self):
        """Point outside polygon is BREACH."""
        outside = (12.9730, 79.1595)
        status = classify_geofence(outside, PASTURE_RECT)
        assert status == GEOFENCE_BREACH

    def test_near_edge_is_warning(self):
        """Point inside but within 10m of edge is WARNING.

        The north edge of our rectangle is at lat=12.9720.
        Moving ~5m south: delta_lat ≈ 5/111320 ≈ 0.0000449
        So lat = 12.9720 - 0.0000449 ≈ 12.9719551, lon near center.
        """
        near_north_edge = (12.9720 - 5.0 / 111320.0, 79.1595)
        status = classify_geofence(near_north_edge, PASTURE_RECT)
        assert status == GEOFENCE_WARNING

    def test_just_outside_warning_band(self):
        """Point inside but >10m from all edges is INSIDE.

        Move 15m south from north edge.
        """
        safely_inside = (12.9720 - 15.0 / 111320.0, 79.1595)
        status = classify_geofence(safely_inside, PASTURE_RECT)
        assert status == GEOFENCE_INSIDE


# ===================================================================
# 5. Risk Score Golden Vectors
# ===================================================================

class TestRiskScore:
    """Composite risk = round(100 * (1 - ∏(1 - S_i)))."""

    def test_all_normal_is_zero(self):
        """Normal animal: no fever, no THI stress, inside geofence → risk = 0."""
        inputs = RiskInputs(
            body_temp=38.6,
            baseline_temp=38.6,
            thi=65.0,
            is_restless=False,
            geofence_status=0,
            is_isolated=False,
            is_tampered=False,
        )
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 0, f"Expected 0, got {score}"

    def test_geofence_breach_alone(self):
        """Only geofence breach active → S_breach=0.90 → risk=90."""
        inputs = RiskInputs(geofence_status=2)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        # 100 * (1 - (1-0.90)) = 100 * 0.90 = 90
        assert score == 90, f"Expected 90, got {score}"

    def test_collar_tamper_alone(self):
        """Only collar tamper active → S_tamper=0.90 → risk=90."""
        inputs = RiskInputs(is_tampered=True)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 90, f"Expected 90, got {score}"

    def test_isolation_alone(self):
        """Only social isolation → S_isol=0.70 → risk=70."""
        inputs = RiskInputs(is_isolated=True)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 70, f"Expected 70, got {score}"

    def test_restless_alone(self):
        """Only restless behaviour → S_restless=0.35 → risk=35."""
        inputs = RiskInputs(is_restless=True)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 35, f"Expected 35, got {score}"

    def test_geofence_warning_alone(self):
        """Only geofence warning → S_warn=0.50 → risk=50."""
        inputs = RiskInputs(geofence_status=1)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 50, f"Expected 50, got {score}"

    def test_fever_mild(self):
        """Mild fever: baseline=38.6, body=39.6 → offset=1.0.
        S_temp = clamp((1.0 - 0.5) / 1.5, 0, 1) = 0.5/1.5 = 0.3333
        risk = round(100 * (1 - (1-0.3333))) = round(33.33) = 33
        """
        inputs = RiskInputs(body_temp=39.6, baseline_temp=38.6)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 33, f"Expected 33, got {score}"

    def test_fever_high(self):
        """High fever: baseline=38.6, body=41.1 → offset=2.5.
        S_temp = clamp((2.5 - 0.5) / 1.5, 0, 1) = 2.0/1.5 = 1.333 → clamped to 1.0
        risk = round(100 * (1 - (1-1.0))) = 100
        """
        inputs = RiskInputs(body_temp=41.1, baseline_temp=38.6)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 100, f"Expected 100, got {score}"

    def test_combined_breach_and_tamper(self):
        """Geofence breach (0.90) + tamper (0.90).
        risk = round(100 * (1 - (1-0.90)*(1-0.90)))
             = round(100 * (1 - 0.10*0.10))
             = round(100 * 0.99) = 99
        """
        inputs = RiskInputs(geofence_status=2, is_tampered=True)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 99, f"Expected 99, got {score}"

    def test_combined_restless_and_isolation(self):
        """Restless (0.35) + isolation (0.70).
        risk = round(100 * (1 - (1-0.35)*(1-0.70)))
             = round(100 * (1 - 0.65*0.30))
             = round(100 * (1 - 0.195))
             = round(80.5) = 80  (Python banker's rounding → 80)
        """
        inputs = RiskInputs(is_restless=True, is_isolated=True)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 80 or score == 81, f"Expected 80 or 81, got {score}"

    def test_thi_stress_moderate(self):
        """THI=76 → S_THI = (76-68)/16 = 0.50 → risk=50."""
        inputs = RiskInputs(thi=76.0)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 50, f"Expected 50, got {score}"

    def test_thi_below_threshold(self):
        """THI=60 → S_THI = clamp((60-68)/16, 0, 1) = 0 → contributes nothing."""
        inputs = RiskInputs(thi=60.0)
        score = compute_risk_score(inputs, DEFAULT_SEVERITY)
        assert score == 0, f"Expected 0, got {score}"


# ===================================================================
# 6. Alert Band Classification
# ===================================================================

class TestAlertBands:
    """Green 0–39, Yellow 40–69, Red 70–100."""

    def test_green_boundary(self):
        assert classify_alert(0) == "green"
        assert classify_alert(39) == "green"

    def test_yellow_boundary(self):
        assert classify_alert(40) == "yellow"
        assert classify_alert(69) == "yellow"

    def test_red_boundary(self):
        assert classify_alert(70) == "red"
        assert classify_alert(100) == "red"


# ===================================================================
# 7. Clamp Utility
# ===================================================================

class TestClamp:
    def test_within_range(self):
        assert clamp(0.5, 0, 1) == 0.5

    def test_below_range(self):
        assert clamp(-1.0, 0, 1) == 0.0

    def test_above_range(self):
        assert clamp(2.0, 0, 1) == 1.0

    def test_at_boundaries(self):
        assert clamp(0.0, 0, 1) == 0.0
        assert clamp(1.0, 0, 1) == 1.0
