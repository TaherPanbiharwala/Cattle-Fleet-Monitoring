"""
geo.py — Geospatial utility functions (standard library only).

Implements:
  - Haversine distance (meters) between two lat/lon points
  - Ray-casting point-in-polygon test
  - Geofence status classification (Inside / Warning / Breach)
  - Minimum distance from a point to a polygon edge (for the 10m warning band)

References:
  - ADR-008: Geofence spatial geometry & 10-meter warning band
  - AGENTS.md §4.2: Mathematical formulas
  - PRD §9.5: Cross-language parity (golden vectors)
"""

from __future__ import annotations

import math
from typing import Sequence

# Earth radius in meters (WGS-84 mean)
_EARTH_RADIUS_M = 6_371_000.0

# Coordinate type alias
Coord = tuple[float, float]  # (latitude_deg, longitude_deg)


def haversine_m(p1: Coord, p2: Coord) -> float:
    """Return the great-circle distance in **meters** between two (lat, lon) points.

    Uses the standard Haversine formula. Inputs are decimal degrees.
    """
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return _EARTH_RADIUS_M * c


def point_in_polygon(point: Coord, polygon: Sequence[Coord]) -> bool:
    """Ray-casting algorithm for point-in-polygon test.

    Args:
        point: (lat, lon) to test.
        polygon: Ordered list of (lat, lon) vertices defining a closed polygon.
                 The polygon is implicitly closed (last vertex connects to first).

    Returns:
        True if the point is strictly inside the polygon.
    """
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the ray from (x, y) heading in +y direction crosses edge (i, j)
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i

    return inside


def _point_to_segment_distance_m(
    point: Coord,
    seg_start: Coord,
    seg_end: Coord,
) -> float:
    """Minimum distance (meters) from a point to a line segment on the Earth's surface.

    Uses a planar approximation (valid for segments < ~1 km at non-polar latitudes).
    Projects the point onto the segment in a local tangent plane, then converts
    the result to meters via Haversine for accuracy.
    """
    # Convert to local tangent plane (meters) relative to seg_start
    lat0, lon0 = seg_start
    cos_lat = math.cos(math.radians(lat0))

    # Approximate meters per degree at this latitude
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * cos_lat

    # Vector from seg_start to point
    px = (point[0] - lat0) * m_per_deg_lat
    py = (point[1] - lon0) * m_per_deg_lon

    # Vector from seg_start to seg_end
    sx = (seg_end[0] - lat0) * m_per_deg_lat
    sy = (seg_end[1] - lon0) * m_per_deg_lon

    seg_len_sq = sx * sx + sy * sy

    if seg_len_sq < 1e-12:
        # Degenerate segment (two identical points)
        return math.sqrt(px * px + py * py)

    # Project point onto the segment, clamped to [0, 1]
    t = max(0.0, min(1.0, (px * sx + py * sy) / seg_len_sq))

    # Closest point on segment
    cx = t * sx
    cy = t * sy

    dx = px - cx
    dy = py - cy

    return math.sqrt(dx * dx + dy * dy)


def min_distance_to_polygon_m(point: Coord, polygon: Sequence[Coord]) -> float:
    """Minimum distance (meters) from a point to any edge of the polygon.

    Used for the 10m warning band classification (ADR-008).
    """
    n = len(polygon)
    min_dist = float("inf")

    for i in range(n):
        j = (i + 1) % n
        d = _point_to_segment_distance_m(point, polygon[i], polygon[j])
        min_dist = min(min_dist, d)

    return min_dist


# --- Geofence status constants (ADR-005 / ADR-008) ---
GEOFENCE_INSIDE = 0
GEOFENCE_WARNING = 1
GEOFENCE_BREACH = 2

# Warning band width in meters
_WARNING_BAND_M = 10.0


def classify_geofence(
    point: Coord,
    polygon: Sequence[Coord],
    warning_band_m: float = _WARNING_BAND_M,
) -> int:
    """Classify a point's geofence status relative to a pasture polygon.

    Returns:
        0 = Inside (more than warning_band_m from any edge)
        1 = Warning band (inside but within warning_band_m of an edge)
        2 = Breach (outside the polygon)

    References:
        ADR-008: Geofence Spatial Geometry & 10-Meter Warning Band
    """
    if not point_in_polygon(point, polygon):
        return GEOFENCE_BREACH

    dist = min_distance_to_polygon_m(point, polygon)
    if dist <= warning_band_m:
        return GEOFENCE_WARNING

    return GEOFENCE_INSIDE


def compute_thi(ambient_temp_c: float, relative_humidity: float) -> float:
    """Compute the Temperature-Humidity Index (THI).

    Formula (from ADR-007 / AGENTS.md §4.2):
        THI = (1.8 * T + 32) - (0.55 - 0.0055 * RH) * (1.8 * T - 26)

    Args:
        ambient_temp_c: Ambient temperature in °C.
        relative_humidity: Relative humidity in % (0–100).

    Returns:
        THI value (dimensionless index).
    """
    t = ambient_temp_c
    rh = relative_humidity
    return (1.8 * t + 32) - (0.55 - 0.0055 * rh) * (1.8 * t - 26)
