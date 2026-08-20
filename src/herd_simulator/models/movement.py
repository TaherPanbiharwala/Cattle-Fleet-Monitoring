"""
movement.py — Herd centroid drift and per-animal position modeling.

The herd centroid performs a bounded random walk inside the pasture
polygon during normal operation, or smoothly steps toward a live Collar-1
GPS fix when the caller supplies one (ADR-010). Each animal's own position
is the centroid plus a bounded, behaviour-modulated individual offset
(herd cohesion / flocking, FR-8).

This module only computes *where things move to*; the caller (the future
engine/simulator.py tick loop) owns *why* — e.g. whether an isolation or
breach excursion is currently scripted for a given animal, and whether a
fresh Collar-1 fix is available to anchor to. Keeping that decision out of
this module is what makes `move_toward` reusable for both cases below.

References:
  Master PRD: "Movement"
  HerdSimulator PRD §6.3 (FR-7..FR-9, FR-29), §9.2
  ADR-008: Geofence Spatial Geometry & 10-Meter Warning Band
  ADR-010: Collar-1 Sniffing & Synchronization
"""

from __future__ import annotations

import math
import random

from herd_simulator.config import MovementConfig
from herd_simulator.models.behaviour import Behaviour
from herd_simulator.utils.geo import Coord, haversine_m, point_in_polygon

_EARTH_RADIUS_M = 6_371_000.0

# GPS jitter applied even to a stationary (Resting) animal — a real GPS fix
# never reports the exact same coordinate twice (Master PRD point 3:
# "Resting cattle remain stationary except for GPS noise").
GPS_NOISE_STD_M = 0.5

# Representative per-behaviour ground speed (m/s), used for every state
# except Walking (which draws from the animal's own profile range instead,
# so no two animals move in lockstep). Strictly ordered
# Resting < Ruminating < Grazing < Restless < Walking, satisfying Master
# PRD point 4: "Walking cattle move faster than grazing cattle."
_BEHAVIOUR_SPEED_MPS: dict[Behaviour, float] = {
    Behaviour.RESTING: 0.0,
    Behaviour.RUMINATING: 0.01,
    Behaviour.GRAZING: 0.08,
    Behaviour.RESTLESS: 0.25,
}

# Isolation excursion (Master PRD point 5, FR-9 "straggler offset"): extra
# meters of centroid distance accrued per second of continuous active
# isolation, capped so a straggler can't wander off indefinitely.
ISOLATION_DRIFT_M_PER_S = 0.15
ISOLATION_MAX_EXTRA_M = 120.0

_DEFAULT_WALKING_SPEED_RANGE_MPS = (0.8, 1.4)


def offset_to_latlon(origin: Coord, north_m: float, east_m: float) -> Coord:
    """Convert a local (north, east) meter offset from `origin` into a
    (lat, lon) coordinate, via an equirectangular approximation — accurate
    at pasture scale (hundreds of meters), consistent with the planar
    approximation already used in utils/geo.py."""
    lat0, lon0 = origin
    dlat = north_m / _EARTH_RADIUS_M
    dlon = east_m / (_EARTH_RADIUS_M * math.cos(math.radians(lat0)))
    return (lat0 + math.degrees(dlat), lon0 + math.degrees(dlon))


def move_toward(current: Coord, target: Coord, max_step_m: float) -> Coord:
    """Bounded step from `current` toward `target`, capped at `max_step_m`.

    Never overshoots and never teleports (Master PRD point 7): if `target`
    is within `max_step_m`, returns `target` exactly; otherwise returns a
    point exactly `max_step_m` closer, along the straight line toward it.
    This single primitive backs both Collar-1 centroid anchoring (ADR-010)
    and scripted breach excursions (Master PRD point 6).
    """
    remaining_m = haversine_m(current, target)
    if remaining_m <= max_step_m or remaining_m == 0.0:
        return target
    frac = max_step_m / remaining_m
    lat = current[0] + (target[0] - current[0]) * frac
    lon = current[1] + (target[1] - current[1]) * frac
    return (lat, lon)


def step_centroid_autonomous(
    centroid: Coord,
    cfg: MovementConfig,
    polygon: list[Coord],
    elapsed_s: float,
    rng: random.Random,
) -> Coord:
    """Advance the centroid one step via a bounded random walk, staying
    inside the pasture polygon (Master PRD point 1). A candidate step that
    would exit the polygon is rejected for this tick (the centroid holds
    position and tries a fresh random bearing next tick) rather than
    clamped or reflected — simple, and never violates containment."""
    bearing = rng.uniform(0.0, 2 * math.pi)
    step_m = cfg.centroid_speed_m_per_s * elapsed_s
    north = step_m * math.cos(bearing)
    east = step_m * math.sin(bearing)
    candidate = offset_to_latlon(centroid, north, east)

    if point_in_polygon(candidate, polygon):
        return candidate
    return centroid


def step_centroid_anchored(
    centroid: Coord,
    anchor: Coord,
    cfg: MovementConfig,
    elapsed_s: float,
) -> Coord:
    """Smoothly move the centroid toward a live Collar-1 fix (ADR-010,
    FR-29), capped at the configured drift speed so the herd flocks to the
    real collar rather than snapping onto it."""
    return move_toward(centroid, anchor, cfg.centroid_speed_m_per_s * elapsed_s)


def _behaviour_speed_mps(
    behaviour: Behaviour,
    walking_speed_range_mps: tuple[float, float],
    rng: random.Random,
) -> float:
    if behaviour == Behaviour.WALKING:
        return rng.uniform(*walking_speed_range_mps)
    return _BEHAVIOUR_SPEED_MPS[behaviour]


def individual_offset_m(
    behaviour: Behaviour,
    preferred_bearing_rad: float,
    preferred_dist_m: float,
    cfg: MovementConfig,
    rng: random.Random,
    elapsed_s: float = 1.0,
    walking_speed_range_mps: tuple[float, float] = _DEFAULT_WALKING_SPEED_RANGE_MPS,
    isolation_extra_m: float = 0.0,
) -> tuple[float, float]:
    """Return (north_m, east_m) offset from the herd centroid for one
    animal this tick: its preferred spot in the herd, pulled toward the
    centroid by `flocking_strength` (0 = independent, 1 = glued — matches
    the config comment), jittered by behaviour-scaled step size, and
    optionally pushed further out by an active isolation excursion
    (Master PRD point 5, FR-9).

    A Resting animal (speed 0) gets zero jitter here by construction — its
    offset is a fixed point relative to the centroid, matching point 3
    ("remain stationary except for GPS noise", added separately by
    `animal_position`).
    """
    cohesion_pull = 1.0 - _clamp(cfg.flocking_strength, 0.0, 1.0)
    base_dist = min(preferred_dist_m * cohesion_pull, cfg.individual_offset_max_m)
    total_dist = min(
        base_dist + isolation_extra_m,
        cfg.individual_offset_max_m + ISOLATION_MAX_EXTRA_M,
    )

    speed_mps = _behaviour_speed_mps(behaviour, walking_speed_range_mps, rng)
    step_m = speed_mps * elapsed_s

    bearing_wander = step_m / max(cfg.individual_offset_max_m, 1e-9)
    bearing = preferred_bearing_rad + rng.uniform(-math.pi, math.pi) * bearing_wander
    dist = max(0.0, total_dist + rng.uniform(-step_m, step_m))

    return (dist * math.cos(bearing), dist * math.sin(bearing))


def animal_position(
    centroid: Coord,
    behaviour: Behaviour,
    preferred_bearing_rad: float,
    preferred_dist_m: float,
    cfg: MovementConfig,
    rng: random.Random,
    elapsed_s: float = 1.0,
    walking_speed_range_mps: tuple[float, float] = _DEFAULT_WALKING_SPEED_RANGE_MPS,
    isolation_extra_m: float = 0.0,
) -> Coord:
    """Full per-animal position: centroid + individual offset, plus GPS
    noise applied to every behaviour including Resting (Master PRD
    point 3)."""
    north, east = individual_offset_m(
        behaviour,
        preferred_bearing_rad,
        preferred_dist_m,
        cfg,
        rng,
        elapsed_s,
        walking_speed_range_mps,
        isolation_extra_m,
    )
    north += rng.gauss(0.0, GPS_NOISE_STD_M)
    east += rng.gauss(0.0, GPS_NOISE_STD_M)
    return offset_to_latlon(centroid, north, east)


def isolation_extra_distance_m(elapsed_s: float, rate_m_per_s: float = ISOLATION_DRIFT_M_PER_S) -> float:
    """Extra centroid distance (meters) accrued by a straggler after
    `elapsed_s` seconds of continuous active isolation (Master PRD
    point 5: "Isolation progressively increases centroid distance")."""
    return min(rate_m_per_s * max(elapsed_s, 0.0), ISOLATION_MAX_EXTRA_M)


_BREACH_SEARCH_STEP_M = 5.0
_BREACH_SEARCH_MAX_STEPS = 200  # 200 * 5m = 1km search radius — well beyond any pasture-scale polygon


def breach_excursion_target(position: Coord, polygon: list[Coord], outward_m: float = 20.0) -> Coord:
    """A point `outward_m` meters beyond the polygon boundary, walking
    outward from `position` along the ray from the polygon's centroid
    through `position` until the boundary is actually crossed.

    Intended to be used as the `target` argument to `move_toward` so a
    scripted breach walks the animal continuously through the warning band
    and across the boundary (Master PRD point 6: "Breach creates a
    continuous path through warning and breach zones") rather than
    teleporting it outside. A fixed step count bounds the search — this is
    a target-finder, not a per-tick mover, so it does not need to respect
    `centroid_speed_m_per_s`.
    """
    centroid_lat = sum(p[0] for p in polygon) / len(polygon)
    centroid_lon = sum(p[1] for p in polygon) / len(polygon)
    bearing = math.atan2(position[1] - centroid_lon, position[0] - centroid_lat)

    point = position
    for _ in range(_BREACH_SEARCH_MAX_STEPS):
        if not point_in_polygon(point, polygon):
            break
        point = offset_to_latlon(
            point,
            _BREACH_SEARCH_STEP_M * math.cos(bearing),
            _BREACH_SEARCH_STEP_M * math.sin(bearing),
        )

    return offset_to_latlon(point, outward_m * math.cos(bearing), outward_m * math.sin(bearing))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
