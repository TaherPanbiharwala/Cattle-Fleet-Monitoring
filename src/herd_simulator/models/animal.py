"""
animal.py — Per-animal profile and runtime state.

An `AnimalProfile` is generated once at startup and held constant for the
run (FR-3): identity, baseline temperature, natural variability,
behavioural tendency, preferred position in the herd, walking-speed range,
cohesion strength, and fault susceptibility. Profiles are deterministic —
one animal's profile depends only on the global seed and its own
animal_id (config comment: "per-animal seeds derived as seed +
animal_id"), never on generation order or any other animal — so re-running
the same config+seed reproduces byte-identical profiles (AGENTS.md golden
rule 4).

`AnimalState` is the mutable per-tick snapshot (behaviour, position, body
temperature, battery). The future engine tick loop updates it by calling
into behaviour.py / movement.py / physiology.py / battery.py in the fixed
composition order from ADR-014 (Physiology -> Movement -> Social State ->
Collar Faults -> Risk Calculation -> Transmission). This module owns the
*data*, not that orchestration — deliberately, so each subsystem stays
independently testable and the engine (a later deliverable) is the only
place the composition order is encoded.

References:
  Master PRD: "Animal profiles and behaviour"
  HerdSimulator PRD §6.1 (FR-1..FR-3)
  AGENTS.md golden rule 4 (determinism)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from herd_simulator.config import SimulatorConfig
from herd_simulator.models.battery import BatteryState, new_battery_state
from herd_simulator.models.behaviour import Behaviour
from herd_simulator.utils.geo import Coord


@dataclass(frozen=True)
class AnimalProfile:
    """Static per-run traits, generated once and held constant (FR-3)."""

    animal_id: int
    is_physical: bool
    baseline_temp_c: float
    temp_noise_std_c: float
    restlessness: float  # per-animal multiplier on the anomaly restless boost
    preferred_bearing_rad: float  # preferred position in the herd, relative to centroid
    preferred_offset_m: float  # preferred distance from centroid (<= movement.individual_offset_max_m)
    walking_speed_range_mps: tuple[float, float]
    cohesion_strength: float  # per-animal deviation from the herd-wide flocking_strength
    fault_modifier: float  # per-animal susceptibility multiplier for injected events


@dataclass
class AnimalState:
    """Mutable per-tick runtime state for one animal."""

    profile: AnimalProfile
    sim_second: int
    behaviour: Behaviour
    position: Coord
    body_temp_c: float
    battery: BatteryState
    rng: random.Random


def _animal_rng(cfg_seed: int, animal_id: int, stream: str) -> random.Random:
    """A dedicated, deterministic RNG stream for one animal.

    Seeded from the string `"<seed+animal_id>:<stream>"` — `random.Random`
    only accepts None/int/float/str/bytes/bytearray (a tuple raises
    TypeError), so the stream name is folded into a string seed rather than
    a magic numeric offset, while still honoring the documented
    `seed + animal_id` per-animal base (default_config.yaml comment).
    """
    return random.Random(f"{cfg_seed + animal_id}:{stream}")


def generate_profile(animal_id: int, cfg: SimulatorConfig, is_physical: bool = False) -> AnimalProfile:
    """Deterministically generate one animal's static profile.

    Depends only on `cfg.seed` and `animal_id` — never on how many other
    profiles were generated before it, or in what order (AGENTS.md golden
    rule 4).
    """
    rng = _animal_rng(cfg.seed, animal_id, "profile")

    baseline_temp_c = rng.gauss(cfg.physiology.baseline_temp_mean, cfg.physiology.baseline_temp_std)
    walking_low = rng.uniform(0.7, 1.0)
    walking_high = walking_low + rng.uniform(0.3, 0.5)

    return AnimalProfile(
        animal_id=animal_id,
        is_physical=is_physical,
        baseline_temp_c=baseline_temp_c,
        temp_noise_std_c=cfg.physiology.noise_std * rng.uniform(0.8, 1.2),
        restlessness=rng.uniform(0.7, 1.3),
        preferred_bearing_rad=rng.uniform(0.0, 2 * math.pi),
        preferred_offset_m=rng.uniform(0.0, cfg.movement.individual_offset_max_m),
        walking_speed_range_mps=(walking_low, walking_high),
        cohesion_strength=_clamp(cfg.movement.flocking_strength + rng.uniform(-0.15, 0.15), 0.0, 1.0),
        fault_modifier=rng.uniform(0.85, 1.15),
    )


def new_animal_state(
    profile: AnimalProfile,
    cfg: SimulatorConfig,
    start_position: Coord,
    sim_second: int = 0,
) -> AnimalState:
    """Build the initial runtime state for an animal at the start of a run.

    All simulated animals start Resting, matching the §9.1 state diagram's
    `[*] --> Resting` entry point. Body temperature starts exactly at this
    animal's baseline (no diurnal/noise/fever applied yet — that happens on
    the first physiology tick).
    """
    return AnimalState(
        profile=profile,
        sim_second=sim_second,
        behaviour=Behaviour.RESTING,
        position=start_position,
        body_temp_c=profile.baseline_temp_c,
        battery=new_battery_state(cfg.battery),
        rng=_animal_rng(cfg.seed, profile.animal_id, "runtime"),
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
