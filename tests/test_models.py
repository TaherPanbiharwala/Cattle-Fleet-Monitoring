"""
test_models.py — Unit tests for the animal, behaviour, movement,
physiology, and battery models (Deliverable #2).

Mirrors the golden-vector style of test_golden_vectors.py: hand-verified
arithmetic wherever a formula is being pinned down, plus determinism and
invariant checks (no teleport, no recharge, no invalid transitions) that
matter more for a stochastic simulation than any single numeric value.
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

# Add src/ to path so we can import herd_simulator
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import BehaviourConfig, BehaviourTransitions, load_config
from herd_simulator.models import animal, battery, behaviour, movement, physiology
from herd_simulator.models.behaviour import Behaviour, InvalidTransitionError
from herd_simulator.utils.geo import compute_thi, haversine_m, point_in_polygon


# ===================================================================
# Shared fixtures
# ===================================================================

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
DEFAULT_CONFIG = load_config(_CONFIG_PATH)

PASTURE_RECT = DEFAULT_CONFIG.pasture_polygon
CENTER = (12.97125, 79.1595)  # center of PASTURE_RECT

BEHAVIOUR_CFG = DEFAULT_CONFIG.behaviour
BATTERY_CFG = DEFAULT_CONFIG.battery
MOVEMENT_CFG = DEFAULT_CONFIG.movement


# ===================================================================
# 1. Behaviour State Machine
# ===================================================================

class TestBehaviourGraph:
    """Valid transition graph mirrors the HerdSimulator PRD §9.1 diagram."""

    def test_graph_matches_default_config(self):
        """The shipped default_config.yaml must not violate the state diagram."""
        behaviour.validate_config_transitions(BEHAVIOUR_CFG)  # must not raise

    def test_restless_only_reachable_from_resting(self):
        for state in Behaviour:
            if state in (Behaviour.OTHER, Behaviour.RESTING):
                continue
            assert Behaviour.RESTLESS not in behaviour.VALID_TRANSITIONS[state]

    def test_code_five_never_a_transition_target(self):
        """AGENTS.md: never map anything to OTHER (code 5) via this machine."""
        all_targets = {t for targets in behaviour.VALID_TRANSITIONS.values() for t in targets}
        assert Behaviour.OTHER not in all_targets

    def test_invalid_transition_rejected(self):
        with pytest.raises(InvalidTransitionError):
            behaviour.validate_transition(Behaviour.RESTING, Behaviour.WALKING)

    def test_staying_in_place_always_valid(self):
        for state in Behaviour:
            if state == Behaviour.OTHER:
                continue
            behaviour.validate_transition(state, state)  # must not raise

    def test_valid_edge_accepted(self):
        behaviour.validate_transition(Behaviour.RESTING, Behaviour.GRAZING)  # must not raise

    def test_broken_config_is_rejected(self):
        """A hand-edited config that adds an illegal edge (resting -> walking)
        must be caught, even though config.py's own loader would accept it
        (it only checks that the target is *a* known state name, not that
        the specific edge matches the diagram)."""
        broken = BehaviourConfig(
            transition_interval_s=60,
            base_transitions=BehaviourTransitions(
                resting={"walking": 0.1},
                grazing={},
                ruminating={},
                walking={},
                restless={},
            ),
        )
        with pytest.raises(InvalidTransitionError):
            behaviour.validate_config_transitions(broken)


class TestBehaviourStep:
    def test_step_never_produces_invalid_transition(self):
        """Property check: across many rng draws and all 5 states, `step`
        never returns something outside the valid graph."""
        for state in (Behaviour.RESTING, Behaviour.GRAZING, Behaviour.RUMINATING,
                      Behaviour.WALKING, Behaviour.RESTLESS):
            rng = random.Random(1234)
            for _ in range(500):
                result = behaviour.step(state, BEHAVIOUR_CFG, hour_of_day=12, rng=rng, anomaly_active=True)
                behaviour.validate_transition(state, result)  # must not raise

    def test_determinism_same_seed_same_sequence(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        seq1 = [behaviour.step(Behaviour.RESTING, BEHAVIOUR_CFG, 10, rng1) for _ in range(50)]
        seq2 = [behaviour.step(Behaviour.RESTING, BEHAVIOUR_CFG, 10, rng2) for _ in range(50)]
        assert seq1 == seq2

    def test_anomaly_boosts_restless_probability_from_resting(self):
        calm = behaviour.transition_weights(Behaviour.RESTING, BEHAVIOUR_CFG, hour_of_day=12, anomaly_active=False)
        stressed = behaviour.transition_weights(Behaviour.RESTING, BEHAVIOUR_CFG, hour_of_day=12, anomaly_active=True)
        assert stressed[Behaviour.RESTLESS] > calm.get(Behaviour.RESTLESS, 0.0)

    def test_anomaly_has_no_restless_edge_from_grazing(self):
        """Grazing has no valid edge to Restless, so the anomaly boost must
        not fabricate one (ADR-006: never invent edges outside the graph)."""
        weights = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=12, anomaly_active=True)
        assert Behaviour.RESTLESS not in weights

    def test_dawn_hour_boosts_activity_edges(self):
        """FR-5: dawn/dusk favor grazing/walking."""
        dawn = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=6)
        midday = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=12)
        assert dawn[Behaviour.WALKING] > midday[Behaviour.WALKING]

    def test_midday_boosts_rest_edges(self):
        """FR-5: midday/night favor resting/ruminating."""
        midday = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=12)
        dawn = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=6)
        assert midday[Behaviour.RUMINATING] > dawn[Behaviour.RUMINATING]

    def test_fractional_hour_is_floored(self):
        """hour_of_day may be a float sim-clock value; must not KeyError or
        silently skip the boost windows."""
        weights_int = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=6)
        weights_float = behaviour.transition_weights(Behaviour.GRAZING, BEHAVIOUR_CFG, hour_of_day=6.75)
        assert weights_int == weights_float

    def test_weights_never_exceed_total_probability_one(self):
        weights = behaviour.transition_weights(Behaviour.RESTING, BEHAVIOUR_CFG, hour_of_day=6, anomaly_active=True)
        assert sum(weights.values()) <= 1.0 + 1e-9

    def test_scaling_kicks_in_when_total_exceeds_one(self):
        """A synthetic config where boosted probabilities would sum > 1.0
        must be rescaled, not left invalid."""
        hot_cfg = BehaviourConfig(
            transition_interval_s=60,
            base_transitions=BehaviourTransitions(
                resting={"grazing": 0.9, "restless": 0.9},
                grazing={}, ruminating={}, walking={}, restless={},
            ),
        )
        weights = behaviour.transition_weights(Behaviour.RESTING, hot_cfg, hour_of_day=12, anomaly_active=True)
        assert sum(weights.values()) <= 1.0 + 1e-9


# ===================================================================
# 2. Physiology
# ===================================================================

class TestDiurnalCurve:
    def test_peak_at_peak_hour(self):
        assert physiology.diurnal_fraction(14.0, peak_hour=14.0) == pytest.approx(1.0)

    def test_trough_twelve_hours_later(self):
        assert physiology.diurnal_fraction(2.0, peak_hour=14.0) == pytest.approx(-1.0)

    def test_ambient_temperature_at_peak_hour(self):
        temp = physiology.ambient_temperature_c(14.0, day_peak_c=32.0, night_trough_c=24.0)
        assert temp == pytest.approx(32.0, abs=1e-6)

    def test_ambient_temperature_at_trough_hour(self):
        temp = physiology.ambient_temperature_c(2.0, day_peak_c=32.0, night_trough_c=24.0)
        assert temp == pytest.approx(24.0, abs=1e-6)

    def test_ambient_temperature_midpoint_is_mean(self):
        """Quarter-cycle away from peak/trough sits exactly at the mean."""
        temp = physiology.ambient_temperature_c(8.0, day_peak_c=32.0, night_trough_c=24.0)
        assert temp == pytest.approx(28.0, abs=1e-6)


class TestBodyTemperature:
    def test_no_noise_no_diurnal_returns_baseline(self):
        rng = random.Random(1)
        temp = physiology.body_temperature_c(
            baseline_c=38.6, hour_of_day=14.0, diurnal_amplitude_c=0.0, noise_std_c=0.0, rng=rng,
        )
        assert temp == pytest.approx(38.6)

    def test_fever_offset_is_additive(self):
        rng = random.Random(1)
        temp = physiology.body_temperature_c(
            baseline_c=38.6, hour_of_day=14.0, diurnal_amplitude_c=0.0, noise_std_c=0.0,
            rng=rng, fever_offset_c=1.5,
        )
        assert temp == pytest.approx(40.1)

    def test_diurnal_swing_applied_at_peak(self):
        rng = random.Random(1)
        temp = physiology.body_temperature_c(
            baseline_c=38.6, hour_of_day=14.0, diurnal_amplitude_c=0.4, noise_std_c=0.0, rng=rng,
        )
        # amplitude/2 * diurnal_fraction(peak)=1.0 -> +0.2
        assert temp == pytest.approx(38.8, abs=1e-6)


class TestFeverRamp:
    """Onset -> plateau -> recovery, hand-calculated (Master PRD 'Physiology')."""

    def test_before_onset_starts(self):
        assert physiology.fever_ramp_offset_c(-1, 100, 200, 100, 2.0) == 0.0

    def test_onset_start_is_zero(self):
        assert physiology.fever_ramp_offset_c(0, 100, 200, 100, 2.0) == pytest.approx(0.0)

    def test_onset_midpoint_is_half_peak(self):
        assert physiology.fever_ramp_offset_c(50, 100, 200, 100, 2.0) == pytest.approx(1.0)

    def test_plateau_holds_at_peak(self):
        assert physiology.fever_ramp_offset_c(150, 100, 200, 100, 2.0) == pytest.approx(2.0)
        assert physiology.fever_ramp_offset_c(299, 100, 200, 100, 2.0) == pytest.approx(2.0)

    def test_recovery_midpoint_is_half_peak(self):
        # onset=100, plateau=200 -> recovery window is [300, 400); midpoint = 350
        assert physiology.fever_ramp_offset_c(350, 100, 200, 100, 2.0) == pytest.approx(1.0)

    def test_after_full_duration_is_zero(self):
        assert physiology.fever_ramp_offset_c(401, 100, 200, 100, 2.0) == 0.0

    def test_zero_onset_jumps_immediately_to_peak(self):
        assert physiology.fever_ramp_offset_c(0, 0, 100, 50, 3.0) == pytest.approx(3.0)


class TestHeatStress:
    def test_heat_injection_changes_ambient_not_thi_directly(self):
        """Master PRD: heat injection changes ambient conditions; THI then
        moves as a *consequence*, through the one canonical formula."""
        base_ambient = 30.0
        humidity = 60.0
        baseline_thi = compute_thi(base_ambient, humidity)

        heated_ambient = physiology.heat_stress_temperature_c(base_ambient, heat_injection_offset_c=5.0)
        heated_thi = compute_thi(heated_ambient, humidity)

        assert heated_ambient == 35.0
        assert heated_thi > baseline_thi


class TestPhysiologyBounds:
    def test_body_temp_within_bounds_passes(self):
        assert physiology.validate_body_temp(38.6) == 38.6

    def test_body_temp_out_of_bounds_rejected(self):
        with pytest.raises(physiology.PhysiologyBoundsError):
            physiology.validate_body_temp(50.0)

    def test_ambient_temp_within_bounds_passes(self):
        assert physiology.validate_ambient_temp(30.0) == 30.0

    def test_ambient_temp_out_of_bounds_rejected(self):
        with pytest.raises(physiology.PhysiologyBoundsError):
            physiology.validate_ambient_temp(100.0)


# ===================================================================
# 3. Battery
# ===================================================================

class TestBatteryDrain:
    def test_base_rate_one_hour(self):
        """base_drain_per_hour=0.5, 1 hour elapsed, no alert -> drains exactly 0.5%."""
        drained = battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=3600, alert_active=False)
        assert drained == pytest.approx(0.5)

    def test_alert_rate_triples(self):
        """alert_drain_multiplier=3.0 -> 1 hour under alert drains 1.5%."""
        drained = battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=3600, alert_active=True)
        assert drained == pytest.approx(1.5)

    def test_zero_elapsed_drains_nothing(self):
        assert battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=0, alert_active=True) == 0.0

    def test_negative_elapsed_rejected(self):
        with pytest.raises(ValueError):
            battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=-1, alert_active=False)

    def test_dry_run_and_real_time_trajectories_match(self):
        """Summing 3600 one-second steps must equal one 3600-second step —
        the trajectory cannot depend on how finely time is sliced (Master
        PRD: 'Dry-run and real-time modes produce the same trajectory')."""
        one_shot = battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=3600, alert_active=False)
        stepped = sum(battery.drain_for_elapsed(BATTERY_CFG, elapsed_s=1, alert_active=False) for _ in range(3600))
        assert stepped == pytest.approx(one_shot, rel=1e-9)


class TestBatteryStep:
    def test_new_state_starts_at_initial_level(self):
        state = battery.new_battery_state(BATTERY_CFG)
        assert state.level_pct == BATTERY_CFG.initial_level
        assert state.dropped_out is False

    def test_step_never_increases_level(self):
        state = battery.new_battery_state(BATTERY_CFG)
        rng = random.Random(7)
        for _ in range(200):
            alert = rng.random() < 0.5
            new_state = battery.step(state, BATTERY_CFG, elapsed_s=60, alert_active=alert)
            assert new_state.level_pct <= state.level_pct
            state = new_state

    def test_level_clamps_at_zero_never_negative(self):
        state = battery.BatteryState(level_pct=1.0, dropped_out=False)
        depleted = battery.step(state, BATTERY_CFG, elapsed_s=1_000_000, alert_active=True)
        assert depleted.level_pct == 0.0

    def test_dropout_latches_and_freezes_level(self):
        state = battery.BatteryState(level_pct=0.0001, dropped_out=False)
        after = battery.step(state, BATTERY_CFG, elapsed_s=100, alert_active=True)
        assert after.dropped_out is True
        assert after.level_pct == 0.0

        frozen = battery.step(after, BATTERY_CFG, elapsed_s=99999, alert_active=True)
        assert frozen.level_pct == 0.0
        assert frozen.dropped_out is True
        assert frozen is after  # dead battery: step() is a no-op, not a re-clamp


# ===================================================================
# 4. Movement
# ===================================================================

class TestOffsetToLatLon:
    def test_zero_offset_is_origin(self):
        assert movement.offset_to_latlon(CENTER, 0.0, 0.0) == CENTER

    def test_offset_round_trips_through_haversine(self):
        point = movement.offset_to_latlon(CENTER, north_m=50.0, east_m=0.0)
        assert haversine_m(CENTER, point) == pytest.approx(50.0, abs=0.5)


class TestMoveToward:
    def test_target_within_step_reaches_exactly(self):
        current = (12.9716, 79.1589)
        target = (12.97165, 79.1589)  # a few meters away
        result = movement.move_toward(current, target, max_step_m=1000.0)
        assert result == target

    def test_target_beyond_step_is_capped(self):
        current = (12.9700, 79.1589)
        target = (12.9800, 79.1589)  # ~1.1km north
        result = movement.move_toward(current, target, max_step_m=50.0)
        assert haversine_m(current, result) == pytest.approx(50.0, abs=0.5)

    def test_never_overshoots_target(self):
        current = (12.9700, 79.1589)
        target = (12.9701, 79.1589)
        result = movement.move_toward(current, target, max_step_m=5.0)
        assert haversine_m(current, result) <= haversine_m(current, target) + 1e-6

    def test_repeated_steps_converge_without_teleporting(self):
        """Master PRD point 7: 'Teleportation is prohibited except in
        explicit tests.' Every intermediate step must respect max_step_m."""
        current = (12.9700, 79.1589)
        target = (12.9750, 79.1620)
        max_step = 10.0
        # Tolerance reflects move_toward's linear lat/lon interpolation
        # (the same equirectangular-style approximation utils/geo.py already
        # documents as accurate at sub-km scale) diverging from haversine's
        # true great-circle distance by a fraction of a millimeter — not a
        # teleport, just floating-point/approximation noise several orders
        # of magnitude below the 10m step itself.
        for _ in range(2000):
            next_point = movement.move_toward(current, target, max_step)
            assert haversine_m(current, next_point) <= max_step + 1e-3
            current = next_point
            if current == target:
                break
        assert current == target


class TestCentroidAutonomous:
    def test_stays_inside_polygon(self):
        centroid = CENTER
        rng = random.Random(99)
        for _ in range(500):
            centroid = movement.step_centroid_autonomous(centroid, MOVEMENT_CFG, PASTURE_RECT, elapsed_s=30, rng=rng)
            assert point_in_polygon(centroid, PASTURE_RECT)

    def test_zero_elapsed_is_a_no_op(self):
        rng = random.Random(1)
        result = movement.step_centroid_autonomous(CENTER, MOVEMENT_CFG, PASTURE_RECT, elapsed_s=0, rng=rng)
        assert result == CENTER


class TestCentroidAnchored:
    def test_anchors_exactly_when_close(self):
        anchor = (12.971251, 79.159501)
        result = movement.step_centroid_anchored(CENTER, anchor, MOVEMENT_CFG, elapsed_s=60)
        assert result == anchor

    def test_bounded_step_when_far(self):
        centroid = (12.9700, 79.1589)
        anchor = (12.9800, 79.1589)
        result = movement.step_centroid_anchored(centroid, anchor, MOVEMENT_CFG, elapsed_s=10)
        max_step = MOVEMENT_CFG.centroid_speed_m_per_s * 10
        assert haversine_m(centroid, result) == pytest.approx(max_step, abs=0.5)


class TestIndividualOffset:
    def test_resting_offset_is_deterministic_fixed_point(self):
        """Speed 0 -> zero jitter -> identical offset regardless of rng
        state (Master PRD point 3: stationary except GPS noise, which is
        layered on separately by animal_position)."""
        offset_a = movement.individual_offset_m(
            Behaviour.RESTING, preferred_bearing_rad=1.0, preferred_dist_m=10.0,
            cfg=MOVEMENT_CFG, rng=random.Random(1),
        )
        offset_b = movement.individual_offset_m(
            Behaviour.RESTING, preferred_bearing_rad=1.0, preferred_dist_m=10.0,
            cfg=MOVEMENT_CFG, rng=random.Random(999),
        )
        assert offset_a == pytest.approx(offset_b)

    def test_speed_ordering_walking_fastest(self):
        """Master PRD point 4: 'Walking cattle move faster than grazing
        cattle.' Verify the full ordering via average step magnitude over
        many draws (smooths out per-draw jitter)."""

        def avg_step(state, samples=300):
            rng = random.Random(5)
            total = 0.0
            for _ in range(samples):
                n, e = movement.individual_offset_m(
                    state, preferred_bearing_rad=0.0, preferred_dist_m=0.0,
                    cfg=MOVEMENT_CFG, rng=rng, elapsed_s=1.0,
                )
                total += math.hypot(n, e)
            return total / samples

        resting = avg_step(Behaviour.RESTING)
        ruminating = avg_step(Behaviour.RUMINATING)
        grazing = avg_step(Behaviour.GRAZING)
        restless = avg_step(Behaviour.RESTLESS)
        walking = avg_step(Behaviour.WALKING)

        assert resting == 0.0
        assert resting < ruminating < grazing < restless < walking

    def test_isolation_extra_pushes_distance_further(self):
        near = movement.individual_offset_m(
            Behaviour.GRAZING, preferred_bearing_rad=0.0, preferred_dist_m=5.0,
            cfg=MOVEMENT_CFG, rng=random.Random(3), isolation_extra_m=0.0,
        )
        far = movement.individual_offset_m(
            Behaviour.GRAZING, preferred_bearing_rad=0.0, preferred_dist_m=5.0,
            cfg=MOVEMENT_CFG, rng=random.Random(3), isolation_extra_m=50.0,
        )
        assert math.hypot(*far) > math.hypot(*near)


class TestAnimalPosition:
    def test_resting_position_stays_near_fixed_offset(self):
        position = movement.animal_position(
            CENTER, Behaviour.RESTING, preferred_bearing_rad=0.5, preferred_dist_m=8.0,
            cfg=MOVEMENT_CFG, rng=random.Random(1),
        )
        # Fixed cohesion offset (~a few meters) plus small GPS noise only.
        assert haversine_m(CENTER, position) < 20.0


class TestIsolationDrift:
    def test_grows_with_elapsed_time(self):
        assert movement.isolation_extra_distance_m(10) < movement.isolation_extra_distance_m(100)

    def test_capped_at_maximum(self):
        assert movement.isolation_extra_distance_m(1_000_000) == movement.ISOLATION_MAX_EXTRA_M

    def test_zero_at_zero_elapsed(self):
        assert movement.isolation_extra_distance_m(0) == 0.0


class TestBreachExcursionTarget:
    def test_target_from_center_is_outside_polygon(self):
        target = movement.breach_excursion_target(CENTER, PASTURE_RECT, outward_m=20.0)
        assert not point_in_polygon(target, PASTURE_RECT)

    def test_target_from_near_edge_is_outside_polygon(self):
        near_north_edge = (12.9719, 79.1595)
        target = movement.breach_excursion_target(near_north_edge, PASTURE_RECT, outward_m=15.0)
        assert not point_in_polygon(target, PASTURE_RECT)

    def test_scales_beyond_default_search_floor_for_large_pastures(self):
        """A pasture larger than the old fixed 1km search cap must still
        find a genuine outside point, not silently return one still inside
        (the bug: a fixed step count times a fixed step size stops
        searching before reaching the boundary of a big-enough polygon)."""
        large_center = (12.9716, 79.1589)
        large_pasture = [
            (large_center[0] + 0.02, large_center[1] - 0.02),
            (large_center[0] + 0.02, large_center[1] + 0.02),
            (large_center[0] - 0.02, large_center[1] + 0.02),
            (large_center[0] - 0.02, large_center[1] - 0.02),
        ]
        assert movement._polygon_diameter_m(large_pasture) > 1000.0  # confirms this exercises the fix

        target = movement.breach_excursion_target(large_center, large_pasture, outward_m=20.0)
        assert not point_in_polygon(target, large_pasture)


# ===================================================================
# 5. Animal Profile & State
# ===================================================================

class TestAnimalProfile:
    def test_deterministic_same_seed_same_id(self):
        p1 = animal.generate_profile(5, DEFAULT_CONFIG)
        p2 = animal.generate_profile(5, DEFAULT_CONFIG)
        assert p1 == p2

    def test_different_ids_produce_different_profiles(self):
        p5 = animal.generate_profile(5, DEFAULT_CONFIG)
        p6 = animal.generate_profile(6, DEFAULT_CONFIG)
        assert p5.baseline_temp_c != p6.baseline_temp_c

    def test_generation_order_independent(self):
        """Generating profile 6 before or after profile 5 must not change
        profile 5's result — each animal's stream is fully independent
        (AGENTS.md golden rule 4)."""
        first = animal.generate_profile(5, DEFAULT_CONFIG)
        animal.generate_profile(6, DEFAULT_CONFIG)
        second = animal.generate_profile(5, DEFAULT_CONFIG)
        assert first == second

    def test_preferred_offset_within_configured_max(self):
        for animal_id in range(2, 21):
            profile = animal.generate_profile(animal_id, DEFAULT_CONFIG)
            assert 0.0 <= profile.preferred_offset_m <= DEFAULT_CONFIG.movement.individual_offset_max_m

    def test_cohesion_strength_bounded_zero_one(self):
        for animal_id in range(2, 21):
            profile = animal.generate_profile(animal_id, DEFAULT_CONFIG)
            assert 0.0 <= profile.cohesion_strength <= 1.0

    def test_walking_speed_range_is_ordered(self):
        profile = animal.generate_profile(7, DEFAULT_CONFIG)
        low, high = profile.walking_speed_range_mps
        assert 0.0 < low < high

    def test_is_physical_flag_propagates(self):
        physical = animal.generate_profile(1, DEFAULT_CONFIG, is_physical=True)
        simulated = animal.generate_profile(2, DEFAULT_CONFIG, is_physical=False)
        assert physical.is_physical is True
        assert simulated.is_physical is False


class TestAnimalState:
    def test_initial_behaviour_is_resting(self):
        """Matches the §9.1 state diagram entry point: [*] --> Resting."""
        profile = animal.generate_profile(3, DEFAULT_CONFIG)
        state = animal.new_animal_state(profile, DEFAULT_CONFIG, start_position=CENTER)
        assert state.behaviour == Behaviour.RESTING

    def test_initial_body_temp_equals_baseline(self):
        profile = animal.generate_profile(3, DEFAULT_CONFIG)
        state = animal.new_animal_state(profile, DEFAULT_CONFIG, start_position=CENTER)
        assert state.body_temp_c == profile.baseline_temp_c

    def test_initial_battery_matches_config(self):
        profile = animal.generate_profile(3, DEFAULT_CONFIG)
        state = animal.new_animal_state(profile, DEFAULT_CONFIG, start_position=CENTER)
        assert state.battery.level_pct == DEFAULT_CONFIG.battery.initial_level
        assert state.battery.dropped_out is False

    def test_runtime_rng_independent_from_profile_rng(self):
        """The runtime stream must not be a continuation of the
        profile-generation stream — otherwise adding a field to
        generate_profile would silently reseed every animal's runtime
        behaviour."""
        profile = animal.generate_profile(3, DEFAULT_CONFIG)
        state = animal.new_animal_state(profile, DEFAULT_CONFIG, start_position=CENTER)
        profile_stream = animal._animal_rng(DEFAULT_CONFIG.seed, 3, "profile")
        assert state.rng.random() != profile_stream.random()

    def test_two_animals_get_different_runtime_streams(self):
        p3 = animal.generate_profile(3, DEFAULT_CONFIG)
        p4 = animal.generate_profile(4, DEFAULT_CONFIG)
        s3 = animal.new_animal_state(p3, DEFAULT_CONFIG, start_position=CENTER)
        s4 = animal.new_animal_state(p4, DEFAULT_CONFIG, start_position=CENTER)
        assert s3.rng.random() != s4.rng.random()


# ===================================================================
# 6. Cross-module smoke test
# ===================================================================

class TestModelsComposeAcrossTicks:
    """Not the engine's tick loop (a later deliverable) — just a smoke test
    that the five modules can be driven together for many ticks without
    violating any invariant a real engine would also have to respect."""

    def test_one_simulated_hour_stays_within_invariants(self):
        cfg = DEFAULT_CONFIG
        profile = animal.generate_profile(4, cfg)
        state = animal.new_animal_state(profile, cfg, start_position=CENTER)
        centroid = CENTER

        for sim_second in range(0, 3600, 60):
            hour_of_day = (sim_second / 3600.0) % 24

            centroid = movement.step_centroid_autonomous(centroid, cfg.movement, PASTURE_RECT, 60, state.rng)
            assert point_in_polygon(centroid, PASTURE_RECT)

            new_behaviour = behaviour.step(state.behaviour, cfg.behaviour, hour_of_day, state.rng)
            behaviour.validate_transition(state.behaviour, new_behaviour)
            state.behaviour = new_behaviour

            state.position = movement.animal_position(
                centroid, state.behaviour, profile.preferred_bearing_rad, profile.preferred_offset_m,
                cfg.movement, state.rng, elapsed_s=60, walking_speed_range_mps=profile.walking_speed_range_mps,
            )
            assert haversine_m(centroid, state.position) < 200.0

            state.body_temp_c = physiology.body_temperature_c(
                profile.baseline_temp_c, hour_of_day, cfg.physiology.diurnal_amplitude,
                profile.temp_noise_std_c, state.rng,
            )
            physiology.validate_body_temp(state.body_temp_c)

            state.battery = battery.step(state.battery, cfg.battery, elapsed_s=60, alert_active=False)

        assert 0.0 <= state.battery.level_pct < 100.0
        assert not state.battery.dropped_out
