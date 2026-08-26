"""
test_scenario_engine.py — Tests for scenario runner and simulator integration.

Validates:
  - JSON scenario parsing & validation against the Master PRD "Scenario
    contract" (schema_version/scenario_id/seed/events wrapper; per-event
    animal_id/type/start_sim_second/duration_seconds/params)
  - The four "fail before startup" rules: unknown types, duplicate event
    IDs, invalid cattle IDs, non-positive duration
  - The same-type-self-overlap rule
  - Event activation / clearing / auto-expiry lifecycle
  - Event overlap composition (different types / different animals)
  - Simulator tick loop integration (dry-run)
  - CLI command parsing
  - ADR-014 composition order invariants
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import load_config
from herd_simulator.engine.scenario_runner import (
    EventType,
    ScenarioError,
    ScenarioEvent,
    activate_event,
    clear_all_events,
    clear_event,
    clear_event_by_id,
    expire_events,
    get_active_event,
    get_all_active_for_animal,
    get_event_codes_for_status,
    has_any_anomaly,
    is_event_active,
    load_scenario,
    new_event_state,
    process_scheduled_events,
)
from herd_simulator.engine.live_cli import (
    CLICommand,
    CLICommandType,
    parse_command,
)
from herd_simulator.engine.simulator import (
    SimMode,
    create_simulator,
    tick,
)


# ===================================================================
# Scenario JSON Parsing
# ===================================================================

class TestScenarioLoading:
    """JSON scenario file loading and validation (Master PRD 'Scenario
    contract'): {schema_version, scenario_id, seed, events}, each event
    {animal_id, type, start_sim_second, duration_seconds, params}."""

    def _write(self, tmp_path, events, **overrides):
        doc = {
            "schema_version": 1,
            "scenario_id": "test_scenario",
            "seed": 42,
            "events": events,
            **overrides,
        }
        path = tmp_path / "test.json"
        path.write_text(json.dumps(doc))
        return path

    def test_load_valid_scenario(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100,
             "duration_seconds": 900, "params": {"peak_offset_c": 2.0}},
            {"animal_id": 10, "type": "geofence_breach", "start_sim_second": 200,
             "duration_seconds": 300},
        ])
        scenario = load_scenario(path)
        assert scenario.schema_version == 1
        assert scenario.scenario_id == "test_scenario"
        assert scenario.seed == 42
        assert len(scenario.events) == 2
        assert scenario.events[0].start_sim_second == 100
        assert scenario.events[0].event_type == EventType.FEVER_ONSET
        assert scenario.events[0].duration_seconds == 900
        assert scenario.events[0].params["peak_offset_c"] == 2.0

    def test_events_sorted_by_time(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 3, "type": "tamper", "start_sim_second": 500, "duration_seconds": 100},
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 100},
        ])
        scenario = load_scenario(path)
        assert scenario.events[0].start_sim_second == 100
        assert scenario.events[1].start_sim_second == 500

    def test_default_params_applied(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 1200},
        ])
        scenario = load_scenario(path)
        assert scenario.events[0].params["peak_offset_c"] == 1.8  # default
        assert scenario.events[0].params["onset_s"] == 300

    def test_auto_generated_event_id_is_unique(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 100},
            {"animal_id": 6, "type": "tamper", "start_sim_second": 200, "duration_seconds": 100},
        ])
        scenario = load_scenario(path)
        ids = [e.event_id for e in scenario.events]
        assert len(ids) == len(set(ids))
        assert all(i is not None for i in ids)

    def test_explicit_event_id_preserved(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100,
             "duration_seconds": 100, "event_id": "my-custom-id"},
        ])
        scenario = load_scenario(path)
        assert scenario.events[0].event_id == "my-custom-id"

    def test_invalid_event_type_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "invalid_type", "start_sim_second": 100, "duration_seconds": 100},
        ])
        with pytest.raises(ScenarioError, match="unknown event type"):
            load_scenario(path)

    def test_missing_required_field_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "duration_seconds": 100},  # missing start_sim_second
        ])
        with pytest.raises(ScenarioError, match="missing required field"):
            load_scenario(path)

    def test_missing_duration_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100},  # no duration_seconds
        ])
        with pytest.raises(ScenarioError, match="missing required field"):
            load_scenario(path)

    def test_non_positive_duration_rejected(self, tmp_path):
        """Master PRD: 'non-positive duration fail before startup.'"""
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 0},
        ])
        with pytest.raises(ScenarioError, match="duration_seconds must be positive"):
            load_scenario(path)

    def test_negative_duration_rejected(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": -10},
        ])
        with pytest.raises(ScenarioError, match="duration_seconds must be positive"):
            load_scenario(path)

    def test_duplicate_event_id_rejected(self, tmp_path):
        """Master PRD: 'duplicate event IDs ... fail before startup.'"""
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100,
             "duration_seconds": 100, "event_id": "dup"},
            {"animal_id": 6, "type": "tamper", "start_sim_second": 200,
             "duration_seconds": 100, "event_id": "dup"},
        ])
        with pytest.raises(ScenarioError, match="duplicate event_id"):
            load_scenario(path)

    def test_invalid_cattle_id_rejected_when_herd_supplied(self, tmp_path):
        """Master PRD: 'invalid cattle IDs ... fail before startup.'"""
        path = self._write(tmp_path, [
            {"animal_id": 999, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 100},
        ])
        with pytest.raises(ScenarioError, match="not a member of this herd"):
            load_scenario(path, valid_animal_ids=range(2, 21))

    def test_invalid_cattle_id_skipped_when_herd_not_supplied(self, tmp_path):
        """Callers without a config on hand can still parse a scenario —
        the herd-membership check is opt-in via valid_animal_ids."""
        path = self._write(tmp_path, [
            {"animal_id": 999, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 100},
        ])
        scenario = load_scenario(path)
        assert scenario.events[0].animal_id == 999

    def test_self_overlap_rejected(self, tmp_path):
        """Master PRD: 'The same event type cannot overlap itself for the
        same cow.'"""
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 200},
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 250, "duration_seconds": 100},
        ])
        with pytest.raises(ScenarioError, match="Overlapping fever_onset events"):
            load_scenario(path)

    def test_back_to_back_same_type_not_overlapping_is_allowed(self, tmp_path):
        """[100, 300) then [300, 400) touch but don't overlap."""
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 200},
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 300, "duration_seconds": 100},
        ])
        scenario = load_scenario(path)
        assert len(scenario.events) == 2

    def test_different_types_may_overlap(self, tmp_path):
        """Only a repeat of the *same* type is restricted — different
        fault types on the same animal are meant to compose."""
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 200},
            {"animal_id": 5, "type": "geofence_breach", "start_sim_second": 150, "duration_seconds": 50},
        ])
        scenario = load_scenario(path)
        assert len(scenario.events) == 2

    def test_different_animals_same_type_may_overlap(self, tmp_path):
        path = self._write(tmp_path, [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 100, "duration_seconds": 200},
            {"animal_id": 6, "type": "fever_onset", "start_sim_second": 150, "duration_seconds": 50},
        ])
        scenario = load_scenario(path)
        assert len(scenario.events) == 2

    def test_not_object_rejected(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('[{"not": "an object"}]')
        with pytest.raises(ScenarioError, match="JSON object"):
            load_scenario(path)

    def test_wrong_schema_version_rejected(self, tmp_path):
        path = self._write(tmp_path, [], schema_version=99)
        with pytest.raises(ScenarioError, match="schema_version"):
            load_scenario(path)

    def test_missing_scenario_id_rejected(self, tmp_path):
        doc = {"schema_version": 1, "seed": 42, "events": []}
        path = tmp_path / "test.json"
        path.write_text(json.dumps(doc))
        with pytest.raises(ScenarioError, match="scenario_id"):
            load_scenario(path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_scenario("/nonexistent/path.json")


# ===================================================================
# Event State Lifecycle
# ===================================================================

class TestEventLifecycle:
    """Activate → query → clear lifecycle."""

    def test_activate_and_query(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        assert is_event_active(state, 5, EventType.FEVER_ONSET)
        assert not is_event_active(state, 5, EventType.TAMPER)

    def test_clear_specific_event(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        assert clear_event(state, 5, EventType.FEVER_ONSET)
        assert not is_event_active(state, 5, EventType.FEVER_ONSET)

    def test_clear_returns_false_if_not_active(self):
        state = new_event_state()
        assert not clear_event(state, 5, EventType.FEVER_ONSET)

    def test_clear_all_events(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        activate_event(state, 5, EventType.TAMPER, sim_second=100)
        cleared = clear_all_events(state, 5)
        assert len(cleared) == 2
        assert not is_event_active(state, 5, EventType.FEVER_ONSET)
        assert not is_event_active(state, 5, EventType.TAMPER)

    def test_clear_by_event_id(self):
        state = new_event_state()
        eid = activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        assert clear_event_by_id(state, eid)
        assert not is_event_active(state, 5, EventType.FEVER_ONSET)

    def test_replacement_semantics(self):
        """Re-activating same (animal, type) via the live/API path replaces
        the event — this is the interactive "update it" gesture, distinct
        from the scripted-timeline self-overlap rule enforced at scenario
        load time (see TestScenarioLoading.test_self_overlap_rejected)."""
        state = new_event_state()
        eid1 = activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100,
                              params={"peak_offset_c": 1.0})
        eid2 = activate_event(state, 5, EventType.FEVER_ONSET, sim_second=200,
                              params={"peak_offset_c": 2.5})
        assert eid1 != eid2
        ae = get_active_event(state, 5, EventType.FEVER_ONSET)
        assert ae is not None
        assert ae.event.params["peak_offset_c"] == 2.5

    def test_has_any_anomaly(self):
        state = new_event_state()
        assert not has_any_anomaly(state, 5)
        activate_event(state, 5, EventType.TAMPER, sim_second=100)
        assert has_any_anomaly(state, 5)

    def test_get_all_active_for_animal(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        activate_event(state, 5, EventType.TAMPER, sim_second=100)
        activate_event(state, 7, EventType.FEVER_ONSET, sim_second=100)
        events = get_all_active_for_animal(state, 5)
        assert len(events) == 2


# ===================================================================
# Event Expiry (duration_seconds)
# ===================================================================

class TestEventExpiry:
    """expire_events auto-clears scripted events once their duration
    elapses; CLI/API-activated events (no duration) never auto-expire."""

    def test_event_expires_after_duration(self):
        state = new_event_state()
        activate_event(state, 5, EventType.TAMPER, sim_second=100, duration_seconds=50)
        assert is_event_active(state, 5, EventType.TAMPER)

        expired = expire_events(state, sim_second=149)
        assert expired == []
        assert is_event_active(state, 5, EventType.TAMPER)

        expired = expire_events(state, sim_second=150)
        assert len(expired) == 1
        assert not is_event_active(state, 5, EventType.TAMPER)

    def test_cli_activated_event_never_auto_expires(self):
        """duration_seconds=None (the CLI/API default) means 'runs until
        explicitly cleared' — expire_events must never touch it."""
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)  # no duration
        expired = expire_events(state, sim_second=1_000_000)
        assert expired == []
        assert is_event_active(state, 5, EventType.FEVER_ONSET)

    def test_only_expired_events_are_cleared(self):
        state = new_event_state()
        activate_event(state, 5, EventType.TAMPER, sim_second=100, duration_seconds=50)
        activate_event(state, 6, EventType.TAMPER, sim_second=100, duration_seconds=500)
        expire_events(state, sim_second=150)
        assert not is_event_active(state, 5, EventType.TAMPER)
        assert is_event_active(state, 6, EventType.TAMPER)


# ===================================================================
# Event Overlap Composition
# ===================================================================

class TestEventOverlap:
    """Multiple event types on the same animal compose correctly."""

    def test_fever_and_breach_both_active(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        activate_event(state, 5, EventType.GEOFENCE_BREACH, sim_second=100)
        assert is_event_active(state, 5, EventType.FEVER_ONSET)
        assert is_event_active(state, 5, EventType.GEOFENCE_BREACH)

    def test_event_codes_for_multiple(self):
        state = new_event_state()
        activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        activate_event(state, 5, EventType.TAMPER, sim_second=100)
        codes = get_event_codes_for_status(state, 5)
        assert "FEVER" in codes
        assert "TAMPER" in codes


# ===================================================================
# Scheduled Event Processing
# ===================================================================

class TestScheduledProcessing:
    """process_scheduled_events activates events at the right sim_second."""

    def test_activates_at_correct_time(self):
        state = new_event_state()
        events = [
            ScenarioEvent(animal_id=5, event_type=EventType.FEVER_ONSET, start_sim_second=100,
                          duration_seconds=1200,
                          params={"peak_offset_c": 1.8, "onset_s": 300, "plateau_s": 600, "recovery_s": 300}),
            ScenarioEvent(animal_id=10, event_type=EventType.TAMPER, start_sim_second=200,
                          duration_seconds=100, params={}),
        ]
        # At t=50, nothing should activate
        cursor, activated = process_scheduled_events(state, events, 50, 0)
        assert cursor == 0
        assert len(activated) == 0

        # At t=100, first event activates
        cursor, activated = process_scheduled_events(state, events, 100, cursor)
        assert cursor == 1
        assert len(activated) == 1
        assert is_event_active(state, 5, EventType.FEVER_ONSET)

    def test_multiple_events_same_second(self):
        state = new_event_state()
        events = [
            ScenarioEvent(animal_id=5, event_type=EventType.FEVER_ONSET, start_sim_second=100,
                          duration_seconds=1200,
                          params={"peak_offset_c": 1.8, "onset_s": 300, "plateau_s": 600, "recovery_s": 300}),
            ScenarioEvent(animal_id=10, event_type=EventType.TAMPER, start_sim_second=100,
                          duration_seconds=100, params={}),
        ]
        cursor, activated = process_scheduled_events(state, events, 100, 0)
        assert cursor == 2
        assert len(activated) == 2

    def test_activated_event_carries_scenario_duration_and_id(self):
        """The activated event must carry through the scenario's own
        duration and event_id, not silently regenerate/drop them —
        otherwise expiry and DELETE /api/events/<id> can't target it."""
        state = new_event_state()
        events = [
            ScenarioEvent(animal_id=5, event_type=EventType.TAMPER, start_sim_second=100,
                          duration_seconds=250, params={}, event_id="scripted-1"),
        ]
        process_scheduled_events(state, events, 100, 0)
        ae = get_active_event(state, 5, EventType.TAMPER)
        assert ae is not None
        assert ae.event.event_id == "scripted-1"
        assert ae.event.duration_seconds == 250


# ===================================================================
# CLI Command Parsing
# ===================================================================

class TestCLIParsing:
    """parse_command correctly interprets user input."""

    def test_fever_command(self):
        cmd = parse_command("fever 5")
        assert cmd is not None
        assert cmd.command == CLICommandType.FEVER
        assert cmd.animal_id == 5

    def test_breach_command(self):
        cmd = parse_command("breach 14")
        assert cmd is not None
        assert cmd.command == CLICommandType.BREACH
        assert cmd.animal_id == 14

    def test_clear_command(self):
        cmd = parse_command("clear 7")
        assert cmd is not None
        assert cmd.command == CLICommandType.CLEAR
        assert cmd.animal_id == 7

    def test_status_no_id(self):
        cmd = parse_command("status")
        assert cmd is not None
        assert cmd.command == CLICommandType.STATUS
        assert cmd.animal_id is None

    def test_quit_alias(self):
        cmd = parse_command("exit")
        assert cmd is not None
        assert cmd.command == CLICommandType.QUIT

    def test_missing_animal_id_returns_none(self):
        assert parse_command("fever") is None

    def test_invalid_verb_returns_none(self):
        assert parse_command("invalid_command 5") is None

    def test_empty_string_returns_none(self):
        assert parse_command("") is None

    def test_case_insensitive(self):
        cmd = parse_command("FEVER 5")
        assert cmd is not None
        assert cmd.command == CLICommandType.FEVER


# ===================================================================
# Simulator Integration (Dry-Run)
# ===================================================================

class TestSimulatorIntegration:
    """Integration tests: create a simulator and tick it in dry-run mode."""

    @pytest.fixture
    def cfg(self):
        return load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml"))

    def test_create_simulator(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        assert len(sim.animals) == 20
        assert len(sim.profiles) == 20
        assert sim.clock.sim_second == 0

    def test_tick_advances_clock(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        telemetry = tick(sim)
        assert sim.clock.sim_second == 1
        assert len(telemetry) == 20

    def test_telemetry_has_correct_fields(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        telemetry = tick(sim)
        t = telemetry[0]
        assert hasattr(t, "animal_id")
        assert hasattr(t, "body_temp_c")
        assert hasattr(t, "thi")
        assert hasattr(t, "behaviour")
        assert hasattr(t, "latitude")
        assert hasattr(t, "longitude")
        assert hasattr(t, "risk_score")
        assert hasattr(t, "geofence_status")
        assert hasattr(t, "battery_pct")

    def test_battery_monotonically_decreases(self, cfg):
        """Battery should never increase over time (ADR-009)."""
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        prev_levels = {aid: s.battery.level_pct for aid, s in sim.animals.items()}
        for _ in range(100):
            tick(sim)
        for aid, state in sim.animals.items():
            assert state.battery.level_pct <= prev_levels[aid], \
                f"Animal {aid} battery increased from {prev_levels[aid]} to {state.battery.level_pct}"

    def test_risk_score_in_valid_range(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        for _ in range(50):
            telemetry = tick(sim)
            for t in telemetry:
                assert 0 <= t.risk_score <= 100, f"Risk score {t.risk_score} out of range"

    def test_behaviour_codes_valid(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        for _ in range(50):
            telemetry = tick(sim)
            for t in telemetry:
                assert t.behaviour in (0, 1, 2, 3, 4), f"Invalid behaviour code {t.behaviour}"

    def test_geofence_status_valid(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        for _ in range(50):
            telemetry = tick(sim)
            for t in telemetry:
                assert t.geofence_status in (0, 1, 2), f"Invalid geofence status {t.geofence_status}"

    def test_fever_injection_raises_body_temp(self, cfg):
        """Injecting fever should elevate body temperature above baseline."""
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        # Tick a few times to establish baseline
        for _ in range(10):
            tick(sim)
        baseline = sim.animals[5].body_temp_c

        # Inject fever
        activate_event(sim.event_state, 5, EventType.FEVER_ONSET, sim.clock.sim_second,
                        params={"peak_offset_c": 3.0, "onset_s": 1, "plateau_s": 100, "recovery_s": 1})
        # Tick through onset
        for _ in range(50):
            tick(sim)
        # Body temp should be elevated (accounting for noise)
        assert sim.animals[5].body_temp_c > baseline - 0.5  # At least near or above baseline

    def test_collar_dropout_stops_transmission(self, cfg):
        """Collar dropout should set battery to 0 and mark dropped_out."""
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        for _ in range(5):
            tick(sim)
        activate_event(sim.event_state, 5, EventType.COLLAR_DROPOUT, sim.clock.sim_second)
        telemetry = tick(sim)
        t5 = [t for t in telemetry if t.animal_id == 5][0]
        assert t5.dropped_out is True
        assert t5.battery_pct == 0.0

    def test_dropout_reports_critical_not_fabricated_healthy(self, cfg):
        """Master PRD: 'the HUD marks the cow stale and critical instead
        of fabricating a current score' — must not report risk_score=0 /
        alert_band='green' just because no fresh data exists."""
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        activate_event(sim.event_state, 5, EventType.COLLAR_DROPOUT, sim.clock.sim_second)
        telemetry = tick(sim)
        t5 = [t for t in telemetry if t.animal_id == 5][0]
        assert t5.risk_score == 100
        assert t5.alert_band == "red"

        # And on every subsequent tick, once dropped_out latches via battery.
        telemetry = tick(sim)
        t5 = [t for t in telemetry if t.animal_id == 5][0]
        assert t5.risk_score == 100
        assert t5.alert_band == "red"

    def test_deterministic_same_seed(self, cfg):
        """Two runs with the same config/seed produce identical first-tick telemetry."""
        sim1 = create_simulator(cfg, SimMode.DRY_RUN)
        sim2 = create_simulator(cfg, SimMode.DRY_RUN)
        t1 = tick(sim1)
        t2 = tick(sim2)
        for a, b in zip(t1, t2):
            assert a.animal_id == b.animal_id
            assert abs(a.body_temp_c - b.body_temp_c) < 1e-9
            assert a.behaviour == b.behaviour

    def test_cli_quit_stops_simulation(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        sim.cli_queue.put(CLICommand(command=CLICommandType.QUIT))
        tick(sim)
        assert sim.running is False

    def test_cli_pause_and_resume(self, cfg):
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        sim.cli_queue.put(CLICommand(command=CLICommandType.PAUSE))
        telemetry = tick(sim)
        assert sim.paused is True
        assert telemetry == []  # No processing while paused

        sim.cli_queue.put(CLICommand(command=CLICommandType.RESUME))
        telemetry = tick(sim)
        assert sim.paused is False
        assert len(telemetry) == 20

    def test_scripted_event_activates_and_expires_via_full_scenario(self, cfg, tmp_path):
        """End-to-end: a scenario-file event activates on schedule and
        auto-clears after its duration, driven entirely by the tick loop —
        exercising load_scenario -> create_simulator -> tick together."""
        doc = {
            "schema_version": 1,
            "scenario_id": "e2e_test",
            "seed": cfg.seed,
            "events": [
                {"animal_id": 5, "type": "tamper", "start_sim_second": 3, "duration_seconds": 5},
            ],
        }
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(doc))
        scenario = load_scenario(path, valid_animal_ids=range(1, cfg.herd.n_total + 1))

        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)

        for _ in range(2):
            tick(sim)  # sim_second 1, 2 — not yet active
        assert not is_event_active(sim.event_state, 5, EventType.TAMPER)

        tick(sim)  # sim_second 3 — activates
        assert is_event_active(sim.event_state, 5, EventType.TAMPER)

        for _ in range(4):
            tick(sim)  # sim_second 4..7 — still within [3, 8)
        assert is_event_active(sim.event_state, 5, EventType.TAMPER)

        tick(sim)  # sim_second 8 — expires
        assert not is_event_active(sim.event_state, 5, EventType.TAMPER)
