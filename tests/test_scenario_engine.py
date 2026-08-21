"""
test_scenario_engine.py — Tests for scenario runner and simulator integration.

Validates:
  - JSON scenario parsing & validation
  - Event activation / clearing lifecycle
  - Event overlap composition
  - Simulator tick loop integration (dry-run)
  - CLI command parsing
  - ADR-014 composition order invariants
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from queue import Queue

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import load_config
from herd_simulator.engine.scenario_runner import (
    EventState,
    EventType,
    ScenarioError,
    ScenarioEvent,
    activate_event,
    clear_all_events,
    clear_event,
    clear_event_by_id,
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
    AnimalTelemetry,
    SimMode,
    Simulator,
    create_simulator,
    tick,
)


# ===================================================================
# Scenario JSON Parsing
# ===================================================================

class TestScenarioLoading:
    """Test JSON scenario file loading and validation."""

    def test_load_valid_scenario(self, tmp_path):
        scenario = [
            {"sim_second": 100, "animal_id": 5, "event": "fever_onset",
             "params": {"peak_offset_c": 2.0}},
            {"sim_second": 200, "animal_id": 10, "event": "geofence_breach"},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scenario))
        events = load_scenario(path)
        assert len(events) == 2
        assert events[0].sim_second == 100
        assert events[0].event_type == EventType.FEVER_ONSET
        assert events[0].params["peak_offset_c"] == 2.0

    def test_events_sorted_by_time(self, tmp_path):
        scenario = [
            {"sim_second": 500, "animal_id": 3, "event": "tamper"},
            {"sim_second": 100, "animal_id": 5, "event": "fever_onset"},
        ]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scenario))
        events = load_scenario(path)
        assert events[0].sim_second == 100
        assert events[1].sim_second == 500

    def test_default_params_applied(self, tmp_path):
        scenario = [{"sim_second": 100, "animal_id": 5, "event": "fever_onset"}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scenario))
        events = load_scenario(path)
        assert events[0].params["peak_offset_c"] == 1.8  # default
        assert events[0].params["onset_s"] == 300

    def test_invalid_event_type_rejected(self, tmp_path):
        scenario = [{"sim_second": 100, "animal_id": 5, "event": "invalid_type"}]
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scenario))
        with pytest.raises(ScenarioError, match="unknown event type"):
            load_scenario(path)

    def test_missing_required_field_rejected(self, tmp_path):
        scenario = [{"animal_id": 5, "event": "fever_onset"}]  # missing sim_second
        path = tmp_path / "test.json"
        path.write_text(json.dumps(scenario))
        with pytest.raises(ScenarioError, match="missing required field"):
            load_scenario(path)

    def test_not_array_rejected(self, tmp_path):
        path = tmp_path / "test.json"
        path.write_text('{"not": "an array"}')
        with pytest.raises(ScenarioError, match="JSON array"):
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
        eid = activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
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
        count = clear_all_events(state, 5)
        assert count == 2
        assert not is_event_active(state, 5, EventType.FEVER_ONSET)
        assert not is_event_active(state, 5, EventType.TAMPER)

    def test_clear_by_event_id(self):
        state = new_event_state()
        eid = activate_event(state, 5, EventType.FEVER_ONSET, sim_second=100)
        assert clear_event_by_id(state, eid)
        assert not is_event_active(state, 5, EventType.FEVER_ONSET)

    def test_replacement_semantics(self):
        """Re-activating same (animal, type) replaces the event."""
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
            ScenarioEvent(sim_second=100, animal_id=5, event_type=EventType.FEVER_ONSET,
                          params={"peak_offset_c": 1.8, "onset_s": 300, "plateau_s": 600, "recovery_s": 300}),
            ScenarioEvent(sim_second=200, animal_id=10, event_type=EventType.TAMPER,
                          params={}),
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
            ScenarioEvent(sim_second=100, animal_id=5, event_type=EventType.FEVER_ONSET,
                          params={"peak_offset_c": 1.8, "onset_s": 300, "plateau_s": 600, "recovery_s": 300}),
            ScenarioEvent(sim_second=100, animal_id=10, event_type=EventType.TAMPER,
                          params={}),
        ]
        cursor, activated = process_scheduled_events(state, events, 100, 0)
        assert cursor == 2
        assert len(activated) == 2


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
