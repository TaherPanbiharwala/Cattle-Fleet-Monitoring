"""
test_logger.py — Tests for the per-run structured logging service.

Covers:
  - Buffered writer mechanics (threshold, flush, close, counter)
  - Run directory creation, manifest/config/profiles JSON
  - Telemetry CSV format and float normalization
  - Events JSONL format (activated/expired/cleared)
  - Transmissions JSONL format
  - Ground truth (190 pairs, deterministic ordering, Haversine match)
  - Summary JSON
  - wire_logger integration with simulator
  - Determinism: two runs with same seed → identical normalized output
"""

from __future__ import annotations

import csv
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import load_config
from herd_simulator.engine.scenario_runner import (
    EventType,
    activate_event,
    load_scenario,
)
from herd_simulator.engine.simulator import (
    AnimalTelemetry,
    SimMode,
    create_simulator,
    tick,
)
from herd_simulator.services.logger import (
    BufferedWriter,
    RunLogger,
    _close_writer,
    _flush_writer,
    _new_buffered_writer,
    _normalize_float,
    _write_line,
    close_logger,
    create_run_logger,
    log_event,
    log_ground_truth_tick,
    log_skipped_transmission,
    log_telemetry_row,
    log_transmission,
    wire_logger,
    write_summary,
    GROUND_TRUTH_HEADER,
    SCHEMA_VERSION,
    TELEMETRY_HEADER,
)
from herd_simulator.services.replay import normalize_telemetry_csv
from herd_simulator.utils.geo import haversine_m


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml"))


def _make_telemetry(animal_id=2, sim_second=1, **overrides):
    defaults = dict(
        animal_id=animal_id,
        is_physical=False,
        sim_second=sim_second,
        body_temp_c=38.612345,
        thi=75.123456,
        behaviour=0,
        latitude=12.971234,
        longitude=79.159234,
        risk_score=0,
        alert_band="green",
        geofence_status=0,
        battery_pct=99.987654,
        event_codes=[],
        dropped_out=False,
    )
    defaults.update(overrides)
    return AnimalTelemetry(**defaults)


# ===================================================================
# Buffered Writer
# ===================================================================

class TestBufferedWriter:

    def test_buffer_does_not_flush_below_threshold(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=100)
        for i in range(99):
            _write_line(w, f"line{i}")
        assert w.lines_written == 0
        assert len(w.buffer) == 99
        _close_writer(w)

    def test_buffer_flushes_at_threshold(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=10)
        for i in range(10):
            _write_line(w, f"line{i}")
        assert w.lines_written == 10
        assert len(w.buffer) == 0
        _close_writer(w)

    def test_flush_writes_partial_buffer(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=100)
        for i in range(50):
            _write_line(w, f"line{i}")
        _flush_writer(w)
        assert w.lines_written == 50
        assert len(w.buffer) == 0
        _close_writer(w)

    def test_close_flushes_remaining(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=100)
        for i in range(50):
            _write_line(w, f"line{i}")
        _close_writer(w)
        assert w.lines_written == 50
        lines = (tmp_path / "test.csv").read_text().strip().split("\n")
        assert len(lines) == 50

    def test_lines_written_counter_across_multiple_flushes(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=100)
        for i in range(250):
            _write_line(w, f"line{i}")
        assert w.lines_written == 200
        _close_writer(w)
        assert w.lines_written == 250

    def test_header_written_on_create(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=100, header="a,b,c")
        _close_writer(w)
        content = (tmp_path / "test.csv").read_text()
        assert content.startswith("a,b,c\n")

    def test_minimum_buffer_size_is_one(self, tmp_path):
        w = _new_buffered_writer(tmp_path / "test.csv", buffer_size=0)
        assert w.buffer_size == 1
        _close_writer(w)


# ===================================================================
# Run Logger Creation
# ===================================================================

class TestRunLoggerCreation:

    def test_creates_run_directory(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        assert rl.run_dir.exists()
        close_logger(rl)

    def test_manifest_json_fields(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles, scenario_id="test_sc")
        manifest = json.loads((rl.run_dir / "manifest.json").read_text())
        assert manifest["schema_version"] == SCHEMA_VERSION
        assert len(manifest["run_id"]) == 36  # UUID format
        assert manifest["mode"] == "dry-run"
        assert manifest["seed"] == 42
        assert manifest["herd_size"] == 20
        assert manifest["scenario_id"] == "test_sc"
        assert "start_time_iso" in manifest
        assert "config_hash" in manifest
        close_logger(rl)

    def test_config_snapshot_json(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        snapshot = json.loads((rl.run_dir / "config.snapshot.json").read_text())
        assert snapshot["seed"] == 42
        assert snapshot["herd"]["n_total"] == 20
        close_logger(rl)

    def test_config_hash_is_deterministic(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl1 = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        rl2 = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        assert rl1.manifest.config_hash == rl2.manifest.config_hash
        close_logger(rl1)
        close_logger(rl2)

    def test_animal_profiles_json_has_all_20(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        profiles = json.loads((rl.run_dir / "animal_profiles.json").read_text())
        assert len(profiles) == 20
        ids = [p["animal_id"] for p in profiles]
        assert ids == list(range(1, 21))
        close_logger(rl)

    def test_telemetry_csv_header(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        close_logger(rl)
        header = (rl.run_dir / "telemetry.csv").read_text().split("\n")[0]
        assert header == TELEMETRY_HEADER

    def test_ground_truth_csv_header(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        close_logger(rl)
        header = (rl.run_dir / "ground_truth_pairs.csv").read_text().split("\n")[0]
        assert header == GROUND_TRUTH_HEADER


# ===================================================================
# Telemetry CSV
# ===================================================================

class TestTelemetryLogging:

    def test_telemetry_row_format(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        t = _make_telemetry()
        log_telemetry_row(rl, t, ambient_temp_c=30.0, humidity_pct=65.0)
        close_logger(rl)
        with open(rl.run_dir / "telemetry.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        row = rows[0]
        assert row["animal_id"] == "2"
        assert row["risk_source"] == "RULE"
        assert int(row["schema_version"]) == SCHEMA_VERSION

    def test_telemetry_floats_normalized(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        t = _make_telemetry(body_temp_c=38.1234567890)
        log_telemetry_row(rl, t, ambient_temp_c=30.0, humidity_pct=65.0)
        close_logger(rl)
        with open(rl.run_dir / "telemetry.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["body_temp_c"] == "38.123457"

    def test_event_codes_pipe_separated(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        t = _make_telemetry(event_codes=["FEVER", "BREACH"])
        log_telemetry_row(rl, t, ambient_temp_c=30.0, humidity_pct=65.0)
        close_logger(rl)
        with open(rl.run_dir / "telemetry.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["event_codes"] == "BREACH|FEVER"

    def test_empty_event_codes(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        t = _make_telemetry(event_codes=[])
        log_telemetry_row(rl, t, ambient_temp_c=30.0, humidity_pct=65.0)
        close_logger(rl)
        with open(rl.run_dir / "telemetry.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["event_codes"] == ""

    def test_telemetry_counter_increments(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        for i in range(5):
            log_telemetry_row(rl, _make_telemetry(sim_second=i), 30.0, 65.0)
        assert rl.total_telemetry_rows == 5
        close_logger(rl)

    def test_disabled_telemetry_writes_nothing(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"), telemetry_csv_enabled=False)
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_telemetry_row(rl, _make_telemetry(), 30.0, 65.0)
        close_logger(rl)
        assert not (rl.run_dir / "telemetry.csv").exists()


# ===================================================================
# Events JSONL
# ===================================================================

class TestEventsLogging:

    def test_event_activated_jsonl(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_event(rl, "activated", "evt-1", 5, "fever_onset", 100,
                  params={"peak_offset_c": 1.8}, duration_seconds=900, source="scenario")
        close_logger(rl)
        line = (rl.run_dir / "events.jsonl").read_text().strip()
        record = json.loads(line)
        assert record["action"] == "activated"
        assert record["event_id"] == "evt-1"
        assert record["animal_id"] == 5
        assert record["event_type"] == "fever_onset"
        assert record["params"]["peak_offset_c"] == 1.8
        assert record["duration_seconds"] == 900
        assert record["source"] == "scenario"

    def test_event_expired_jsonl(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_event(rl, "expired", "evt-1", 5, "fever_onset", 1000)
        close_logger(rl)
        record = json.loads((rl.run_dir / "events.jsonl").read_text().strip())
        assert record["action"] == "expired"
        assert "params" not in record
        assert "duration_seconds" not in record

    def test_event_cleared_jsonl(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_event(rl, "cleared", "evt-1", 5, "fever_onset", 500)
        close_logger(rl)
        record = json.loads((rl.run_dir / "events.jsonl").read_text().strip())
        assert record["action"] == "cleared"

    def test_event_counters(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_event(rl, "activated", "e1", 5, "fever_onset", 100)
        log_event(rl, "activated", "e2", 6, "tamper", 100)
        log_event(rl, "expired", "e1", 5, "fever_onset", 200)
        log_event(rl, "cleared", "e2", 6, "tamper", 200)
        assert rl.total_events_activated == 2
        assert rl.total_events_expired == 1
        assert rl.total_events_cleared == 1
        close_logger(rl)

    def test_disabled_events_writes_nothing(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"), events_jsonl_enabled=False)
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_event(rl, "activated", "e1", 5, "fever_onset", 100)
        close_logger(rl)
        assert not (rl.run_dir / "events.jsonl").exists()


# ===================================================================
# Transmissions JSONL
# ===================================================================

class TestTransmissionsLogging:

    def test_transmission_jsonl_format(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        t = _make_telemetry()
        log_transmission(rl, t)
        close_logger(rl)
        record = json.loads((rl.run_dir / "transmissions.jsonl").read_text().strip())
        assert record["animal_id"] == 2
        assert record["sim_second"] == 1
        assert "alert_band" in record

    def test_transmission_counter(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        for _ in range(3):
            log_transmission(rl, _make_telemetry())
        assert rl.total_transmissions == 3
        close_logger(rl)

    def test_skipped_dropout_is_logged_without_counting_as_transmission(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        log_skipped_transmission(rl, _make_telemetry(dropped_out=True), "collar_dropout")
        close_logger(rl)

        record = json.loads((rl.run_dir / "transmissions.jsonl").read_text().strip())
        assert record == {
            "type": "skipped",
            "reason": "collar_dropout",
            "sim_second": 1,
            "animal_id": 2,
        }
        assert rl.total_transmissions == 0


# ===================================================================
# Ground Truth
# ===================================================================

class TestGroundTruth:

    def test_190_pairs_per_tick(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        telemetry = tick(sim)
        log_ground_truth_tick(rl, telemetry)
        close_logger(rl)
        with open(rl.run_dir / "ground_truth_pairs.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 190
        assert rl.total_ground_truth_pairs == 190

    def test_pair_ordering_deterministic(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        telemetry = tick(sim)
        log_ground_truth_tick(rl, telemetry)
        close_logger(rl)
        with open(rl.run_dir / "ground_truth_pairs.csv") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert int(row["animal_a_id"]) < int(row["animal_b_id"])

    def test_distances_match_haversine(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        telemetry = tick(sim)
        log_ground_truth_tick(rl, telemetry)
        close_logger(rl)
        with open(rl.run_dir / "ground_truth_pairs.csv") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        a_id = int(row["animal_a_id"])
        b_id = int(row["animal_b_id"])
        t_a = next(t for t in telemetry if t.animal_id == a_id)
        t_b = next(t for t in telemetry if t.animal_id == b_id)
        expected = haversine_m((t_a.latitude, t_a.longitude), (t_b.latitude, t_b.longitude))
        actual = float(row["distance_m"])
        assert abs(actual - round(expected, 6)) < 1e-6

    def test_anomaly_states_captured(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        activate_event(sim.event_state, 5, EventType.FEVER_ONSET, 0)
        telemetry = tick(sim)
        log_ground_truth_tick(rl, telemetry)
        close_logger(rl)
        with open(rl.run_dir / "ground_truth_pairs.csv") as f:
            reader = csv.DictReader(f)
            found_fever = False
            for row in reader:
                if int(row["animal_a_id"]) == 5:
                    if "FEVER" in row["animal_a_anomalies"]:
                        found_fever = True
                        break
                elif int(row["animal_b_id"]) == 5:
                    if "FEVER" in row["animal_b_anomalies"]:
                        found_fever = True
                        break
        assert found_fever

    def test_ground_truth_disabled_when_config_false(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"), ground_truth_enabled=False)
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        telemetry = tick(sim)
        log_ground_truth_tick(rl, telemetry)
        close_logger(rl)
        assert not (rl.run_dir / "ground_truth_pairs.csv").exists()
        assert rl.total_ground_truth_pairs == 0


# ===================================================================
# Summary
# ===================================================================

class TestSummary:

    def test_summary_json_fields(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        write_summary(rl, sim_second=100, total_writes=10, sweeps_completed=1)
        close_logger(rl)
        summary = json.loads((rl.run_dir / "summary.json").read_text())
        assert summary["schema_version"] == SCHEMA_VERSION
        assert summary["final_sim_second"] == 100
        assert summary["total_scheduler_writes"] == 10
        assert summary["total_scheduler_sweeps"] == 1

    def test_summary_counters_match(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        wire_logger(sim, rl)
        for _ in range(10):
            tick(sim)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes,
                       sim.scheduler_state.sweeps_completed)
        close_logger(rl)
        summary = json.loads((rl.run_dir / "summary.json").read_text())
        assert summary["total_ticks"] == 10
        assert summary["total_telemetry_rows"] == 200  # 20 animals * 10 ticks


# ===================================================================
# Wire Logger Integration
# ===================================================================

class TestWireLogger:

    def test_wire_logger_registers_all_callbacks(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        wire_logger(sim, rl)
        assert sim.on_telemetry is not None
        assert sim.on_transmit is not None
        assert sim.on_event_activated is not None
        assert sim.on_event_expired is not None
        assert sim.on_event_cleared is not None
        assert sim.on_tick_complete is not None
        close_logger(rl)

    def test_integration_10_ticks(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
        wire_logger(sim, rl)
        for _ in range(10):
            tick(sim)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes,
                       sim.scheduler_state.sweeps_completed)
        close_logger(rl)
        assert (rl.run_dir / "manifest.json").exists()
        assert (rl.run_dir / "config.snapshot.json").exists()
        assert (rl.run_dir / "animal_profiles.json").exists()
        assert (rl.run_dir / "telemetry.csv").exists()
        assert (rl.run_dir / "transmissions.jsonl").exists()
        assert (rl.run_dir / "ground_truth_pairs.csv").exists()
        assert (rl.run_dir / "summary.json").exists()
        with open(rl.run_dir / "telemetry.csv") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 200
        with open(rl.run_dir / "ground_truth_pairs.csv") as f:
            reader = csv.DictReader(f)
            gt_rows = list(reader)
        assert len(gt_rows) == 1900  # 190 pairs * 10 ticks

    def test_scenario_events_logged(self, cfg, tmp_path):
        cfg_mod = _cfg_with_log_dir(cfg, str(tmp_path / "logs"))
        doc = {
            "schema_version": 1,
            "scenario_id": "log_test",
            "seed": 42,
            "events": [
                {"animal_id": 5, "type": "tamper", "start_sim_second": 2,
                 "duration_seconds": 3},
            ],
        }
        path = tmp_path / "scenario.json"
        path.write_text(json.dumps(doc))
        scenario = load_scenario(path, valid_animal_ids=range(1, 21))
        sim = create_simulator(cfg_mod, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg_mod, "dry-run", sim.profiles, scenario_id="log_test")
        wire_logger(sim, rl)
        for _ in range(10):
            tick(sim)
        close_logger(rl)
        lines = (rl.run_dir / "events.jsonl").read_text().strip().split("\n")
        actions = [json.loads(l)["action"] for l in lines]
        assert "activated" in actions
        assert "expired" in actions


# ===================================================================
# Determinism
# ===================================================================

class TestDeterminism:

    def test_two_runs_same_seed_produce_identical_normalized_output(self, cfg, tmp_path):
        for run_name in ("run_a", "run_b"):
            log_dir = str(tmp_path / run_name)
            cfg_mod = _cfg_with_log_dir(cfg, log_dir)
            sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
            rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
            wire_logger(sim, rl)
            for _ in range(50):
                tick(sim)
            write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes,
                           sim.scheduler_state.sweeps_completed)
            close_logger(rl)

        run_a_dir = next(iter(sorted((tmp_path / "run_a").iterdir())))
        run_b_dir = next(iter(sorted((tmp_path / "run_b").iterdir())))

        norm_a = tmp_path / "norm_a.csv"
        norm_b = tmp_path / "norm_b.csv"
        normalize_telemetry_csv(run_a_dir / "telemetry.csv", norm_a)
        normalize_telemetry_csv(run_b_dir / "telemetry.csv", norm_b)
        assert norm_a.read_text() == norm_b.read_text()


# ===================================================================
# Helpers
# ===================================================================

def _cfg_with_log_dir(
    cfg,
    log_dir: str,
    ground_truth_enabled: bool = True,
    telemetry_csv_enabled: bool = True,
    events_jsonl_enabled: bool = True,
):
    """Return a copy of cfg with a modified logging section."""
    from herd_simulator.config import LoggingConfig
    new_logging = LoggingConfig(
        log_dir=log_dir,
        ground_truth_enabled=ground_truth_enabled,
        telemetry_csv_enabled=telemetry_csv_enabled,
        events_jsonl_enabled=events_jsonl_enabled,
        buffer_size=cfg.logging.buffer_size,
    )
    import dataclasses
    return dataclasses.replace(cfg, logging=new_logging)
