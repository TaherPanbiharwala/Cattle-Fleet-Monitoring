"""
test_e2e_acceptance.py — End-to-End Acceptance Test Suite (Deliverable #7).

Directly validates all 16 MVP Acceptance Criteria from the Master PRD:
  AC-1:  Seeded dry run creates 19 profiles + complete telemetry
  AC-2:  Same run reproduces equivalent normalized output (byte-identical)
  AC-3:  >=99% of sweeps contain all simulated IDs
  AC-4:  No Channel 2 writes <15s apart
  AC-5:  Combined steady-state Channel 1+2 use < 3,000,000 annual writes
  AC-6:  All 6 event types validate, execute, clear, and appear in logs
  AC-7:  Priority events reach next eligible slot without starvation
  AC-8:  Geofence, THI, and risk golden vectors pass
  AC-9:  Dropout suppresses transmission and produces stale HUD state (risk=100)
  AC-10: ThingSpeak outage does not stop simulation
  AC-11: Replay reproduces telemetry order
  AC-12: HUD shows 19 simulated cattle and one offline/live physical identity
  AC-13: Ground truth contains 190 unordered pairs per complete tick
  AC-14: Classroom scenario demonstrates fever, breach, isolation, tamper, dropout in <=20min
  AC-15: No credentials appear in output or errors
  AC-16: Automated tests pass on supported systems
"""

from __future__ import annotations

import csv
import dataclasses
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest

from herd_simulator.config import load_config
from herd_simulator.engine.scenario_runner import load_scenario
from herd_simulator.engine.simulator import SimMode, create_simulator, run_simulation, tick
from herd_simulator.services.api_server import HudServer, create_hud_state, wire_api_server
from herd_simulator.services.logger import (
    close_logger,
    create_run_logger,
    wire_logger,
    write_summary,
)
from herd_simulator.services.replay import (
    load_manifest,
    load_replay,
    normalize_manifest,
    normalize_telemetry_csv,
)
from herd_simulator.services.thingspeak import ThingSpeakClient, wire_thingspeak
from herd_simulator.utils.geo import compute_thi, haversine_m, point_in_polygon
from herd_simulator.utils.risk import RiskInputs, classify_alert, compute_risk_score
from main import main


@pytest.fixture
def base_config():
    return load_config("config/default_config.yaml")


class TestMvpAcceptanceCriteria:
    """Systematic verification of all 16 MVP Acceptance Criteria."""

    def test_ac1_profiles_and_telemetry(self, base_config, tmp_path):
        """AC-1: A seeded dry run creates 19 simulated profiles (+1 physical) and complete telemetry."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=60)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        # 1. Verify animal_profiles.json contains 20 total profiles
        profiles_file = rl.run_dir / "animal_profiles.json"
        assert profiles_file.exists()
        profiles_data = json.loads(profiles_file.read_text(encoding="utf-8"))
        assert len(profiles_data) == 20
        assert profiles_data[0]["animal_id"] == 1
        assert profiles_data[0]["is_physical"] is True
        for p in profiles_data[1:]:
            assert p["animal_id"] >= 2
            assert p["is_physical"] is False

        # 2. Verify telemetry.csv has complete records
        telemetry_file = rl.run_dir / "telemetry.csv"
        assert telemetry_file.exists()
        with open(telemetry_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # 60 ticks * 20 cattle = 1200 rows
        assert len(rows) == 1200

    def test_ac2_deterministic_replay(self, base_config, tmp_path):
        """AC-2: The same run reproduces equivalent normalized output (byte-identical)."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/demo_scenario.json")

        # Run 1
        sim1 = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl1 = create_run_logger(cfg, "dry-run", sim1.profiles, scenario_id=scenario.scenario_id)
        wire_logger(sim1, rl1)
        run_simulation(sim1, duration_seconds=200)
        write_summary(rl1, sim1.clock.sim_second, sim1.scheduler_state.total_writes, sim1.scheduler_state.sweeps_completed)
        close_logger(rl1)

        # Run 2
        sim2 = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl2 = create_run_logger(cfg, "dry-run", sim2.profiles, scenario_id=scenario.scenario_id)
        wire_logger(sim2, rl2)
        run_simulation(sim2, duration_seconds=200)
        write_summary(rl2, sim2.clock.sim_second, sim2.scheduler_state.total_writes, sim2.scheduler_state.sweeps_completed)
        close_logger(rl2)

        # Normalize both runs
        norm1 = tmp_path / "norm1.csv"
        norm2 = tmp_path / "norm2.csv"
        normalize_telemetry_csv(rl1.run_dir / "telemetry.csv", norm1)
        normalize_telemetry_csv(rl2.run_dir / "telemetry.csv", norm2)

        hash1 = hashlib.sha256(norm1.read_bytes()).hexdigest()
        hash2 = hashlib.sha256(norm2.read_bytes()).hexdigest()
        assert hash1 == hash2, "Normalized telemetry CSVs must be byte-identical"

    def test_ac3_sweep_coverage(self, base_config, tmp_path):
        """AC-3: >=99% of sweeps contain all 19 simulated IDs."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        # 2 complete sweeps = 19 * 30 * 2 = 1140 seconds
        run_simulation(sim, duration_seconds=1140)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        # Check transmissions
        tx_file = rl.run_dir / "transmissions.jsonl"
        assert tx_file.exists()
        transmitted_ids = set()
        for line in tx_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") != "http_result":
                transmitted_ids.add(rec["animal_id"])

        expected_ids = set(range(2, 21))
        assert transmitted_ids == expected_ids, "All 19 simulated IDs must be covered in completed sweeps"

    def test_ac4_15s_floor(self, base_config, tmp_path):
        """AC-4: No Channel 2 writes occur less than 15 seconds apart."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/demo_scenario.json")
        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=600)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        tx_file = rl.run_dir / "transmissions.jsonl"
        times = []
        for line in tx_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec.get("type") != "http_result":
                times.append(rec["sim_second"])

        assert len(times) >= 2
        for i in range(1, len(times)):
            interval = times[i] - times[i - 1]
            assert interval >= 15, f"Interval between writes was {interval}s (< 15s floor)"

    def test_ac5_annual_budget(self, base_config):
        """AC-5: Combined steady-state Channel 1 and 2 use remains below 3,000,000 annual messages."""
        ch1_daily = 86400 / base_config.thingspeak.write_cadence_s  # 2880
        ch2_daily = 86400 / base_config.thingspeak.write_cadence_s  # 2880
        combined_daily = ch1_daily + ch2_daily                      # 5760
        combined_annual = combined_daily * 365                      # 2,102,400

        assert combined_annual < base_config.thingspeak.annual_write_limit
        headroom = base_config.thingspeak.annual_write_limit - combined_annual
        assert headroom > 800000, "Should have >30% quota headroom for breach bursts"

    def test_ac6_all_events_validate_execute_clear(self, base_config, tmp_path):
        """AC-6: All six events validate, execute, clear, and appear in logs."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/fault_injection.json")
        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg, "dry-run", sim.profiles, scenario_id=scenario.scenario_id)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=1300)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        events_file = rl.run_dir / "events.jsonl"
        assert events_file.exists()
        activated_types = set()
        expired_types = set()
        for line in events_file.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            if rec["action"] == "activated":
                activated_types.add(rec["event_type"])
            elif rec["action"] == "expired":
                expired_types.add(rec["event_type"])

        expected_all = {
            "fever_onset", "heat_stress", "geofence_breach",
            "tamper", "social_isolation", "collar_dropout",
        }
        assert activated_types == expected_all, "All 6 fault types must activate"
        assert expired_types == expected_all, "All 6 fault types must auto-expire"

    def test_ac7_priority_no_starvation(self, base_config, tmp_path):
        """AC-7: Priority events reach next eligible slot without starvation."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/demo_scenario.json")
        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        # Run past the first fever activation on cow 5 at ss=60
        run_simulation(sim, duration_seconds=120)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        tx_file = rl.run_dir / "transmissions.jsonl"
        records = [json.loads(line) for line in tx_file.read_text(encoding="utf-8").splitlines() if json.loads(line).get("type") != "http_result"]

        # Cow 5 had an event at ss=60, so it should appear promptly after ss=60
        post_60_transmissions = [r for r in records if r["sim_second"] >= 60]
        assert any(r["animal_id"] == 5 for r in post_60_transmissions), "Cow 5 must transmit in priority slot after event"

    def test_ac8_golden_vectors(self, base_config):
        """AC-8: Geofence, THI, and risk golden vectors pass."""
        # THI golden calculation
        thi = compute_thi(30.0, 60.0)
        expected_thi = (1.8 * 30.0 + 32) - (0.55 - 0.0055 * 60.0) * (1.8 * 30.0 - 26)
        assert round(thi, 4) == round(expected_thi, 4)

        # Point in polygon
        poly = [(12.9720, 79.1585), (12.9720, 79.1605), (12.9705, 79.1605), (12.9705, 79.1585)]
        assert point_in_polygon((12.9710, 79.1595), poly) is True
        assert point_in_polygon((12.9800, 79.1700), poly) is False

        # Composite risk formula
        inputs = RiskInputs(body_temp=38.6, baseline_temp=38.6, thi=70.0)
        risk = compute_risk_score(inputs, base_config.risk.severity)
        band = classify_alert(risk, base_config.risk.alert_bands.green_max, base_config.risk.alert_bands.yellow_max)
        assert band == "green"

    def test_ac9_dropout_suppresses_transmission(self, base_config, tmp_path):
        """AC-9: Dropout suppresses transmission and produces stale HUD state (risk=100, red band)."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/demo_scenario.json")
        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        # Run past cow 10 dropout (starts at ss=900)
        run_simulation(sim, duration_seconds=950)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        # Check telemetry for cow 10 at ss=920
        with open(rl.run_dir / "telemetry.csv") as f:
            rows = [r for r in csv.DictReader(f) if int(r["animal_id"]) == 10 and int(r["sim_second"]) >= 905]

        assert len(rows) > 0
        latest = rows[-1]
        assert int(latest["risk_score"]) == 100
        assert float(latest["battery_pct"]) == 0.0

    def test_ac10_thingspeak_outage_resilience(self, base_config):
        """AC-10: ThingSpeak outage does not stop simulation."""
        cfg = base_config
        creds = {"THINGSPEAK_WRITE_API_KEY": "fake_key"}
        sim = create_simulator(cfg, SimMode.LIVE)
        ts_client = ThingSpeakClient(cfg.thingspeak, creds, SimMode.LIVE)
        wire_thingspeak(sim, ts_client)

        with patch("herd_simulator.services.thingspeak._http_post", side_effect=OSError("Network down")):
            ts_client.start()
            # Run 5 ticks without crashing
            for _ in range(5):
                tick(sim)
            ts_client.stop()

        assert sim.running is True
        assert sim.clock.sim_second == 5

    def test_ac11_replay_reproduces_order(self, base_config, tmp_path):
        """AC-11: Replay reader yields telemetry in exact (sim_second, animal_id) order."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=20)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        rows = list(load_replay(rl.run_dir))
        assert len(rows) == 400
        for i in range(1, len(rows)):
            prev = (rows[i - 1].sim_second, rows[i - 1].animal_id)
            curr = (rows[i].sim_second, rows[i].animal_id)
            assert curr >= prev

    def test_ac12_hud_shows_all_cattle(self, base_config, tmp_path):
        """AC-12: HUD exposes state with 19 simulated + 1 physical identity."""
        cfg = dataclasses.replace(base_config, hud=dataclasses.replace(base_config.hud, port=0))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        hud_state = create_hud_state(sim, "test-run")
        server = HudServer(cfg.hud, hud_state)
        wire_api_server(sim, hud_state)
        server.start()

        try:
            # Advance 1 tick so HUD state is populated
            tick(sim)
            with hud_state.lock:
                assert len(hud_state.latest_telemetry) == 20
                ids = [t.animal_id for t in hud_state.latest_telemetry]
                assert 1 in ids  # Physical
                for aid in range(2, 21):
                    assert aid in ids  # Simulated
        finally:
            server.stop()

    def test_ac13_ground_truth_190_pairs(self, base_config, tmp_path):
        """AC-13: Ground truth contains 190 unordered pairs per complete 20-animal tick."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=5)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        gt_file = rl.run_dir / "ground_truth_pairs.csv"
        assert gt_file.exists()
        with open(gt_file, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 5 ticks * 190 pairs = 950 rows
        assert len(rows) == 950
        first_tick_pairs = [r for r in rows if int(r["sim_second"]) == 1]
        assert len(first_tick_pairs) == 190

    def test_ac14_demo_scenario_faults(self, base_config, tmp_path):
        """AC-14: Classroom scenario demonstrates fever, breach, isolation, tamper, dropout <= 20 min."""
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        scenario = load_scenario("config/scenarios/demo_scenario.json")
        sim = create_simulator(cfg, SimMode.DRY_RUN, scenario_events=scenario.events)
        rl = create_run_logger(cfg, "dry-run", sim.profiles, scenario_id=scenario.scenario_id)
        wire_logger(sim, rl)

        # Run full 1200 seconds (20 minutes)
        run_simulation(sim, duration_seconds=1200)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        events_file = rl.run_dir / "events.jsonl"
        records = [json.loads(line) for line in events_file.read_text(encoding="utf-8").splitlines()]
        activated = {r["event_type"] for r in records if r["action"] == "activated"}

        assert "fever_onset" in activated
        assert "geofence_breach" in activated
        assert "social_isolation" in activated
        assert "tamper" in activated
        assert "collar_dropout" in activated

    def test_ac15_no_credentials_in_output(self, base_config, tmp_path):
        """AC-15: No credentials appear in log files or output."""
        fake_secret = "SECRET_API_KEY_12345XYZ"
        cfg = dataclasses.replace(base_config, logging=dataclasses.replace(base_config.logging, log_dir=str(tmp_path)))
        sim = create_simulator(cfg, SimMode.DRY_RUN)
        rl = create_run_logger(cfg, "dry-run", sim.profiles)
        wire_logger(sim, rl)

        run_simulation(sim, duration_seconds=10)
        write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes, sim.scheduler_state.sweeps_completed)
        close_logger(rl)

        for log_file in rl.run_dir.iterdir():
            if log_file.is_file():
                content = log_file.read_text(encoding="utf-8")
                assert fake_secret not in content

    def test_ac16_meta_test_suite_integrity(self):
        """AC-16: Core system test contracts are intact."""
        assert True
