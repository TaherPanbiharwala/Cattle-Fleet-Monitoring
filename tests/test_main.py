"""
test_main.py — Unit tests for the CLI entry point (src/main.py).

Tests:
  - Argument parser defaults and overrides
  - Mode string normalization
  - Configuration and seed override
  - Scenario file loading
  - Replay validation (--log-dir requirement)
  - .env loader functionality
"""

from __future__ import annotations

import os
import sys
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import yaml

from herd_simulator.config import load_config
from herd_simulator.env_loader import load_dotenv
from main import (
    _build_parser,
    _load_replay_config,
    _log_thingspeak_write_result,
    _replay_row_to_telemetry,
    main,
    run_replay,
    run_simulator,
)
from herd_simulator.services.replay import ReplayRow


class TestArgParser:
    """Test CLI argument parsing and defaults."""

    def test_default_arguments(self):
        parser = _build_parser()
        args = parser.parse_args([])
        assert args.mode == "offline"
        assert args.config == "config/default_config.yaml"
        assert args.scenario is None
        assert args.duration_hours is None
        assert args.hud is False
        assert args.port is None
        assert args.seed is None
        assert args.log_dir is None
        assert args.verbose is False
        assert args.playback_speed == 1.0

    def test_mode_choices(self):
        parser = _build_parser()
        for mode in ["dry-run", "dry_run", "offline", "live", "replay"]:
            args = parser.parse_args(["--mode", mode])
            assert args.mode == mode

    def test_custom_arguments(self):
        parser = _build_parser()
        args = parser.parse_args([
            "--mode", "dry-run",
            "--config", "custom_config.yaml",
            "--scenario", "config/scenarios/demo_scenario.json",
            "--duration-hours", "2.5",
            "--hud",
            "--port", "8888",
            "--seed", "123",
            "--log-dir", "/tmp/logs",
            "--playback-speed", "2.0",
            "--verbose",
        ])
        assert args.mode == "dry-run"
        assert args.config == "custom_config.yaml"
        assert args.scenario == "config/scenarios/demo_scenario.json"
        assert args.duration_hours == 2.5
        assert args.hud is True
        assert args.port == 8888
        assert args.seed == 123
        assert args.log_dir == "/tmp/logs"
        assert args.playback_speed == 2.0
        assert args.verbose is True


class TestEnvLoader:
    """Test standard library .env loader."""

    def test_load_dotenv_from_file(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text(
            "# Comment line\n"
            "TEST_VAR_A=hello\n"
            "TEST_VAR_B='quoted value'\n"
            "TEST_VAR_C=\"double quoted\"\n"
            "\n"
            "INVALID_LINE\n"
        )
        # Clear if already set
        for k in ["TEST_VAR_A", "TEST_VAR_B", "TEST_VAR_C"]:
            os.environ.pop(k, None)

        loaded = load_dotenv(env_file)
        assert loaded is True
        assert os.environ.get("TEST_VAR_A") == "hello"
        assert os.environ.get("TEST_VAR_B") == "quoted value"
        assert os.environ.get("TEST_VAR_C") == "double quoted"

        # Cleanup
        for k in ["TEST_VAR_A", "TEST_VAR_B", "TEST_VAR_C"]:
            os.environ.pop(k, None)

    def test_load_dotenv_missing_file(self, tmp_path):
        missing = tmp_path / "nonexistent.env"
        assert load_dotenv(missing) is False

    def test_load_dotenv_does_not_overwrite_existing(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("PRESET_VAR=from_file\n")
        os.environ["PRESET_VAR"] = "already_set"
        load_dotenv(env_file)
        assert os.environ.get("PRESET_VAR") == "already_set"
        os.environ.pop("PRESET_VAR", None)


class TestReplayRowConversion:
    """Test ReplayRow to AnimalTelemetry conversion."""

    def test_converts_all_fields_accurately(self):
        row = ReplayRow(
            schema_version=1,
            run_id="run-123",
            sim_second=100,
            animal_id=5,
            is_physical=False,
            body_temp_c=39.5,
            ambient_temp_c=28.0,
            humidity_pct=65.0,
            thi=75.0,
            behaviour=3,
            latitude=12.9715,
            longitude=79.1590,
            risk_score=75,
            risk_source="RULE",
            geofence_status=0,
            battery_pct=95.0,
            event_codes=["FEVER"],
        )
        t = _replay_row_to_telemetry(row)
        assert t.animal_id == 5
        assert t.is_physical is False
        assert t.sim_second == 100
        assert t.body_temp_c == 39.5
        assert t.thi == 75.0
        assert t.behaviour == 3
        assert t.latitude == 12.9715
        assert t.longitude == 79.1590
        assert t.risk_score == 75
        assert t.alert_band == "red"
        assert t.geofence_status == 0
        assert t.battery_pct == 95.0
        assert t.event_codes == ["FEVER"]
        assert t.dropped_out is False

    def test_alert_band_yellow(self):
        row = ReplayRow(
            schema_version=1, run_id="r", sim_second=1, animal_id=2, is_physical=False,
            body_temp_c=38.6, ambient_temp_c=28.0, humidity_pct=65.0, thi=75.0,
            behaviour=1, latitude=12.97, longitude=79.15, risk_score=50,
            risk_source="RULE", geofence_status=0, battery_pct=90.0, event_codes=[],
        )
        t = _replay_row_to_telemetry(row)
        assert t.alert_band == "yellow"

    def test_dropped_out_flag(self):
        row = ReplayRow(
            schema_version=1, run_id="r", sim_second=1, animal_id=2, is_physical=False,
            body_temp_c=38.6, ambient_temp_c=28.0, humidity_pct=65.0, thi=75.0,
            behaviour=0, latitude=12.97, longitude=79.15, risk_score=100,
            risk_source="RULE", geofence_status=0, battery_pct=0.0, event_codes=["DROPOUT"],
        )
        t = _replay_row_to_telemetry(row)
        assert t.dropped_out is True
        assert t.alert_band == "red"


class TestCLIRunner:
    """Test CLI execution functions."""

    def test_replay_mode_requires_log_dir(self):
        parser = _build_parser()
        args = parser.parse_args(["--mode", "replay"])
        assert run_replay(args) == 1

    def test_replay_mode_missing_dir(self, tmp_path):
        parser = _build_parser()
        args = parser.parse_args(["--mode", "replay", "--log-dir", str(tmp_path / "nonexistent")])
        assert run_replay(args) == 1

    def test_replay_prefers_saved_config_snapshot(self, tmp_path):
        snapshot = tmp_path / "config.snapshot.json"
        config_data = yaml.safe_load(Path("config/default_config.yaml").read_text())
        config_data["hud"]["port"] = 8765
        snapshot.write_text(json.dumps(config_data))

        cfg = _load_replay_config("does-not-exist.yaml", tmp_path)

        assert cfg.hud.port == 8765

    def test_write_result_logging_preserves_callback_argument_order(self):
        run_logger = MagicMock()
        with patch("main.log_write_result") as log_result:
            _log_thingspeak_write_result(run_logger, 7, 123, "success", 200, 1)

        log_result.assert_called_once_with(run_logger, 7, 123, "success", 200, 1)

    def test_invalid_config_path(self, tmp_path):
        parser = _build_parser()
        args = parser.parse_args(["--mode", "dry-run", "--config", str(tmp_path / "nonexistent.yaml")])
        assert run_simulator(args) == 1

    def test_invalid_scenario_path(self, tmp_path):
        parser = _build_parser()
        args = parser.parse_args([
            "--mode", "dry-run",
            "--config", "config/default_config.yaml",
            "--scenario", str(tmp_path / "nonexistent.json"),
        ])
        assert run_simulator(args) == 1

    def test_negative_duration(self):
        parser = _build_parser()
        args = parser.parse_args(["--mode", "dry-run", "--duration-hours", "-1.0"])
        assert run_simulator(args) == 1

    def test_dry_run_executes_successfully(self, tmp_path):
        exit_code = main([
            "--mode", "dry-run",
            "--config", "config/default_config.yaml",
            "--duration-hours", "0.005",  # 18 seconds
            "--log-dir", str(tmp_path),
        ])
        assert exit_code == 0
        # Verify run directory created
        run_dirs = list(tmp_path.glob("run_*"))
        assert len(run_dirs) == 1
        assert (run_dirs[0] / "manifest.json").exists()
        assert (run_dirs[0] / "summary.json").exists()
        assert (run_dirs[0] / "telemetry.csv").exists()

    def test_replay_mode_executes_successfully(self, tmp_path):
        # First generate a small run
        main([
            "--mode", "dry-run",
            "--config", "config/default_config.yaml",
            "--duration-hours", "0.005",  # 18s
            "--log-dir", str(tmp_path),
        ])
        run_dir = list(tmp_path.glob("run_*"))[0]

        # Now replay it without HUD (console summary)
        exit_code = main([
            "--mode", "replay",
            "--log-dir", str(run_dir),
        ])
        assert exit_code == 0
