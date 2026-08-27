"""
test_replay.py — Tests for replay reader and normalization.

Covers:
  - Manifest loading
  - Replay iterator (row types, ordering, event code parsing)
  - Normalization (float rounding, row sorting, run_id replacement)
  - End-to-end determinism verification via normalize + compare
"""

from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import load_config
from herd_simulator.engine.simulator import SimMode, create_simulator, tick
from herd_simulator.services.logger import (
    close_logger,
    create_run_logger,
    wire_logger,
    write_summary,
    TELEMETRY_HEADER,
)
from herd_simulator.services.replay import (
    ReplayRow,
    load_manifest,
    load_replay,
    normalize_manifest,
    normalize_telemetry_csv,
)


@pytest.fixture
def cfg():
    return load_config(os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml"))


def _cfg_with_log_dir(cfg, log_dir: str):
    from herd_simulator.config import LoggingConfig
    import dataclasses
    new_logging = LoggingConfig(
        log_dir=log_dir,
        ground_truth_enabled=True,
        telemetry_csv_enabled=True,
        events_jsonl_enabled=True,
        buffer_size=cfg.logging.buffer_size,
    )
    return dataclasses.replace(cfg, logging=new_logging)


def _run_simulation(cfg, tmp_path, run_name, n_ticks=10):
    log_dir = str(tmp_path / run_name)
    cfg_mod = _cfg_with_log_dir(cfg, log_dir)
    sim = create_simulator(cfg_mod, SimMode.DRY_RUN)
    rl = create_run_logger(cfg_mod, "dry-run", sim.profiles)
    wire_logger(sim, rl)
    for _ in range(n_ticks):
        tick(sim)
    write_summary(rl, sim.clock.sim_second, sim.scheduler_state.total_writes,
                   sim.scheduler_state.sweeps_completed)
    close_logger(rl)
    return next(iter(sorted((tmp_path / run_name).iterdir())))


# ===================================================================
# Load Manifest
# ===================================================================

class TestLoadManifest:

    def test_load_manifest_parses_correctly(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1")
        manifest = load_manifest(run_dir)
        assert manifest["schema_version"] == 1
        assert manifest["seed"] == 42
        assert len(manifest["run_id"]) == 36

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_manifest(tmp_path / "nonexistent")


# ===================================================================
# Load Replay
# ===================================================================

class TestLoadReplay:

    def test_replay_yields_all_rows(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=5)
        rows = list(load_replay(run_dir))
        assert len(rows) == 100  # 5 ticks * 20 animals

    def test_replay_rows_sorted(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=5)
        rows = list(load_replay(run_dir))
        for i in range(len(rows) - 1):
            key_a = (rows[i].sim_second, rows[i].animal_id)
            key_b = (rows[i + 1].sim_second, rows[i + 1].animal_id)
            assert key_a <= key_b

    def test_replay_row_types(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=1)
        row = next(load_replay(run_dir))
        assert isinstance(row, ReplayRow)
        assert isinstance(row.sim_second, int)
        assert isinstance(row.animal_id, int)
        assert isinstance(row.body_temp_c, float)
        assert isinstance(row.is_physical, bool)
        assert isinstance(row.risk_score, int)
        assert isinstance(row.event_codes, list)

    def test_replay_empty_event_codes(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=1)
        rows = list(load_replay(run_dir))
        healthy_row = next(r for r in rows if not r.event_codes or r.event_codes == [])
        assert healthy_row.event_codes == []


# ===================================================================
# Normalization
# ===================================================================

class TestNormalization:

    def test_normalize_rounds_floats(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=1)
        output = tmp_path / "normalized.csv"
        normalize_telemetry_csv(run_dir / "telemetry.csv", output)
        with open(output) as f:
            lines = f.readlines()
        data_line = lines[1].strip().split(",")
        body_temp = data_line[5]
        parts = body_temp.split(".")
        assert len(parts) == 2
        assert len(parts[1]) == 6

    def test_normalize_replaces_run_id(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=1)
        output = tmp_path / "normalized.csv"
        normalize_telemetry_csv(run_dir / "telemetry.csv", output)
        with open(output) as f:
            lines = f.readlines()
        data_line = lines[1].strip().split(",")
        assert data_line[1] == "NORMALIZED"

    def test_normalize_sorts_rows(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=3)
        output = tmp_path / "normalized.csv"
        normalize_telemetry_csv(run_dir / "telemetry.csv", output)
        import csv as csvmod
        with open(output) as f:
            reader = csvmod.DictReader(f)
            rows = list(reader)
        for i in range(len(rows) - 1):
            key_a = (int(rows[i]["sim_second"]), int(rows[i]["animal_id"]))
            key_b = (int(rows[i + 1]["sim_second"]), int(rows[i + 1]["animal_id"]))
            assert key_a <= key_b

    def test_normalize_manifest_strips_timestamps(self, cfg, tmp_path):
        run_dir = _run_simulation(cfg, tmp_path, "run1", n_ticks=1)
        output = tmp_path / "normalized_manifest.json"
        normalize_manifest(run_dir / "manifest.json", output)
        data = json.loads(output.read_text())
        assert data["run_id"] == "NORMALIZED"
        assert data["start_time_iso"] == "1970-01-01T00:00:00+00:00"


# ===================================================================
# End-to-End Determinism
# ===================================================================

class TestDeterministicReplay:

    def test_end_to_end_determinism(self, cfg, tmp_path):
        run_a_dir = _run_simulation(cfg, tmp_path, "run_a", n_ticks=50)
        run_b_dir = _run_simulation(cfg, tmp_path, "run_b", n_ticks=50)

        norm_a = tmp_path / "norm_a.csv"
        norm_b = tmp_path / "norm_b.csv"
        normalize_telemetry_csv(run_a_dir / "telemetry.csv", norm_a)
        normalize_telemetry_csv(run_b_dir / "telemetry.csv", norm_b)
        assert norm_a.read_text() == norm_b.read_text()

        norm_ma = tmp_path / "norm_ma.json"
        norm_mb = tmp_path / "norm_mb.json"
        normalize_manifest(run_a_dir / "manifest.json", norm_ma)
        normalize_manifest(run_b_dir / "manifest.json", norm_mb)
        assert norm_ma.read_text() == norm_mb.read_text()

        gt_a = (run_a_dir / "ground_truth_pairs.csv").read_text()
        gt_b = (run_b_dir / "ground_truth_pairs.csv").read_text()
        assert gt_a == gt_b
