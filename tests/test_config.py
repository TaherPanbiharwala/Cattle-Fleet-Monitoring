"""
test_config.py — Unit tests for config.py's fail-fast YAML validation.

Focused on the risk.severity relationship guard added during code review:
risk.py divides by (temp_offset_high - temp_offset_low) and
(thi_high - thi_low) with no zero-guard, so config.py must reject a
misconfigured or swapped pair before the tick loop can ever see it.
"""

from __future__ import annotations

import os
import sys

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import ConfigError, load_config

_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")


def _load_and_mutate(tmp_path, mutate):
    """Load the shipped default config, apply `mutate` to the raw dict,
    write it out, and return the path — for testing one bad value at a
    time against an otherwise-valid config."""
    with open(_DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    mutate(raw)
    path = tmp_path / "mutated.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(raw, f)
    return path


class TestSeverityRelationshipGuard:
    def test_default_config_loads_cleanly(self):
        """Sanity check: the shipped config satisfies its own guard."""
        cfg = load_config(_DEFAULT_CONFIG_PATH)
        assert cfg.risk.severity.temp_offset_high > cfg.risk.severity.temp_offset_low
        assert cfg.risk.severity.thi_high > cfg.risk.severity.thi_low

    def test_equal_temp_offset_rejected(self, tmp_path):
        path = _load_and_mutate(
            tmp_path,
            lambda raw: raw["risk"]["severity"].__setitem__("temp_offset_high", raw["risk"]["severity"]["temp_offset_low"]),
        )
        with pytest.raises(ConfigError, match="temp_offset_high"):
            load_config(path)

    def test_swapped_temp_offsets_rejected(self, tmp_path):
        path = _load_and_mutate(
            tmp_path,
            lambda raw: raw["risk"]["severity"].update(
                temp_offset_low=2.0, temp_offset_high=0.5,
            ),
        )
        with pytest.raises(ConfigError, match="temp_offset_high"):
            load_config(path)

    def test_equal_thi_thresholds_rejected(self, tmp_path):
        path = _load_and_mutate(
            tmp_path,
            lambda raw: raw["risk"]["severity"].__setitem__("thi_high", raw["risk"]["severity"]["thi_low"]),
        )
        with pytest.raises(ConfigError, match="thi_high"):
            load_config(path)

    def test_swapped_thi_thresholds_rejected(self, tmp_path):
        path = _load_and_mutate(
            tmp_path,
            lambda raw: raw["risk"]["severity"].update(thi_low=84, thi_high=68),
        )
        with pytest.raises(ConfigError, match="thi_high"):
            load_config(path)
