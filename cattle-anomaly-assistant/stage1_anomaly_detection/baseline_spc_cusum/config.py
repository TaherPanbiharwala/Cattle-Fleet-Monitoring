"""Configuration for the conservative daily MmCows detector."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .errors import fail


@dataclass(frozen=True)
class DetectorConfig:
    """Parameters deliberately kept separate from the raw dataset layout."""

    schema_version: int = 1
    timezone: str = "America/Chicago"
    baseline_days: int = 7
    coverage_minimum: float = 0.75
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    # Header overrides are intentionally narrow.  They allow a locally
    # exported CSV to name fields differently without adding a parser that
    # guesses at health-related columns.
    columns: dict[str, dict[str, str]] = field(default_factory=dict)
    ankle_lying_values: tuple[str, ...] = ("1", "true", "yes", "lying", "lie")
    ankle_standing_values: tuple[str, ...] = ("0", "false", "no", "standing", "stand")

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ankle_lying_values"] = list(self.ankle_lying_values)
        result["ankle_standing_values"] = list(self.ankle_standing_values)
        return result


def _expect_mapping(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail("INVALID_CONFIG", f"'{field_name}' must be a JSON object.", field=field_name)
    return value


def load_config(path: Path | None) -> DetectorConfig:
    """Load optional JSON config and reject unknown or unsafe values."""

    if path is None:
        return DetectorConfig()
    if not path.is_file():
        fail("CONFIG_NOT_FOUND", "The detector config file does not exist.", path=str(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail("INVALID_CONFIG", "The detector config must be valid JSON.", path=str(path), error=str(exc))
    values = _expect_mapping(payload, field_name="config")
    allowed = set(DetectorConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        fail("INVALID_CONFIG", "The config contains unsupported settings.", fields=unknown)

    if "columns" in values:
        columns = _expect_mapping(values["columns"], field_name="columns")
        for stream, override in columns.items():
            if stream not in {"cbt", "ankle", "thi", "immu"} or not isinstance(override, dict):
                fail("INVALID_CONFIG", "Each columns override must target a known stream.", stream=stream)
            if not all(isinstance(key, str) and isinstance(value, str) for key, value in override.items()):
                fail("INVALID_CONFIG", "Column override names must be strings.", stream=stream)

    for field_name in ("ankle_lying_values", "ankle_standing_values"):
        if field_name in values:
            raw_values = values[field_name]
            if not isinstance(raw_values, list) or not raw_values or not all(isinstance(value, str) for value in raw_values):
                fail("INVALID_CONFIG", f"'{field_name}' must be a non-empty string list.", field=field_name)
            values[field_name] = tuple(value.lower() for value in raw_values)

    try:
        config = DetectorConfig(**values)
    except TypeError as exc:
        fail("INVALID_CONFIG", "The config has an invalid setting type.", error=str(exc))
    if config.schema_version != 1:
        fail("INVALID_CONFIG", "Only detector config schema_version 1 is supported.", schema_version=config.schema_version)
    if config.timezone != "America/Chicago":
        fail("INVALID_CONFIG", "MmCows daily windows must use America/Chicago.", timezone=config.timezone)
    if config.baseline_days < 2:
        fail("INVALID_CONFIG", "baseline_days must be at least 2 for a sample standard deviation.")
    if not 0 < config.coverage_minimum <= 1:
        fail("INVALID_CONFIG", "coverage_minimum must be in (0, 1].")
    if config.cusum_k < 0 or config.cusum_h <= 0:
        fail("INVALID_CONFIG", "cusum_k must be non-negative and cusum_h must be positive.")
    return config
