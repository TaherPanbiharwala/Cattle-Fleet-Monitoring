"""
replay.py — Replay reader and normalization for determinism verification.

Reads telemetry.csv and manifest.json from a previous run directory.
Provides normalization functions that strip non-deterministic fields
(run_id, timestamps) and round floats to 6 decimal places, enabling
byte-identical comparison of two runs with the same config/seed/scenario.

References:
  Master PRD: "Normalized replay removes wall-clock timestamps and
  network-response metadata, sorts records by simulated second and
  animal ID, and rounds floating-point telemetry to six decimal places."
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from herd_simulator.services.logger import TELEMETRY_HEADER


@dataclass(frozen=True)
class ReplayRow:
    """One parsed row from telemetry.csv."""

    schema_version: int
    run_id: str
    sim_second: int
    animal_id: int
    is_physical: bool
    body_temp_c: float
    ambient_temp_c: float
    humidity_pct: float
    thi: float
    behaviour: int
    latitude: float
    longitude: float
    risk_score: int
    risk_source: str
    geofence_status: int
    battery_pct: float
    event_codes: list[str]


def _parse_row(row: dict[str, str]) -> ReplayRow:
    codes_raw = row["event_codes"].strip()
    return ReplayRow(
        schema_version=int(row["schema_version"]),
        run_id=row["run_id"],
        sim_second=int(row["sim_second"]),
        animal_id=int(row["animal_id"]),
        is_physical=bool(int(row["is_physical"])),
        body_temp_c=float(row["body_temp_c"]),
        ambient_temp_c=float(row["ambient_temp_c"]),
        humidity_pct=float(row["humidity_pct"]),
        thi=float(row["thi"]),
        behaviour=int(row["behaviour"]),
        latitude=float(row["latitude"]),
        longitude=float(row["longitude"]),
        risk_score=int(row["risk_score"]),
        risk_source=row["risk_source"],
        geofence_status=int(row["geofence_status"]),
        battery_pct=float(row["battery_pct"]),
        event_codes=codes_raw.split("|") if codes_raw else [],
    )


# -----------------------------------------------------------------------
# Readers
# -----------------------------------------------------------------------

def load_manifest(log_dir: Path) -> dict:
    """Read and parse manifest.json from a run directory."""
    path = log_dir / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_replay(log_dir: Path) -> Iterator[ReplayRow]:
    """Yield ReplayRow objects from telemetry.csv, sorted by (sim_second, animal_id)."""
    path = log_dir / "telemetry.csv"
    rows: list[ReplayRow] = []
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            rows.append(_parse_row(raw))
    rows.sort(key=lambda r: (r.sim_second, r.animal_id))
    yield from rows


# -----------------------------------------------------------------------
# Normalization for determinism verification
# -----------------------------------------------------------------------

_FLOAT_COLUMNS = {
    "body_temp_c", "ambient_temp_c", "humidity_pct", "thi",
    "latitude", "longitude", "battery_pct",
}


def normalize_telemetry_csv(input_path: Path, output_path: Path) -> None:
    """Create a normalized copy of telemetry.csv for determinism verification.

    Normalization:
    - Replaces run_id with "NORMALIZED"
    - Rounds all float columns to 6 decimal places
    - Sorts rows by (sim_second, animal_id)
    """
    with open(input_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    for row in rows:
        row["run_id"] = "NORMALIZED"
        for col in _FLOAT_COLUMNS:
            if col in row:
                row[col] = f"{round(float(row[col]), 6):.6f}"

    rows.sort(key=lambda r: (int(r["sim_second"]), int(r["animal_id"])))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_manifest(input_path: Path, output_path: Path) -> None:
    """Create a normalized manifest.json (strips run_id, timestamps, and config_hash)."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    data["run_id"] = "NORMALIZED"
    data["start_time_iso"] = "1970-01-01T00:00:00+00:00"
    data["config_hash"] = "NORMALIZED"
    output_path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
