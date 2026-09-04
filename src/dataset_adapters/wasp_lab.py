"""Strict adapter for the WASP-lab free-grazing cow IMU dataset.

The adapter deliberately reads only the MPU9250 accelerometer and gyroscope
channels.  They are the closest available source channels to the project's
future MPU6050 data path.  It returns real labelled-cow data only; simulator
telemetry is never accepted here.
"""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

DATASET_SOURCE_URL: Final[str] = "https://github.com/WASP-lab/db-cow-walking"

REQUIRED_MPU9250_COLUMNS: Final[tuple[str, ...]] = (
    "MPU9250_AX",
    "MPU9250_AY",
    "MPU9250_AZ",
    "MPU9250_GX",
    "MPU9250_GY",
    "MPU9250_GZ",
)

# The directory name, rather than a free-text event label, is the authoritative
# behaviour class.  This protects the stable platform behaviour-code contract.
_DIRECTORY_LABELS: Final[dict[str, tuple[int, str]]] = {
    "resting": (0, "Resting"),
    "grazing": (1, "Grazing"),
    "walking": (3, "Walking"),
    "miscellaneous behaviors": (5, "Miscellaneous"),
    "miscellaneous behaviours": (5, "Miscellaneous"),
}
_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<event_id>[^_]+)_(?P<event_label>.+?)_(?P<cow_id>[^_]+)_"
    r"(?P<date>\d{8})_(?P<time>\d{6})\.csv$"
)


class DatasetValidationError(ValueError):
    """Raised when a raw dataset file cannot support a safe benchmark run."""


@dataclass(frozen=True)
class ImuSample:
    """One timestamped six-axis IMU observation from a real WASP event."""

    timestamp: datetime
    accel_x: float
    accel_y: float
    accel_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float

    def as_six_axes(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.accel_x,
            self.accel_y,
            self.accel_z,
            self.gyro_x,
            self.gyro_y,
            self.gyro_z,
        )


@dataclass(frozen=True)
class WaspEvent:
    """A single labelled event; windows must never cross this boundary."""

    event_id: str
    cow_id: str
    behaviour_code: int
    behaviour_name: str
    source_path: Path
    samples: tuple[ImuSample, ...]


def _normalise_directory_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _parse_metadata(path: Path) -> tuple[str, str]:
    match = _FILENAME_RE.match(path.name)
    if match is None:
        raise DatasetValidationError(
            f"Invalid WASP filename '{path.name}'; expected "
            "<event_id>_<behaviour>_<cow_id>_<YYYYMMDD>_<HHMMSS>.csv."
        )
    return match.group("event_id"), match.group("cow_id")


def _parse_timestamp(raw_value: str, path: Path, row_number: int) -> datetime:
    try:
        return datetime.fromisoformat(raw_value.strip())
    except ValueError as exc:
        raise DatasetValidationError(
            f"{path}:{row_number} has invalid Time value '{raw_value}'."
        ) from exc


def _parse_float(raw_value: str | None, column: str, path: Path, row_number: int) -> float:
    if raw_value is None:
        raise DatasetValidationError(f"{path}:{row_number} is missing '{column}'.")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise DatasetValidationError(
            f"{path}:{row_number} has non-numeric '{column}' value '{raw_value}'."
        ) from exc
    if not math.isfinite(value):
        raise DatasetValidationError(f"{path}:{row_number} has non-finite '{column}' value.")
    return value


def load_wasp_event(path: Path) -> WaspEvent:
    """Load and validate one WASP event CSV without resampling it."""

    if not path.is_file() or path.suffix.casefold() != ".csv":
        raise DatasetValidationError(f"WASP event path is not a CSV file: {path}")

    label = _DIRECTORY_LABELS.get(_normalise_directory_name(path.parent.name))
    if label is None:
        expected = ", ".join(sorted(_DIRECTORY_LABELS))
        raise DatasetValidationError(
            f"Unsupported WASP label directory '{path.parent.name}'. Expected one of: {expected}."
        )

    event_id, cow_id = _parse_metadata(path)
    samples: list[ImuSample] = []
    previous_timestamp: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        headers = tuple(reader.fieldnames or ())
        missing = ("Time", *REQUIRED_MPU9250_COLUMNS)
        missing = tuple(column for column in missing if column not in headers)
        if missing:
            raise DatasetValidationError(f"{path} is missing required columns: {', '.join(missing)}.")

        for row_number, row in enumerate(reader, start=2):
            timestamp = _parse_timestamp(row.get("Time", ""), path, row_number)
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise DatasetValidationError(
                    f"{path}:{row_number} is not strictly later than the previous timestamp."
                )
            previous_timestamp = timestamp
            values = [
                _parse_float(row.get(column), column, path, row_number)
                for column in REQUIRED_MPU9250_COLUMNS
            ]
            samples.append(ImuSample(timestamp, *values))

    if not samples:
        raise DatasetValidationError(f"WASP event file contains no samples: {path}")

    return WaspEvent(
        event_id=event_id,
        cow_id=cow_id,
        behaviour_code=label[0],
        behaviour_name=label[1],
        source_path=path,
        samples=tuple(samples),
    )


def discover_wasp_event_paths(dataset_dir: Path) -> tuple[Path, ...]:
    """Return only known-label CSVs in stable relative-path order."""

    if not dataset_dir.is_dir():
        raise DatasetValidationError(f"WASP dataset directory does not exist: {dataset_dir}")

    paths = tuple(
        sorted(
            (
                path
                for path in dataset_dir.rglob("*.csv")
                if _normalise_directory_name(path.parent.name) in _DIRECTORY_LABELS
            ),
            key=lambda path: path.relative_to(dataset_dir).as_posix(),
        )
    )
    if not paths:
        raise DatasetValidationError(
            f"No labelled WASP CSV files found below {dataset_dir}. "
            "Expected Resting, Grazing, Walking, and Miscellaneous behaviors folders."
        )
    return paths


def load_wasp_dataset(dataset_dir: Path) -> tuple[WaspEvent, ...]:
    """Load all known labelled events and ensure every benchmark class exists."""

    events = tuple(load_wasp_event(path) for path in discover_wasp_event_paths(dataset_dir))
    observed_codes = {event.behaviour_code for event in events}
    required_codes = {0, 1, 3, 5}
    missing_codes = sorted(required_codes - observed_codes)
    if missing_codes:
        raise DatasetValidationError(
            f"WASP dataset is missing required mapped behaviour codes: {missing_codes}."
        )
    return events


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_dataset_manifest(dataset_dir: Path) -> dict[str, object]:
    """Create hash-only provenance; it never stores any raw IMU samples."""

    paths = discover_wasp_event_paths(dataset_dir)
    files = [
        {
            "path": path.relative_to(dataset_dir).as_posix(),
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    combined = hashlib.sha256()
    for entry in files:
        combined.update(str(entry["path"]).encode("utf-8"))
        combined.update(str(entry["sha256"]).encode("ascii"))
    return {
        "schema_version": 1,
        "dataset": "WASP-lab/db-cow-walking",
        "source_url": DATASET_SOURCE_URL,
        "file_count": len(files),
        "combined_sha256": combined.hexdigest(),
        "files": files,
        "raw_samples_embedded": False,
    }
