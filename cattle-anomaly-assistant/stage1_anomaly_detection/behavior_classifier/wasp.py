"""Strict, six-axis-only reader for the public WASP db-cow-walking data."""

from __future__ import annotations

import csv
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

from .errors import fail

DATASET_NAME: Final[str] = "WASP-lab/db-cow-walking"
DATASET_SOURCE_URL: Final[str] = "https://github.com/WASP-lab/db-cow-walking"
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "MPU9250_AX",
    "MPU9250_AY",
    "MPU9250_AZ",
    "MPU9250_GX",
    "MPU9250_GY",
    "MPU9250_GZ",
)
# Internal labels are contiguous because XGBoost's multiclass objective needs
# them. The safe platform mapping is exposed separately in ``CLASS_DETAILS``.
CLASS_DETAILS: Final[tuple[tuple[str, int], ...]] = (
    ("resting", 0),
    ("grazing", 1),
    ("walking", 3),
    ("miscellaneous", 5),
)
CLASS_NAMES: Final[tuple[str, ...]] = tuple(name for name, _ in CLASS_DETAILS)
PLATFORM_CODES: Final[dict[str, int]] = dict(CLASS_DETAILS)
_DIRECTORY_TO_CLASS: Final[dict[str, str]] = {
    "resting": "resting",
    "grazing": "grazing",
    "walking": "walking",
    "miscellaneous behaviors": "miscellaneous",
    "miscellaneous behaviours": "miscellaneous",
}
_FILENAME_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<event_id>[^_]+)_(?P<label>.+?)_(?P<cow_id>[^_]+)_\d{8}_\d{6}\.csv$"
)


@dataclass(frozen=True)
class WaspSample:
    timestamp: datetime
    values: tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class WaspEvent:
    event_id: str
    cow_id: str
    label: str
    samples: tuple[WaspSample, ...]


def _normalise(value: str) -> str:
    return " ".join(value.casefold().split())


def _parse_float(value: str | None, path: Path, row: int, column: str) -> float:
    try:
        parsed = float(value or "")
    except ValueError as exc:
        fail("WASP_LAYOUT_INVALID", "WASP IMU value is not numeric.", path=str(path), row=row, column=column)
        raise AssertionError from exc  # satisfies type checkers
    if not math.isfinite(parsed):
        fail("WASP_LAYOUT_INVALID", "WASP IMU value is not finite.", path=str(path), row=row, column=column)
    return parsed


def _parse_event(path: Path) -> WaspEvent:
    label = _DIRECTORY_TO_CLASS.get(_normalise(path.parent.name))
    if label is None:
        fail("WASP_LAYOUT_INVALID", "WASP CSV is in an unsupported label directory.", path=str(path))
    match = _FILENAME_RE.match(path.name)
    if match is None:
        fail(
            "WASP_LAYOUT_INVALID",
            "WASP filename must follow <event>_<label>_<cow>_<YYYYMMDD>_<HHMMSS>.csv.",
            path=str(path),
        )
    samples: list[WaspSample] = []
    prior: datetime | None = None
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or ())
        missing = [column for column in ("Time", *REQUIRED_COLUMNS) if column not in headers]
        if missing:
            fail("WASP_LAYOUT_INVALID", "WASP CSV is missing required MPU9250 columns.", path=str(path), missing=missing)
        for row_number, row in enumerate(reader, start=2):
            try:
                timestamp = datetime.fromisoformat((row.get("Time") or "").strip())
            except ValueError as exc:
                fail("WASP_LAYOUT_INVALID", "WASP timestamp is invalid.", path=str(path), row=row_number)
                raise AssertionError from exc
            if prior is not None and timestamp <= prior:
                fail("WASP_LAYOUT_INVALID", "WASP timestamps must be strictly increasing.", path=str(path), row=row_number)
            prior = timestamp
            values = tuple(_parse_float(row.get(column), path, row_number, column) for column in REQUIRED_COLUMNS)
            samples.append(WaspSample(timestamp=timestamp, values=values))
    if not samples:
        fail("WASP_LAYOUT_INVALID", "WASP CSV contains no samples.", path=str(path))
    return WaspEvent(match.group("event_id"), match.group("cow_id"), label, tuple(samples))


def discover_event_paths(dataset_dir: Path) -> tuple[Path, ...]:
    root = dataset_dir.expanduser().resolve()
    if not root.is_dir():
        fail("WASP_LAYOUT_INVALID", "--dataset-dir does not exist or is not a directory.", dataset_dir=str(root))
    paths = tuple(
        sorted(
            (path for path in root.rglob("*.csv") if _normalise(path.parent.name) in _DIRECTORY_TO_CLASS),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not paths:
        fail(
            "WASP_LAYOUT_INVALID",
            "No labelled WASP CSV files were found. Expected Resting, Grazing, Walking, and Miscellaneous behaviors folders.",
            dataset_dir=str(root),
        )
    return paths


def load_dataset(dataset_dir: Path) -> tuple[WaspEvent, ...]:
    events = tuple(_parse_event(path) for path in discover_event_paths(dataset_dir))
    labels = {event.label for event in events}
    missing = sorted(set(CLASS_NAMES) - labels)
    if missing:
        fail("INSUFFICIENT_CLASS_SUPPORT", "WASP dataset is missing required behavior classes.", missing_classes=missing)
    return events


def dataset_provenance(dataset_dir: Path) -> dict[str, object]:
    """Return only aggregate hash provenance, never raw rows or paths."""

    digest = hashlib.sha256()
    count = 0
    size = 0
    for path in discover_event_paths(dataset_dir):
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(path.relative_to(dataset_dir.expanduser().resolve()).as_posix().encode("utf-8"))
        digest.update(file_digest.encode("ascii"))
        count += 1
        size += path.stat().st_size
    return {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "source_url": DATASET_SOURCE_URL,
        "file_count": count,
        "total_bytes": size,
        "combined_sha256": digest.hexdigest(),
        "raw_sensor_rows_persisted": False,
    }
