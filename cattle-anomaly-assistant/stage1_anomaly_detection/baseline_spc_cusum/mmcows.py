"""Strict-but-portable CSV ingestion for the selected MmCows sensor streams.

MmCows publishes timestamps as Unix time and stores wearable data under
``main_data``.  This adapter supports the two useful CSV shapes encountered
when people stage only part of that archive: one file per tag, and a wide CSV
with T01...T10 columns.  It deliberately rejects ambiguous headers instead of
silently selecting a plausible-looking health signal.
"""

from __future__ import annotations

import csv
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from math import sqrt
from pathlib import Path
from statistics import median
from typing import Iterable, Iterator
from zoneinfo import ZoneInfo

from .config import DetectorConfig
from .errors import fail

# Most wearable exports use T01...T10; the upstream ankle benchmark uses
# C01...C10 directories.  Both identify the same animal, so normalise them
# to the public T01...T10 identity at the adapter boundary.
_TAG_PATTERN = re.compile(r"(?:^|[^A-Z0-9])[TC]?(0?[1-9]|1[0-4])(?:$|[^A-Z0-9])", re.IGNORECASE)
_TAG_COLUMN_PATTERN = re.compile(r"^[TC](0[1-9]|10|13|14)$", re.IGNORECASE)
_SHARED_COW_ID = "__shared__"
_ALLOWED_COWS = {f"T{number:02d}" for number in range(1, 11)}
_EXCLUDED_TAGS = {"T13", "T14"}

_TIMESTAMP_ALIASES = ("timestamp", "time", "ts", "unix_timestamp", "unix", "epoch")
_COW_ALIASES = ("cow_id", "cow", "tag_id", "tag", "device_id", "device", "id")
_VALUE_ALIASES = {
    "cbt": ("cbt", "core_body_temperature", "body_temperature", "temperature", "temperature_c", "temp", "value"),
    "thi": ("thi", "temperature_humidity_index", "temperaturehumidityindex", "value"),
    "ankle": ("lying", "lying_state", "is_lying", "posture", "orientation", "state", "value"),
}
_AXIS_ALIASES = {
    "x": ("ax", "acc_x", "acceleration_x", "accel_x_mps2", "x"),
    "y": ("ay", "acc_y", "acceleration_y", "accel_y_mps2", "y"),
    "z": ("az", "acc_z", "acceleration_z", "accel_z_mps2", "z"),
}


@dataclass(frozen=True)
class ScalarSample:
    timestamp: datetime
    value: float


@dataclass
class _CadenceTracker:
    """Bounded-memory cadence estimate from observed timestamps.

    The first 2,048 positive deltas are enough to identify a regular sensor
    cadence while avoiding a list of millions of 100ms IMMU samples.  The
    bound still permits deliberately sparse fixture/preflight exports.
    """

    last_timestamp: float | None = None
    deltas: list[float] = field(default_factory=list)

    def add(self, timestamp: datetime) -> None:
        current = timestamp.timestamp()
        if self.last_timestamp is not None:
            delta = current - self.last_timestamp
            if 0 < delta <= 86_400 and len(self.deltas) < 2048:
                self.deltas.append(delta)
        self.last_timestamp = current

    def value(self) -> float | None:
        return median(self.deltas) if self.deltas else None


@dataclass
class AccelerationDailyStats:
    """Three-axis Welford state; never retains raw acceleration samples."""

    count: int = 0
    mean_x: float = 0.0
    mean_y: float = 0.0
    mean_z: float = 0.0
    m2_x: float = 0.0
    m2_y: float = 0.0
    m2_z: float = 0.0

    def add(self, x: float, y: float, z: float) -> None:
        self.count += 1
        for axis, value in (("x", x), ("y", y), ("z", z)):
            mean_name = f"mean_{axis}"
            m2_name = f"m2_{axis}"
            prior_mean = getattr(self, mean_name)
            delta = value - prior_mean
            next_mean = prior_mean + delta / self.count
            setattr(self, mean_name, next_mean)
            setattr(self, m2_name, getattr(self, m2_name) + delta * (value - next_mean))

    def gravity_centered_rms(self) -> float | None:
        if self.count == 0:
            return None
        return sqrt((self.m2_x + self.m2_y + self.m2_z) / self.count)


@dataclass
class StreamData:
    """Samples grouped by cow; THI may use the internal shared identity."""

    samples: dict[str, list[ScalarSample]] = field(default_factory=lambda: defaultdict(list))
    files: list[Path] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)

    def cadence_seconds(self, cow_id: str) -> float | None:
        timestamps = sorted({sample.timestamp.timestamp() for sample in self.samples.get(cow_id, [])})
        if len(timestamps) < 2:
            return None
        deltas = [right - left for left, right in zip(timestamps, timestamps[1:]) if right > left]
        return median(deltas) if deltas else None


@dataclass
class AccelerationData:
    daily: dict[str, dict[date, AccelerationDailyStats]] = field(default_factory=lambda: defaultdict(dict))
    cadence: dict[str, _CadenceTracker] = field(default_factory=lambda: defaultdict(_CadenceTracker))
    files: list[Path] = field(default_factory=list)
    headers: dict[str, list[str]] = field(default_factory=dict)

    def cadence_seconds(self, cow_id: str) -> float | None:
        tracker = self.cadence.get(cow_id)
        return tracker.value() if tracker else None

    def add(self, cow_id: str, timestamp: datetime, x: float, y: float, z: float) -> None:
        stats = self.daily[cow_id].setdefault(timestamp.date(), AccelerationDailyStats())
        stats.add(x, y, z)
        self.cadence[cow_id].add(timestamp)


@dataclass(frozen=True)
class MmCowsInputs:
    cbt: StreamData
    ankle: StreamData
    thi: StreamData
    immu: AccelerationData | None
    root: Path

    @property
    def available_streams(self) -> list[str]:
        streams = ["cbt", "ankle", "thi"]
        if self.immu is not None:
            streams.append("immu")
        return streams


def _normalize_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _normalize_tag(value: str) -> str | None:
    match = _TAG_PATTERN.search(value.strip().upper())
    if not match:
        return None
    return f"T{int(match.group(1)):02d}"


def _tag_from_path(path: Path) -> str | None:
    for part in (path.stem, *reversed(path.parts[:-1])):
        tag = _normalize_tag(part)
        if tag:
            return tag
    return None


def _resolve_main_data_root(data_root: Path) -> Path:
    resolved = data_root.expanduser().resolve()
    if not resolved.is_dir():
        fail("DATA_ROOT_NOT_FOUND", "--data-root must be an extracted MmCows directory.", path=str(resolved))
    main_data = resolved / "main_data"
    return main_data if main_data.is_dir() else resolved


def _csv_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.rglob("*.csv") if path.is_file())


def _stream_files(main_data: Path, stream: str) -> list[Path]:
    if stream != "immu":
        return _csv_files(main_data / stream)

    # Accept both the early staged layout (immu/acceleration/T01.csv) and
    # the public MmCows layout (immu/T01/T01_0721.csv).  Prefer the explicit
    # acceleration child when it exists so files are never loaded twice.
    acceleration_directory = main_data / "immu" / "acceleration"
    acceleration_files = _csv_files(acceleration_directory)
    return acceleration_files or _csv_files(main_data / "immu")


def _match_column(
    headers: list[str],
    aliases: Iterable[str],
    override: str | None,
    *,
    stream: str,
    role: str,
    path: Path,
    required: bool = True,
) -> str | None:
    if override:
        if override not in headers:
            fail(
                "UNSUPPORTED_HEADERS",
                f"Configured {role} column is absent from a {stream} file.",
                stream=stream,
                role=role,
                path=str(path),
                configured_column=override,
                headers=headers,
            )
        return override
    normalized = {_normalize_header(header): header for header in headers}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    if required:
        fail(
            "UNSUPPORTED_HEADERS",
            f"Could not identify the {role} column for a {stream} CSV.",
            stream=stream,
            role=role,
            path=str(path),
            headers=headers,
            fix="Pass --config with columns.<stream> overrides for this export.",
        )
    return None


def _parse_timestamp(raw: str, *, path: Path, row_number: int, timezone: ZoneInfo) -> datetime:
    try:
        epoch = float(raw)
    except (TypeError, ValueError):
        fail(
            "INVALID_TIMESTAMP",
            "MmCows timestamps must be Unix timestamps.",
            path=str(path),
            row=row_number,
        )
    if epoch > 100_000_000_000:
        epoch /= 1000.0
    try:
        return datetime.fromtimestamp(epoch, tz=timezone)
    except (OverflowError, OSError, ValueError):
        fail("INVALID_TIMESTAMP", "The Unix timestamp is outside the supported range.", path=str(path), row=row_number)


def _parse_float(
    raw: str | None,
    *,
    path: Path,
    row_number: int,
    stream: str,
    nonfinite_is_missing: bool = False,
) -> float | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError:
        fail(
            "INVALID_NUMERIC_VALUE",
            f"A {stream} value is not numeric.",
            stream=stream,
            path=str(path),
            row=row_number,
        )
    if not math.isfinite(value):
        if nonfinite_is_missing:
            return None
        fail(
            "INVALID_NUMERIC_VALUE",
            f"A {stream} value must be finite.",
            stream=stream,
            path=str(path),
            row=row_number,
        )
    return value


def _normalise_ankle(raw: str | None, *, path: Path, row_number: int, config: DetectorConfig) -> float | None:
    if raw is None or not raw.strip():
        return None
    value = raw.strip().lower()
    if value in config.ankle_lying_values:
        return 1.0
    if value in config.ankle_standing_values:
        return 0.0
    try:
        numeric_value = float(value)
    except ValueError:
        numeric_value = None
    if numeric_value == 1.0:
        return 1.0
    if numeric_value == 0.0:
        return 0.0
    fail(
        "UNSUPPORTED_ANKLE_VALUE",
        "Ankle values must explicitly encode lying or standing; no orientation threshold is inferred.",
        path=str(path),
        row=row_number,
        accepted_lying=list(config.ankle_lying_values),
        accepted_standing=list(config.ankle_standing_values),
    )


def _reader(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        fail("INPUT_READ_ERROR", "Could not read a staged input CSV.", path=str(path), error=str(exc))
    reader = csv.DictReader(handle)
    headers = reader.fieldnames or []
    if not headers:
        handle.close()
        fail("EMPTY_CSV", "An input CSV has no header row.", path=str(path))

    def rows() -> Iterator[dict[str, str]]:
        try:
            yield from reader
        finally:
            handle.close()

    return headers, rows()


def _cow_from_row_or_path(row: dict[str, str], cow_column: str | None, path: Path) -> str | None:
    if cow_column:
        tag = _normalize_tag(row.get(cow_column, ""))
        if tag:
            return tag
    return _tag_from_path(path)


def _load_scalar_stream(main_data: Path, stream: str, config: DetectorConfig) -> StreamData:
    files = _stream_files(main_data, stream)
    if not files:
        fail(
            "MISSING_STREAM",
            f"The required MmCows '{stream}' CSV stream is missing.",
            stream=stream,
            expected_directory=str(main_data / stream),
        )
    result = StreamData(files=files)
    timezone = ZoneInfo(config.timezone)
    override = config.columns.get(stream, {})
    for path in files:
        headers, rows = _reader(path)
        result.headers[str(path)] = headers
        timestamp_column = _match_column(
            headers, _TIMESTAMP_ALIASES, override.get("timestamp"), stream=stream, role="timestamp", path=path
        )
        cow_column = _match_column(
            headers, _COW_ALIASES, override.get("cow_id"), stream=stream, role="cow ID", path=path, required=False
        )
        value_column = _match_column(
            headers, _VALUE_ALIASES[stream], override.get("value"), stream=stream, role="value", path=path, required=False
        )
        wide_columns = [header for header in headers if _TAG_COLUMN_PATTERN.fullmatch(header.strip())]
        if value_column is None and not wide_columns:
            fail(
                "UNSUPPORTED_HEADERS",
                f"Could not identify a {stream} value column or T01...T10 wide columns.",
                stream=stream,
                path=str(path),
                headers=headers,
                fix="Pass --config with columns.<stream>.value for this export.",
            )

        for row_number, row in enumerate(rows, start=2):
            raw_timestamp = row.get(timestamp_column)
            if raw_timestamp is None or not raw_timestamp.strip():
                continue
            timestamp = _parse_timestamp(raw_timestamp, path=path, row_number=row_number, timezone=timezone)
            if wide_columns and value_column is None:
                for header in wide_columns:
                    cow_id = _normalize_tag(header)
                    if cow_id not in _ALLOWED_COWS:
                        continue
                    value = _normalise_ankle(row.get(header), path=path, row_number=row_number, config=config) if stream == "ankle" else _parse_float(row.get(header), path=path, row_number=row_number, stream=stream)
                    if value is not None:
                        result.samples[cow_id].append(ScalarSample(timestamp=timestamp, value=value))
                continue

            cow_id = _cow_from_row_or_path(row, cow_column, path)
            if cow_id in _EXCLUDED_TAGS:
                continue
            if cow_id is None:
                if stream == "thi":
                    cow_id = _SHARED_COW_ID
                else:
                    fail(
                        "MISSING_COW_ID",
                        f"A {stream} CSV needs a T01...T10 filename/parent folder or cow ID column.",
                        stream=stream,
                        path=str(path),
                        headers=headers,
                    )
            if cow_id not in _ALLOWED_COWS and cow_id != _SHARED_COW_ID:
                continue
            raw_value = row.get(value_column) if value_column else None
            value = _normalise_ankle(raw_value, path=path, row_number=row_number, config=config) if stream == "ankle" else _parse_float(raw_value, path=path, row_number=row_number, stream=stream)
            if value is not None:
                result.samples[cow_id].append(ScalarSample(timestamp=timestamp, value=value))
    if not result.samples:
        fail("NO_USABLE_RECORDS", f"No usable {stream} records were found for T01...T10.", stream=stream)
    return result


def _load_acceleration(main_data: Path, config: DetectorConfig, *, required: bool) -> AccelerationData | None:
    files = _stream_files(main_data, "immu")
    if not files:
        if required:
            fail(
                "MISSING_IMMU",
                "IMMU acceleration was required but no CSV files were found.",
                expected_directories=[str(main_data / "immu" / "acceleration"), str(main_data / "immu" / "T01")],
            )
        return None
    result = AccelerationData(files=files)
    timezone = ZoneInfo(config.timezone)
    override = config.columns.get("immu", {})
    for path in files:
        headers, rows = _reader(path)
        result.headers[str(path)] = headers
        timestamp_column = _match_column(
            headers, _TIMESTAMP_ALIASES, override.get("timestamp"), stream="immu", role="timestamp", path=path
        )
        cow_column = _match_column(
            headers, _COW_ALIASES, override.get("cow_id"), stream="immu", role="cow ID", path=path, required=False
        )
        axes = {
            axis: _match_column(
                headers,
                aliases,
                override.get(axis),
                stream="immu",
                role=f"acceleration {axis}-axis",
                path=path,
            )
            for axis, aliases in _AXIS_ALIASES.items()
        }
        for row_number, row in enumerate(rows, start=2):
            raw_timestamp = row.get(timestamp_column)
            if raw_timestamp is None or not raw_timestamp.strip():
                continue
            cow_id = _cow_from_row_or_path(row, cow_column, path)
            if cow_id in _EXCLUDED_TAGS:
                continue
            if cow_id not in _ALLOWED_COWS:
                if cow_id is None:
                    fail(
                        "MISSING_COW_ID",
                        "An IMMU CSV needs a T01...T10 filename/parent folder or cow ID column.",
                        stream="immu",
                        path=str(path),
                        headers=headers,
                    )
                continue
            values = [
                _parse_float(
                    row.get(axes[axis]),
                    path=path,
                    row_number=row_number,
                    stream="immu",
                    nonfinite_is_missing=True,
                )
                for axis in ("x", "y", "z")
            ]
            if any(value is None for value in values):
                continue
            result.add(
                cow_id,
                _parse_timestamp(raw_timestamp, path=path, row_number=row_number, timezone=timezone),
                values[0],
                values[1],
                values[2],
            )
    if not result.daily:
        if required:
            fail("NO_USABLE_RECORDS", "No usable IMMU records were found for T01...T10.", stream="immu")
        return None
    return result


def load_mmcows_inputs(data_root: Path, config: DetectorConfig, *, require_immu: bool = False) -> MmCowsInputs:
    """Load only selected streams, retaining no input data outside this process."""

    main_data = _resolve_main_data_root(data_root)
    cbt = _load_scalar_stream(main_data, "cbt", config)
    ankle = _load_scalar_stream(main_data, "ankle", config)
    thi = _load_scalar_stream(main_data, "thi", config)
    immu = _load_acceleration(main_data, config, required=require_immu)
    common_cows = sorted(set(cbt.samples) & set(ankle.samples) & (_ALLOWED_COWS))
    if not common_cows:
        fail(
            "NO_COMMON_COWS",
            "No T01...T10 cow has both CBT and ankle records.",
            cbt_cows=sorted(cbt.samples),
            ankle_cows=sorted(ankle.samples),
        )
    return MmCowsInputs(cbt=cbt, ankle=ankle, thi=thi, immu=immu, root=main_data)
