"""Deterministic, six-axis 112-feature extraction for five-second IMU windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Final

from .errors import fail
from .wasp import WaspEvent

SAMPLE_RATE_HZ: Final[float] = 10.0
WINDOW_SECONDS: Final[float] = 5.0
WINDOW_SAMPLES: Final[int] = 50
WINDOW_STRIDE_SAMPLES: Final[int] = 25
TIMESTAMP_TOLERANCE_SECONDS: Final[float] = 0.02
SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "accel_x", "accel_y", "accel_z", "accel_magnitude", "gyro_x", "gyro_y", "gyro_z", "gyro_magnitude"
)
FEATURE_BASE_NAMES: Final[tuple[str, ...]] = (
    "mean", "median", "std", "zero_crossing_rate", "peak_to_peak", "sum", "absolute_sum", "rms",
    "average_acceleration_variation", "skewness", "kurtosis", "dominant_frequency_hz",
    "dominant_spectral_density", "average_spectral_density",
)
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(f"{signal}__{feature}" for signal in SIGNAL_NAMES for feature in FEATURE_BASE_NAMES)


def _np() -> Any:
    try:
        import numpy
    except ImportError as exc:  # pragma: no cover - optional-install boundary
        fail("DEPENDENCY_MISSING", "Behavior classification requires the behavior extra.", install="python -m pip install -e '.[behavior,dev]'")
        raise AssertionError from exc
    return numpy


@dataclass(frozen=True)
class Window:
    cow_id: str
    event_id: str
    label: str
    start_time: datetime
    samples: Any


@dataclass(frozen=True)
class FeatureDataset:
    features: Any
    labels: Any
    cow_ids: Any
    event_ids: Any
    start_times: tuple[datetime, ...]
    labels_by_index: tuple[str, ...]
    gap_segments_discarded: int


def _expected_gap(previous: datetime, current: datetime) -> bool:
    return abs((current - previous).total_seconds() - (1 / SAMPLE_RATE_HZ)) <= TIMESTAMP_TOLERANCE_SECONDS


def windows_for_event(event: WaspEvent) -> tuple[tuple[Window, ...], int]:
    """Split at timing gaps. A 50-sample window never crosses an event or gap."""

    numpy = _np()
    if not event.samples:
        return (), 0
    segments: list[list[tuple[float, ...]]] = [[event.samples[0].values]]
    starts = [event.samples[0].timestamp]
    for earlier, current in zip(event.samples, event.samples[1:], strict=False):
        if not _expected_gap(earlier.timestamp, current.timestamp):
            segments.append([])
            starts.append(current.timestamp)
        segments[-1].append(current.values)
    output: list[Window] = []
    discarded = 0
    for start, rows in zip(starts, segments, strict=True):
        if len(rows) < WINDOW_SAMPLES:
            discarded += 1
            continue
        matrix = numpy.asarray(rows, dtype=numpy.float64)
        if not numpy.isfinite(matrix).all():
            discarded += 1
            continue
        for offset in range(0, len(rows) - WINDOW_SAMPLES + 1, WINDOW_STRIDE_SAMPLES):
            output.append(Window(event.cow_id, event.event_id, event.label, start + timedelta(seconds=offset / SAMPLE_RATE_HZ), matrix[offset : offset + WINDOW_SAMPLES]))
    return tuple(output), discarded


def _spectral(values: Any) -> tuple[float, float, float]:
    numpy = _np()
    density = numpy.abs(numpy.fft.rfft(values - numpy.mean(values))) ** 2 / values.size
    frequencies = numpy.fft.rfftfreq(values.size, d=1 / SAMPLE_RATE_HZ)
    if density.size <= 1:
        return 0.0, 0.0, 0.0
    index = int(numpy.argmax(density[1:])) + 1
    return float(frequencies[index]), float(density[index]), float(numpy.mean(density[1:]))


def _signal_features(values: Any) -> tuple[float, ...]:
    numpy = _np()
    mean = float(numpy.mean(values))
    std = float(numpy.std(values))
    standardised = (values - mean) / std if std else numpy.zeros_like(values)
    crossings = numpy.count_nonzero(numpy.signbit(values[:-1]) != numpy.signbit(values[1:]))
    dominant_frequency, dominant_density, average_density = _spectral(values)
    return (
        mean, float(numpy.median(values)), std, float(crossings / max(values.size - 1, 1)), float(numpy.ptp(values)),
        float(numpy.sum(values)), float(numpy.sum(numpy.abs(values))), float(numpy.sqrt(numpy.mean(values ** 2))),
        float(numpy.mean(numpy.abs(numpy.diff(values)))), float(numpy.mean(standardised ** 3)),
        float(numpy.mean(standardised ** 4) - 3.0), dominant_frequency, dominant_density, average_density,
    )


def extract_features(samples: Any) -> Any:
    """Return 112 finite derived features for exactly one 50x6 window."""

    numpy = _np()
    matrix = numpy.asarray(samples, dtype=numpy.float64)
    if matrix.shape != (WINDOW_SAMPLES, 6) or not numpy.isfinite(matrix).all():
        fail("FEATURE_CONTRACT", "A behavior window must contain exactly 50 finite six-axis samples.", expected_shape=[WINDOW_SAMPLES, 6], actual_shape=list(matrix.shape))
    signals = numpy.column_stack((matrix[:, :3], numpy.linalg.norm(matrix[:, :3], axis=1), matrix[:, 3:], numpy.linalg.norm(matrix[:, 3:], axis=1)))
    result = numpy.asarray([value for index in range(signals.shape[1]) for value in _signal_features(signals[:, index])], dtype=numpy.float64)
    if result.shape != (112,) or not numpy.isfinite(result).all():
        fail("FEATURE_CONTRACT", "Feature extraction did not produce 112 finite values.")
    return result


def build_feature_dataset(events: tuple[WaspEvent, ...]) -> FeatureDataset:
    numpy = _np()
    all_windows: list[Window] = []
    discarded = 0
    for event in events:
        windows, skipped = windows_for_event(event)
        all_windows.extend(windows)
        discarded += skipped
    if not all_windows:
        fail("INSUFFICIENT_CLASS_SUPPORT", "No valid five-second behavior windows were produced.")
    labels = tuple(window.label for window in all_windows)
    missing = sorted({"resting", "grazing", "walking", "miscellaneous"} - set(labels))
    if missing:
        fail("INSUFFICIENT_CLASS_SUPPORT", "Timing and window validation removed a required class.", missing_classes=missing)
    return FeatureDataset(
        features=numpy.vstack([extract_features(window.samples) for window in all_windows]),
        labels=numpy.asarray([("resting", "grazing", "walking", "miscellaneous").index(label) for label in labels], dtype=numpy.int64),
        cow_ids=numpy.asarray([window.cow_id for window in all_windows], dtype=str),
        event_ids=numpy.asarray([window.event_id for window in all_windows], dtype=str),
        start_times=tuple(window.start_time for window in all_windows),
        labels_by_index=("resting", "grazing", "walking", "miscellaneous"),
        gap_segments_discarded=discarded,
    )
