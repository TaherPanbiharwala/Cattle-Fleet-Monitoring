"""Deterministic 112-feature extraction for labelled six-axis IMU windows."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    import numpy as np

from dataset_adapters.wasp_lab import WaspEvent

SAMPLE_RATE_HZ: Final[float] = 10.0
WINDOW_SECONDS: Final[float] = 5.0
WINDOW_SAMPLES: Final[int] = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)
WINDOW_STRIDE_SAMPLES: Final[int] = WINDOW_SAMPLES // 2
MIN_VALID_SAMPLES: Final[int] = 45
TIMESTAMP_TOLERANCE_SECONDS: Final[float] = 0.02

SIGNAL_NAMES: Final[tuple[str, ...]] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "accel_magnitude",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "gyro_magnitude",
)
FEATURE_BASE_NAMES: Final[tuple[str, ...]] = (
    "mean",
    "median",
    "std",
    "zero_crossing_rate",
    "peak_to_peak",
    "sum",
    "absolute_sum",
    "rms",
    "average_acceleration_variation",
    "skewness",
    "kurtosis",
    "dominant_frequency_hz",
    "dominant_spectral_density",
    "average_spectral_density",
)
FEATURE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{signal}__{feature}" for signal in SIGNAL_NAMES for feature in FEATURE_BASE_NAMES
)


def _np():
    try:
        import numpy as numpy
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("Phase 3 feature extraction requires the 'ml' optional dependencies.") from exc
    return numpy


@dataclass(frozen=True)
class WindowRecord:
    """A labelled source-bounded window suitable for grouped splitting."""

    cow_id: str
    event_id: str
    behaviour_code: int
    start_time: datetime
    samples: "np.ndarray"


@dataclass(frozen=True)
class FeatureDataset:
    """Features and source-group identities; raw data is not written to artifacts."""

    features: "np.ndarray"
    raw_windows: "np.ndarray"
    labels: "np.ndarray"
    cow_ids: "np.ndarray"
    event_ids: "np.ndarray"
    window_start_times: tuple[datetime, ...]


def _is_expected_interval(previous: datetime, current: datetime) -> bool:
    expected = 1.0 / SAMPLE_RATE_HZ
    observed = (current - previous).total_seconds()
    return abs(observed - expected) <= TIMESTAMP_TOLERANCE_SECONDS


def _contiguous_sample_segments(event: WaspEvent) -> tuple[tuple[datetime, tuple[tuple[float, ...], ...]], ...]:
    """Split an event at each timing gap; no model window crosses a split."""

    if not event.samples:
        return ()

    segments: list[tuple[datetime, tuple[tuple[float, ...], ...]]] = []
    start_time = event.samples[0].timestamp
    values: list[tuple[float, ...]] = [event.samples[0].as_six_axes()]
    previous = event.samples[0]
    for sample in event.samples[1:]:
        if not _is_expected_interval(previous.timestamp, sample.timestamp):
            segments.append((start_time, tuple(values)))
            start_time = sample.timestamp
            values = []
        values.append(sample.as_six_axes())
        previous = sample
    segments.append((start_time, tuple(values)))
    return tuple(segments)


def segment_event(event: WaspEvent) -> tuple[WindowRecord, ...]:
    """Create exactly-5-second windows with 50% overlap inside contiguous data."""

    numpy = _np()
    windows: list[WindowRecord] = []
    for segment_start, rows in _contiguous_sample_segments(event):
        if len(rows) < WINDOW_SAMPLES:
            continue
        values = numpy.asarray(rows, dtype=numpy.float64)
        if not numpy.isfinite(values).all():
            continue
        for start_index in range(0, len(rows) - WINDOW_SAMPLES + 1, WINDOW_STRIDE_SAMPLES):
            window = values[start_index : start_index + WINDOW_SAMPLES]
            if window.shape[0] < MIN_VALID_SAMPLES:
                continue
            windows.append(
                WindowRecord(
                    cow_id=event.cow_id,
                    event_id=event.event_id,
                    behaviour_code=event.behaviour_code,
                    start_time=segment_start + timedelta(seconds=start_index / SAMPLE_RATE_HZ),
                    samples=window,
                )
            )
    return tuple(windows)


def _signal_matrix(samples: "np.ndarray") -> "np.ndarray":
    numpy = _np()
    if samples.ndim != 2 or samples.shape[1] != 6:
        raise ValueError("IMU window must have shape (samples, 6).")
    if samples.shape[0] < MIN_VALID_SAMPLES:
        raise ValueError(f"IMU window needs at least {MIN_VALID_SAMPLES} valid samples.")
    if not numpy.isfinite(samples).all():
        raise ValueError("IMU window contains non-finite values.")
    accel_magnitude = numpy.linalg.norm(samples[:, :3], axis=1)
    gyro_magnitude = numpy.linalg.norm(samples[:, 3:], axis=1)
    return numpy.column_stack((samples[:, :3], accel_magnitude, samples[:, 3:], gyro_magnitude))


def _spectral_features(values: "np.ndarray") -> tuple[float, float, float]:
    numpy = _np()
    centered = values - numpy.mean(values)
    spectrum = numpy.fft.rfft(centered)
    frequencies = numpy.fft.rfftfreq(values.size, d=1.0 / SAMPLE_RATE_HZ)
    if spectrum.size <= 1:
        return 0.0, 0.0, 0.0
    density = (numpy.abs(spectrum) ** 2) / values.size
    positive_density = density[1:]
    positive_frequencies = frequencies[1:]
    dominant_index = int(numpy.argmax(positive_density))
    return (
        float(positive_frequencies[dominant_index]),
        float(positive_density[dominant_index]),
        float(numpy.mean(positive_density)),
    )


def _feature_vector_for_signal(values: "np.ndarray") -> tuple[float, ...]:
    numpy = _np()
    mean = float(numpy.mean(values))
    std = float(numpy.std(values))
    if std == 0.0:
        skewness = 0.0
        kurtosis = 0.0
    else:
        standardised = (values - mean) / std
        skewness = float(numpy.mean(standardised**3))
        kurtosis = float(numpy.mean(standardised**4) - 3.0)
    sign_changes = numpy.count_nonzero(numpy.signbit(values[1:]) != numpy.signbit(values[:-1]))
    zero_crossing_rate = float(sign_changes / max(values.size - 1, 1))
    dominant_frequency, dominant_density, average_density = _spectral_features(values)
    return (
        mean,
        float(numpy.median(values)),
        std,
        zero_crossing_rate,
        float(numpy.ptp(values)),
        float(numpy.sum(values)),
        float(numpy.sum(numpy.abs(values))),
        float(numpy.sqrt(numpy.mean(values**2))),
        float(numpy.mean(numpy.abs(numpy.diff(values)))) if values.size > 1 else 0.0,
        skewness,
        kurtosis,
        dominant_frequency,
        dominant_density,
        average_density,
    )


def extract_window_features(samples: "np.ndarray") -> "np.ndarray":
    """Return the documented 14 features for each of eight IMU signals."""

    numpy = _np()
    signals = _signal_matrix(samples)
    result = numpy.asarray(
        [value for index in range(signals.shape[1]) for value in _feature_vector_for_signal(signals[:, index])],
        dtype=numpy.float64,
    )
    if result.shape != (len(FEATURE_NAMES),):
        raise RuntimeError("Feature extraction did not produce the required 112 features.")
    return result


def build_feature_dataset(events: tuple[WaspEvent, ...]) -> FeatureDataset:
    """Turn validated events into a deterministic feature table and raw CNN inputs."""

    numpy = _np()
    records = tuple(record for event in events for record in segment_event(event))
    if not records:
        raise ValueError("No valid 5-second windows could be created from the WASP dataset.")
    return FeatureDataset(
        features=numpy.vstack([extract_window_features(record.samples) for record in records]),
        raw_windows=numpy.stack([record.samples for record in records]),
        labels=numpy.asarray([record.behaviour_code for record in records], dtype=numpy.int64),
        cow_ids=numpy.asarray([record.cow_id for record in records], dtype=str),
        event_ids=numpy.asarray([record.event_id for record in records], dtype=str),
        window_start_times=tuple(record.start_time for record in records),
    )
