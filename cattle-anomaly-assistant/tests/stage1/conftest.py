from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest


CENTRAL = ZoneInfo("America/Chicago")


def _rows_for_day(day_index: int, value: float, *, start: datetime) -> list[dict[str, str]]:
    rows = []
    for hour in (0, 6, 12, 18):
        timestamp = start + timedelta(days=day_index, hours=hour)
        rows.append({"timestamp": str(int(timestamp.timestamp())), "value": str(value)})
    return rows


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def make_mmcows_subset(
    root: Path,
    *,
    include_immu: bool = True,
    immu_layout: str = "acceleration_child",
    days: int = 10,
    missing_cbt_day: int | None = None,
    constant_cbt: bool = False,
    include_stationary: bool = False,
) -> Path:
    """Write a small per-tag + shared-THI fixture matching the public layout."""

    start = datetime(2023, 7, 22, tzinfo=CENTRAL)
    main_data = root / "main_data"
    cbt_rows: list[dict[str, str]] = []
    ankle_rows: list[dict[str, str]] = []
    thi_rows: list[dict[str, str]] = []
    accel_rows: list[dict[str, str]] = []
    baseline_cbt = [38.0, 38.1, 38.2, 38.3, 38.4, 38.5, 38.6]
    # 3/8...5/8 keeps the baseline around 50% while leaving room for both
    # +/-2.5 sigma lying-time injections inside the physical 0-100% range.
    lying_samples = (
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
    )
    for day_index in range(days):
        cbt_value = 38.3 if day_index >= 7 else baseline_cbt[day_index]
        if constant_cbt:
            cbt_value = 38.4
        day_cbt_rows = _rows_for_day(day_index, cbt_value, start=start)
        if missing_cbt_day == day_index:
            day_cbt_rows = day_cbt_rows[:2]
        cbt_rows.extend(day_cbt_rows)
        pattern = lying_samples[day_index] if day_index < 7 else [1, 1, 1, 1, 0, 0, 0, 0]
        for sample_index, hour in enumerate(range(0, 24, 3)):
            timestamp = start + timedelta(days=day_index, hours=hour)
            ankle_rows.append({"timestamp": str(int(timestamp.timestamp())), "value": str(pattern[sample_index])})
        for sample_index, hour in enumerate((0, 6, 12, 18)):
            timestamp = start + timedelta(days=day_index, hours=hour)
            thi_rows.append({"timestamp": str(int(timestamp.timestamp())), "value": "72.0"})
            if include_immu:
                amplitude = 1.0 + (day_index % 3) * 0.1 if day_index < 7 else sum(1.0 + (index % 3) * 0.1 for index in range(7)) / 7
                vectors = ((amplitude, 0.0, 1.0), (-amplitude, 0.0, 1.0), (0.0, amplitude, 1.0), (0.0, -amplitude, 1.0))
                x, y, z = vectors[sample_index]
                accel_rows.append({"timestamp": str(int(timestamp.timestamp())), "ax": str(x), "ay": str(y), "az": str(z)})
    _write_csv(main_data / "cbt" / "T01.csv", ["timestamp", "temperature_C"], [
        {"timestamp": row["timestamp"], "temperature_C": row["value"]} for row in cbt_rows
    ])
    _write_csv(main_data / "ankle" / "C01" / "C01_0722.csv", ["timestamp", "value"], ankle_rows)
    _write_csv(main_data / "thi" / "ambient.csv", ["timestamp", "value"], thi_rows)
    if include_immu:
        if immu_layout == "acceleration_child":
            _write_csv(main_data / "immu" / "acceleration" / "T01.csv", ["timestamp", "ax", "ay", "az"], accel_rows)
        elif immu_layout == "public_per_tag":
            _write_csv(
                main_data / "immu" / "T01" / "T01_0722.csv",
                ["timestamp", "accel_x_mps2", "accel_y_mps2", "accel_z_mps2"],
                [
                    {
                        "timestamp": row["timestamp"],
                        "accel_x_mps2": row["ax"],
                        "accel_y_mps2": row["ay"],
                        "accel_z_mps2": row["az"],
                    }
                    for row in accel_rows
                ],
            )
        else:
            raise ValueError(f"Unknown IMMU fixture layout: {immu_layout}")
    if include_stationary:
        stationary_rows = _rows_for_day(0, 99.0, start=start)
        _write_csv(main_data / "cbt" / "T13.csv", ["timestamp", "value"], stationary_rows)
        _write_csv(main_data / "ankle" / "T13.csv", ["timestamp", "value"], stationary_rows)
    return root


@pytest.fixture
def mmcows_subset(tmp_path: Path) -> Path:
    return make_mmcows_subset(tmp_path / "mmcows", include_stationary=True)
