"""
logger.py — Per-run structured logging service (Deliverable #4).

Produces 8 output files per run:
  manifest.json, config.snapshot.json, animal_profiles.json,
  telemetry.csv, events.jsonl, transmissions.jsonl,
  ground_truth_pairs.csv, summary.json

All file writers use buffered I/O (configurable via LoggingConfig.buffer_size).
Ground truth computes C(20,2)=190 pairwise Haversine distances per tick.

Integration: call create_run_logger() then wire_logger(sim, rl) before
the simulation loop.  Call write_summary() and close_logger() after.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from herd_simulator.config import LoggingConfig, SimulatorConfig
from herd_simulator.models.animal import AnimalProfile
from herd_simulator.utils.geo import haversine_m

if TYPE_CHECKING:
    from herd_simulator.engine.simulator import AnimalTelemetry, Simulator

SCHEMA_VERSION = 1

TELEMETRY_HEADER = (
    "schema_version,run_id,sim_second,animal_id,is_physical,"
    "body_temp_c,ambient_temp_c,humidity_pct,thi,behaviour,"
    "latitude,longitude,risk_score,risk_source,geofence_status,"
    "battery_pct,event_codes"
)

GROUND_TRUTH_HEADER = (
    "sim_second,animal_a_id,animal_b_id,distance_m,"
    "animal_a_anomalies,animal_b_anomalies"
)


# -----------------------------------------------------------------------
# Buffered writer
# -----------------------------------------------------------------------

@dataclass
class BufferedWriter:
    path: Path
    handle: io.TextIOWrapper
    buffer: list[str]
    buffer_size: int
    lines_written: int = 0


def _new_buffered_writer(
    path: Path,
    buffer_size: int,
    header: Optional[str] = None,
) -> BufferedWriter:
    handle = open(path, "w", encoding="utf-8", newline="")
    if header is not None:
        handle.write(header + "\n")
    return BufferedWriter(
        path=path,
        handle=handle,
        buffer=[],
        buffer_size=max(buffer_size, 1),
    )


def _write_line(writer: BufferedWriter, line: str) -> None:
    writer.buffer.append(line)
    if len(writer.buffer) >= writer.buffer_size:
        _flush_writer(writer)


def _flush_writer(writer: BufferedWriter) -> None:
    if not writer.buffer:
        return
    writer.handle.write("\n".join(writer.buffer) + "\n")
    writer.lines_written += len(writer.buffer)
    writer.buffer.clear()
    writer.handle.flush()


def _close_writer(writer: BufferedWriter) -> None:
    _flush_writer(writer)
    writer.handle.close()


# -----------------------------------------------------------------------
# Run manifest
# -----------------------------------------------------------------------

@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    run_id: str
    start_time_iso: str
    config_hash: str
    mode: str
    seed: int
    herd_size: int
    scenario_id: Optional[str]


# -----------------------------------------------------------------------
# Run logger
# -----------------------------------------------------------------------

@dataclass
class RunLogger:
    run_dir: Path
    manifest: RunManifest
    telemetry_writer: Optional[BufferedWriter]
    ground_truth_writer: Optional[BufferedWriter]
    events_writer: Optional[BufferedWriter]
    transmissions_writer: Optional[BufferedWriter]
    logging_cfg: LoggingConfig
    profiles: dict[int, AnimalProfile]

    total_ticks: int = 0
    total_telemetry_rows: int = 0
    total_events_activated: int = 0
    total_events_expired: int = 0
    total_events_cleared: int = 0
    total_transmissions: int = 0
    total_ground_truth_pairs: int = 0


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _config_to_dict(cfg: SimulatorConfig) -> dict:
    d = dataclasses.asdict(cfg)
    return d


def _config_hash(cfg: SimulatorConfig) -> str:
    raw = json.dumps(
        _config_to_dict(cfg), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _normalize_float(value: float, decimals: int = 6) -> str:
    return f"{round(value, decimals):.{decimals}f}"


# -----------------------------------------------------------------------
# Initialization
# -----------------------------------------------------------------------

def create_run_logger(
    cfg: SimulatorConfig,
    mode: str,
    profiles: dict[int, AnimalProfile],
    scenario_id: Optional[str] = None,
) -> RunLogger:
    run_id = str(uuid.uuid4())
    start_time = datetime.now(timezone.utc)
    start_iso = start_time.isoformat()
    config_hash = _config_hash(cfg)
    stamp = start_time.strftime("%Y%m%d_%H%M%S")

    manifest = RunManifest(
        schema_version=SCHEMA_VERSION,
        run_id=run_id,
        start_time_iso=start_iso,
        config_hash=config_hash,
        mode=mode,
        seed=cfg.seed,
        herd_size=cfg.herd.n_total,
        scenario_id=scenario_id,
    )

    run_dir = Path(cfg.logging.log_dir) / f"run_{stamp}_{run_id[:8]}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # manifest.json
    (run_dir / "manifest.json").write_text(
        json.dumps(dataclasses.asdict(manifest), indent=2) + "\n",
        encoding="utf-8",
    )

    # config.snapshot.json
    (run_dir / "config.snapshot.json").write_text(
        json.dumps(_config_to_dict(cfg), indent=2) + "\n",
        encoding="utf-8",
    )

    # animal_profiles.json
    sorted_profiles = sorted(profiles.values(), key=lambda p: p.animal_id)
    profiles_data = [dataclasses.asdict(p) for p in sorted_profiles]
    (run_dir / "animal_profiles.json").write_text(
        json.dumps(profiles_data, indent=2) + "\n",
        encoding="utf-8",
    )

    bs = cfg.logging.buffer_size

    telemetry_writer = None
    if cfg.logging.telemetry_csv_enabled:
        telemetry_writer = _new_buffered_writer(
            run_dir / "telemetry.csv", bs, header=TELEMETRY_HEADER,
        )

    ground_truth_writer = None
    if cfg.logging.ground_truth_enabled:
        ground_truth_writer = _new_buffered_writer(
            run_dir / "ground_truth_pairs.csv", bs, header=GROUND_TRUTH_HEADER,
        )

    events_writer = None
    if cfg.logging.events_jsonl_enabled:
        events_writer = _new_buffered_writer(run_dir / "events.jsonl", bs)

    transmissions_writer = _new_buffered_writer(
        run_dir / "transmissions.jsonl", bs,
    )

    return RunLogger(
        run_dir=run_dir,
        manifest=manifest,
        telemetry_writer=telemetry_writer,
        ground_truth_writer=ground_truth_writer,
        events_writer=events_writer,
        transmissions_writer=transmissions_writer,
        logging_cfg=cfg.logging,
        profiles=profiles,
    )


# -----------------------------------------------------------------------
# Telemetry CSV
# -----------------------------------------------------------------------

def log_telemetry_row(
    rl: RunLogger,
    t: AnimalTelemetry,
    ambient_temp_c: float,
    humidity_pct: float,
) -> None:
    if rl.telemetry_writer is None:
        return
    event_codes_str = "|".join(sorted(t.event_codes)) if t.event_codes else ""
    line = ",".join([
        str(SCHEMA_VERSION),
        rl.manifest.run_id,
        str(t.sim_second),
        str(t.animal_id),
        str(int(t.is_physical)),
        _normalize_float(t.body_temp_c),
        _normalize_float(ambient_temp_c),
        _normalize_float(humidity_pct),
        _normalize_float(t.thi),
        str(t.behaviour),
        _normalize_float(t.latitude),
        _normalize_float(t.longitude),
        str(t.risk_score),
        "RULE",
        str(t.geofence_status),
        _normalize_float(t.battery_pct),
        event_codes_str,
    ])
    _write_line(rl.telemetry_writer, line)
    rl.total_telemetry_rows += 1


# -----------------------------------------------------------------------
# Events JSONL
# -----------------------------------------------------------------------

def log_event(
    rl: RunLogger,
    action: str,
    event_id: str,
    animal_id: int,
    event_type: str,
    sim_second: int,
    params: Optional[dict] = None,
    duration_seconds: Optional[int] = None,
    source: Optional[str] = None,
) -> None:
    if rl.events_writer is None:
        return
    record: dict = {
        "action": action,
        "event_id": event_id,
        "animal_id": animal_id,
        "event_type": event_type,
        "sim_second": sim_second,
    }
    if params is not None:
        record["params"] = params
    if duration_seconds is not None:
        record["duration_seconds"] = duration_seconds
    if source is not None:
        record["source"] = source

    _write_line(rl.events_writer, json.dumps(record, separators=(",", ":")))

    if action == "activated":
        rl.total_events_activated += 1
    elif action == "expired":
        rl.total_events_expired += 1
    elif action == "cleared":
        rl.total_events_cleared += 1


# -----------------------------------------------------------------------
# Transmissions JSONL
# -----------------------------------------------------------------------

def log_transmission(rl: RunLogger, t: AnimalTelemetry) -> None:
    if rl.transmissions_writer is None:
        return
    record = {
        "sim_second": t.sim_second,
        "animal_id": t.animal_id,
        "body_temp_c": round(t.body_temp_c, 6),
        "thi": round(t.thi, 6),
        "behaviour": t.behaviour,
        "latitude": round(t.latitude, 6),
        "longitude": round(t.longitude, 6),
        "risk_score": t.risk_score,
        "alert_band": t.alert_band,
        "geofence_status": t.geofence_status,
        "battery_pct": round(t.battery_pct, 6),
    }
    _write_line(rl.transmissions_writer, json.dumps(record, separators=(",", ":")))
    rl.total_transmissions += 1


def log_write_result(
    rl: RunLogger,
    animal_id: int,
    sim_second: int,
    outcome: str,
    status_code: Optional[int] = None,
    attempts: int = 0,
) -> None:
    if rl.transmissions_writer is None:
        return
    record: dict = {
        "type": "http_result",
        "animal_id": animal_id,
        "sim_second": sim_second,
        "outcome": outcome,
    }
    if status_code is not None:
        record["http_status"] = status_code
    if attempts > 0:
        record["attempts"] = attempts
    _write_line(rl.transmissions_writer, json.dumps(record, separators=(",", ":")))


def log_skipped_transmission(
    rl: RunLogger,
    t: AnimalTelemetry,
    reason: str,
) -> None:
    """Record an intentional no-send without counting it as a write."""
    if rl.transmissions_writer is None:
        return
    record = {
        "type": "skipped",
        "reason": reason,
        "sim_second": t.sim_second,
        "animal_id": t.animal_id,
    }
    _write_line(rl.transmissions_writer, json.dumps(record, separators=(",", ":")))


# -----------------------------------------------------------------------
# Ground truth
# -----------------------------------------------------------------------

def log_ground_truth_tick(
    rl: RunLogger,
    tick_telemetry: list[AnimalTelemetry],
) -> None:
    if rl.ground_truth_writer is None:
        return
    sorted_t = sorted(tick_telemetry, key=lambda t: t.animal_id)
    ss = sorted_t[0].sim_second
    for a, b in combinations(sorted_t, 2):
        dist = haversine_m((a.latitude, a.longitude), (b.latitude, b.longitude))
        a_anomalies = "|".join(sorted(a.event_codes)) if a.event_codes else ""
        b_anomalies = "|".join(sorted(b.event_codes)) if b.event_codes else ""
        line = ",".join([
            str(ss),
            str(a.animal_id),
            str(b.animal_id),
            _normalize_float(dist),
            a_anomalies,
            b_anomalies,
        ])
        _write_line(rl.ground_truth_writer, line)
    rl.total_ground_truth_pairs += len(sorted_t) * (len(sorted_t) - 1) // 2


# -----------------------------------------------------------------------
# Summary & close
# -----------------------------------------------------------------------

def write_summary(
    rl: RunLogger,
    sim_second: int,
    total_writes: int,
    sweeps_completed: int,
) -> None:
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": rl.manifest.run_id,
        "total_ticks": rl.total_ticks,
        "total_telemetry_rows": rl.total_telemetry_rows,
        "total_events_activated": rl.total_events_activated,
        "total_events_expired": rl.total_events_expired,
        "total_events_cleared": rl.total_events_cleared,
        "total_transmissions": rl.total_transmissions,
        "total_ground_truth_pairs": rl.total_ground_truth_pairs,
        "total_scheduler_writes": total_writes,
        "total_scheduler_sweeps": sweeps_completed,
        "final_sim_second": sim_second,
    }
    (rl.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )


def close_logger(rl: RunLogger) -> None:
    if rl.telemetry_writer is not None:
        _close_writer(rl.telemetry_writer)
    if rl.ground_truth_writer is not None:
        _close_writer(rl.ground_truth_writer)
    if rl.events_writer is not None:
        _close_writer(rl.events_writer)
    if rl.transmissions_writer is not None:
        _close_writer(rl.transmissions_writer)


# -----------------------------------------------------------------------
# Wiring
# -----------------------------------------------------------------------

def wire_logger(sim: Simulator, rl: RunLogger) -> None:
    """Register logger callbacks on the simulator's hooks."""

    def _on_telemetry(t: AnimalTelemetry) -> None:
        log_telemetry_row(rl, t, sim._ambient_temp_c, sim._humidity_pct)

    def _on_transmit(t: AnimalTelemetry) -> None:
        log_transmission(rl, t)

    def _on_transmission_skipped(t: AnimalTelemetry, reason: str) -> None:
        log_skipped_transmission(rl, t, reason)

    def _on_event_activated(event_id: str, animal_id: int, event_type: str) -> None:
        from herd_simulator.engine.scenario_runner import get_active_event, EventType
        ae = get_active_event(sim.event_state, animal_id, EventType(event_type))
        params = ae.event.params if ae else None
        duration = ae.event.duration_seconds if ae else None
        source = "scenario" if duration is not None else "cli"
        log_event(
            rl, "activated", event_id, animal_id, event_type,
            sim.clock.sim_second,
            params=params,
            duration_seconds=duration,
            source=source,
        )

    def _on_event_expired(event_id: str, animal_id: int, event_type: str) -> None:
        log_event(rl, "expired", event_id, animal_id, event_type, sim.clock.sim_second)

    def _on_event_cleared(event_id: str, animal_id: int, event_type: str) -> None:
        log_event(rl, "cleared", event_id, animal_id, event_type, sim.clock.sim_second)

    def _on_tick_complete(telemetry: list[AnimalTelemetry], ss: int) -> None:
        log_ground_truth_tick(rl, telemetry)
        rl.total_ticks += 1

    sim.on_telemetry = _on_telemetry
    sim.on_transmit = _on_transmit
    sim.on_transmission_skipped = _on_transmission_skipped
    sim.on_event_activated = _on_event_activated
    sim.on_event_expired = _on_event_expired
    sim.on_event_cleared = _on_event_cleared
    sim.on_tick_complete = _on_tick_complete
