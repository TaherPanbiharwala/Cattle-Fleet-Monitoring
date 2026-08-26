"""
simulator.py — 1-second tick loop coordinator with SimClock abstraction.

This is the central orchestrator that ties together all subsystems.
Each tick processes, in order:
  1. Scheduled scenario events (activation)
  2. Live CLI command queue (activation)
  3. Auto-expiry of scripted events whose duration has elapsed
  4. Physiology (body temp, ambient temp, THI, fever ramp)
  5. Behaviour (Markov state transitions)
  6. Movement — the ADR-014 "Social State" stage: centroid drift, individual
     positions, isolation drift, breach excursions
  7. Battery (activity-aware drain)
  8. Risk scoring — folds in the ADR-014 "Collar Faults" stage (tamper) —
     via the product-rule composite
  9. Geofence classification
  10. Scheduler (the ADR-014 "Transmission" stage: decide who transmits next)

This is a finer-grained breakdown of ADR-014's own composition order
(Physiology -> Movement -> Social State -> Collar Faults -> Risk
Calculation -> Transmission); "Social State" and "Collar Faults" aren't
separate function calls here, they're folded into steps 6 and 8 above.

SimClock abstraction (ADR-012):
  - dry-run mode:  ticks as fast as possible, sim_second increments by 1 each loop
  - offline mode:  real-time 1s sleeps between ticks
  - live mode:     real-time 1s sleeps + ThingSpeak writes enabled

References:
  ADR-012: Simulation Speed & SimClock Abstraction
  ADR-014: Dual Fault-Injection Engine
  HerdSimulator PRD §6.1–§6.6, §9.1–§9.5
  AGENTS.md §6: Execution Modes & CLI Commands
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from queue import Empty, Queue
from typing import Callable, Optional

from herd_simulator.config import SimulatorConfig
from herd_simulator.engine.live_cli import (
    CLICommand,
    CLICommandType,
    CLI_TO_EVENT_TYPE,
)
from herd_simulator.engine.scenario_runner import (
    ActiveEvent,
    EventState,
    EventType,
    ScenarioEvent,
    activate_event,
    clear_all_events,
    expire_events,
    get_active_event,
    get_all_active_for_animal,
    get_event_codes_for_status,
    has_any_anomaly,
    is_event_active,
    new_event_state,
    process_scheduled_events,
)
from herd_simulator.engine.scheduler import (
    SchedulerConfig,
    SchedulerState,
    enqueue_priority,
    get_queue_snapshot,
    is_write_allowed,
    new_scheduler,
    next_animal,
    next_cadence_s,
    record_write,
)
from herd_simulator.models.animal import (
    AnimalProfile,
    AnimalState,
    generate_profile,
    new_animal_state,
)
from herd_simulator.models.battery import step as battery_step
from herd_simulator.models.behaviour import Behaviour, step as behaviour_step
from herd_simulator.models.movement import (
    animal_position,
    breach_excursion_target,
    individual_offset_m,
    isolation_extra_distance_m,
    move_toward,
    step_centroid_anchored,
    step_centroid_autonomous,
)
from herd_simulator.models.physiology import (
    ambient_humidity_pct,
    ambient_temperature_c,
    body_temperature_c,
    fever_ramp_offset_c,
    heat_stress_temperature_c,
    validate_body_temp,
)
from herd_simulator.utils.geo import (
    Coord,
    classify_geofence,
    compute_thi,
)
from herd_simulator.utils.risk import (
    RiskInputs,
    classify_alert,
    compute_risk_score,
)

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# SimClock — ADR-012
# -----------------------------------------------------------------------

class SimMode(str, Enum):
    DRY_RUN = "dry-run"    # Ticks as fast as possible, no ThingSpeak
    OFFLINE = "offline"    # Real-time 1s ticks, no ThingSpeak
    LIVE = "live"          # Real-time 1s ticks + ThingSpeak writes


@dataclass
class SimClock:
    """Abstracts time progression for different execution modes."""
    mode: SimMode
    sim_second: int = 0
    wall_start: float = field(default_factory=time.monotonic)

    def tick(self) -> None:
        """Advance by one simulated second."""
        self.sim_second += 1
        if self.mode in (SimMode.OFFLINE, SimMode.LIVE):
            # Real-time: sleep to maintain 1s cadence
            elapsed = time.monotonic() - self.wall_start
            expected = self.sim_second
            drift = expected - elapsed
            if drift > 0:
                time.sleep(drift)

    @property
    def hour_of_day(self) -> float:
        """Current hour of the simulated day (0.0–23.999...)."""
        return (self.sim_second % 86400) / 3600.0


# -----------------------------------------------------------------------
# Per-tick telemetry snapshot
# -----------------------------------------------------------------------

@dataclass
class AnimalTelemetry:
    """One tick's complete telemetry for a single animal (8-field contract)."""
    animal_id: int
    is_physical: bool
    sim_second: int
    body_temp_c: float
    thi: float
    behaviour: int          # Behaviour code (0–4)
    latitude: float
    longitude: float
    risk_score: int         # 0–100
    alert_band: str         # "green" / "yellow" / "red"
    geofence_status: int    # 0 / 1 / 2
    battery_pct: float
    event_codes: list[str]  # ["FEVER", "BREACH", ...]
    dropped_out: bool


# -----------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------

@dataclass
class Simulator:
    """Central simulation coordinator.

    Holds all state: config, animals, clock, events, scheduler.
    The `tick()` method advances the simulation by one second.
    """

    cfg: SimulatorConfig
    clock: SimClock
    animals: dict[int, AnimalState]
    profiles: dict[int, AnimalProfile]
    event_state: EventState
    scheduler_state: SchedulerState
    scenario_events: list[ScenarioEvent]
    scenario_cursor: int
    centroid: Coord
    cli_queue: Queue[CLICommand]

    # Shared RNG for weather (all animals share one weather)
    weather_rng: random.Random = field(default_factory=lambda: random.Random(42))

    # Callbacks for external integration (ThingSpeak writer, logger, etc.)
    on_telemetry: Optional[Callable[[AnimalTelemetry], None]] = None
    on_transmit: Optional[Callable[[AnimalTelemetry], None]] = None
    on_event_activated: Optional[Callable[[str, int, str], None]] = None
    on_event_expired: Optional[Callable[[str, int, str], None]] = None
    on_event_cleared: Optional[Callable[[str, int, str], None]] = None
    on_tick_complete: Optional[Callable[[list[AnimalTelemetry], int], None]] = None

    # Control flags
    paused: bool = False
    running: bool = True

    # Cached per-tick shared weather
    _ambient_temp_c: float = 28.0
    _humidity_pct: float = 65.0
    _thi: float = 75.0


def create_simulator(
    cfg: SimulatorConfig,
    mode: SimMode,
    scenario_events: Optional[list[ScenarioEvent]] = None,
    cli_queue: Optional[Queue[CLICommand]] = None,
) -> Simulator:
    """Factory: build a fully initialized Simulator ready to tick.

    Generates all 20 animal profiles and initial states deterministically
    from cfg.seed. The centroid starts at the polygon centroid.
    """
    # Compute polygon centroid as starting position
    lats = [p[0] for p in cfg.pasture_polygon]
    lons = [p[1] for p in cfg.pasture_polygon]
    centroid: Coord = (sum(lats) / len(lats), sum(lons) / len(lons))

    # Generate profiles
    profiles: dict[int, AnimalProfile] = {}
    animals: dict[int, AnimalState] = {}

    # ID 1 = physical collar (reserved)
    p1 = generate_profile(cfg.herd.physical_collar_id, cfg, is_physical=True)
    profiles[cfg.herd.physical_collar_id] = p1
    animals[cfg.herd.physical_collar_id] = new_animal_state(p1, cfg, centroid)

    # IDs 2..n_total = simulated
    for aid in range(2, cfg.herd.n_total + 1):
        profile = generate_profile(aid, cfg)
        profiles[aid] = profile
        animals[aid] = new_animal_state(profile, cfg, centroid)

    # Scheduler (simulated IDs only — physical collar uses Channel 1)
    sched_cfg = SchedulerConfig(
        animal_ids=list(range(2, cfg.herd.n_total + 1)),
        normal_cadence_s=cfg.thingspeak.write_cadence_s,
        alert_cadence_s=cfg.thingspeak.breach_cadence_s,
        min_interval_s=cfg.thingspeak.min_interval_s,
    )

    return Simulator(
        cfg=cfg,
        clock=SimClock(mode=mode),
        animals=animals,
        profiles=profiles,
        event_state=new_event_state(),
        scheduler_state=new_scheduler(sched_cfg),
        scenario_events=scenario_events or [],
        scenario_cursor=0,
        centroid=centroid,
        cli_queue=cli_queue or Queue(),
        weather_rng=random.Random(cfg.seed),
    )


def _process_cli_commands(sim: Simulator) -> None:
    """Drain the CLI command queue and apply each command."""
    while True:
        try:
            cmd = sim.cli_queue.get_nowait()
        except Empty:
            break

        if cmd.command == CLICommandType.QUIT:
            sim.running = False
            logger.info("CLI: quit received")
        elif cmd.command == CLICommandType.PAUSE:
            sim.paused = True
            logger.info("CLI: simulation paused")
        elif cmd.command == CLICommandType.RESUME:
            sim.paused = False
            logger.info("CLI: simulation resumed")
        elif cmd.command == CLICommandType.STATUS:
            _log_status(sim)
        elif cmd.command == CLICommandType.CLEAR:
            if cmd.animal_id is not None:
                cleared = clear_all_events(sim.event_state, cmd.animal_id)
                logger.info("CLI: cleared %d events on animal %d", len(cleared), cmd.animal_id)
                for evt_id, aid, etype in cleared:
                    if sim.on_event_cleared:
                        sim.on_event_cleared(evt_id, aid, etype)
        elif cmd.command in CLI_TO_EVENT_TYPE:
            if cmd.animal_id is not None:
                evt_type = EventType(CLI_TO_EVENT_TYPE[cmd.command])
                event_id = activate_event(
                    sim.event_state,
                    cmd.animal_id,
                    evt_type,
                    sim.clock.sim_second,
                )
                # Priority jump for the affected animal
                if cmd.animal_id >= 2:  # Only simulated animals in the scheduler
                    enqueue_priority(sim.scheduler_state, cmd.animal_id)
                logger.info(
                    "CLI: activated %s on animal %d (event_id=%s)",
                    evt_type.value, cmd.animal_id, event_id,
                )
                if sim.on_event_activated:
                    sim.on_event_activated(event_id, cmd.animal_id, evt_type.value)


def _log_status(sim: Simulator) -> None:
    """Log current simulation status (for CLI `status` command)."""
    logger.info(
        "Status: sim_second=%d, active_events=%d, writes=%d, sweeps=%d",
        sim.clock.sim_second,
        len(sim.event_state.active),
        sim.scheduler_state.total_writes,
        sim.scheduler_state.sweeps_completed,
    )
    for (aid, etype), ae in sim.event_state.active.items():
        elapsed = sim.clock.sim_second - ae.activated_at
        logger.info("  Animal %d: %s (active for %ds)", aid, etype, elapsed)


def tick(sim: Simulator) -> list[AnimalTelemetry]:
    """Advance the simulation by one second.

    Returns a list of AnimalTelemetry for ALL animals this tick
    (used for logging). The scheduler separately decides which one
    gets transmitted to ThingSpeak.
    """
    # 0. Advance clock
    sim.clock.tick()
    ss = sim.clock.sim_second
    hour = sim.clock.hour_of_day

    # 1. Process scheduled scenario events
    sim.scenario_cursor, activated_ids = process_scheduled_events(
        sim.event_state,
        sim.scenario_events,
        ss,
        sim.scenario_cursor,
    )
    # Priority-jump and notify for newly activated scenario events
    for evt_id in activated_ids:
        for (aid, etype), ae in sim.event_state.active.items():
            if ae.event.event_id == evt_id:
                if sim.on_event_activated:
                    sim.on_event_activated(evt_id, aid, etype)
                if aid >= 2:
                    enqueue_priority(sim.scheduler_state, aid)

    # 1b. Auto-clear any scripted event whose duration has elapsed. Only
    # scenario-sourced events carry a duration — CLI/API-activated events
    # run until an explicit `clear` command (see expire_events docstring).
    expired_info = expire_events(sim.event_state, ss)
    if expired_info:
        logger.info("Expired %d event(s): %s", len(expired_info), [e[0] for e in expired_info])
        for evt_id, aid, etype in expired_info:
            if sim.on_event_expired:
                sim.on_event_expired(evt_id, aid, etype)

    # 2. Drain CLI commands
    _process_cli_commands(sim)

    if sim.paused:
        return []

    # 3. Shared weather for this tick
    sim._ambient_temp_c = ambient_temperature_c(
        hour, sim.cfg.weather.ambient_temp_day, sim.cfg.weather.ambient_temp_night,
    )
    sim._humidity_pct = ambient_humidity_pct(
        hour, sim.cfg.weather.humidity_mean, sim.cfg.weather.humidity_std, sim.weather_rng,
    )

    # Apply global heat stress (if any animal has heat_stress, it affects shared weather)
    for (aid, etype), ae in sim.event_state.active.items():
        if etype == EventType.HEAT_STRESS.value and not ae.cleared:
            boost = ae.event.params.get("ambient_boost_c", 8.0)
            sim._ambient_temp_c = heat_stress_temperature_c(sim._ambient_temp_c, boost)
            break  # Only apply once (heat stress is environmental)

    sim._thi = compute_thi(sim._ambient_temp_c, sim._humidity_pct)

    # 4. Update centroid (autonomous drift — no Collar-1 sniffing in P1 sim-only mode)
    sim.centroid = step_centroid_autonomous(
        sim.centroid,
        sim.cfg.movement,
        sim.cfg.pasture_polygon,
        1.0,
        sim.weather_rng,
    )

    # 5. Per-animal tick
    telemetry: list[AnimalTelemetry] = []
    for aid, state in sim.animals.items():
        t = _tick_animal(sim, state, ss, hour)
        telemetry.append(t)
        if sim.on_telemetry:
            sim.on_telemetry(t)

    # 6. Scheduler — decide who transmits
    sched_cfg = SchedulerConfig(
        animal_ids=list(range(2, sim.cfg.herd.n_total + 1)),
        normal_cadence_s=sim.cfg.thingspeak.write_cadence_s,
        alert_cadence_s=sim.cfg.thingspeak.breach_cadence_s,
        min_interval_s=sim.cfg.thingspeak.min_interval_s,
    )
    cadence = next_cadence_s(sim.scheduler_state, sched_cfg)
    now_ts = float(ss)

    if is_write_allowed(sim.scheduler_state, now_ts, sched_cfg.min_interval_s):
        if now_ts - sim.scheduler_state.last_write_time >= cadence or sim.scheduler_state.total_writes == 0:
            tx_id = next_animal(sim.scheduler_state, sched_cfg)
            if tx_id is not None:
                record_write(sim.scheduler_state, now_ts)
                # Find the telemetry for this animal
                for t in telemetry:
                    if t.animal_id == tx_id and sim.on_transmit:
                        sim.on_transmit(t)
                        break

    # 7. Notify tick-complete listeners (ground truth, batch logging)
    if telemetry and sim.on_tick_complete:
        sim.on_tick_complete(telemetry, ss)

    return telemetry


def _tick_animal(sim: Simulator, state: AnimalState, ss: int, hour: float) -> AnimalTelemetry:
    """Run one tick for a single animal. Updates state in-place and returns telemetry."""
    aid = state.profile.animal_id
    profile = state.profile

    # Skip ticking dropped-out animals (they're dead)
    if state.battery.dropped_out:
        return AnimalTelemetry(
            animal_id=aid,
            is_physical=profile.is_physical,
            sim_second=ss,
            body_temp_c=state.body_temp_c,
            thi=sim._thi,
            behaviour=state.behaviour.value,
            latitude=state.position[0],
            longitude=state.position[1],
            # Master PRD: "Dropout suppresses transmission; the HUD marks
            # the cow stale and critical instead of fabricating a current
            # score" — 100/red, not a fabricated healthy 0/green.
            risk_score=100,
            alert_band="red",
            geofence_status=0,
            battery_pct=0.0,
            event_codes=["DROPOUT"],
            dropped_out=True,
        )

    # -- Check for collar_dropout event --
    if is_event_active(sim.event_state, aid, EventType.COLLAR_DROPOUT):
        from herd_simulator.models.battery import BatteryState
        state.battery = BatteryState(level_pct=0.0, dropped_out=True)
        return AnimalTelemetry(
            animal_id=aid,
            is_physical=profile.is_physical,
            sim_second=ss,
            body_temp_c=state.body_temp_c,
            thi=sim._thi,
            behaviour=state.behaviour.value,
            latitude=state.position[0],
            longitude=state.position[1],
            # Master PRD: "Dropout suppresses transmission; the HUD marks
            # the cow stale and critical instead of fabricating a current
            # score" — 100/red, not a fabricated healthy 0/green.
            risk_score=100,
            alert_band="red",
            geofence_status=0,
            battery_pct=0.0,
            event_codes=["DROPOUT"],
            dropped_out=True,
        )

    # -- Physiology --
    fever_offset = 0.0
    fever_evt = get_active_event(sim.event_state, aid, EventType.FEVER_ONSET)
    if fever_evt:
        elapsed = ss - fever_evt.activated_at
        fever_offset = fever_ramp_offset_c(
            elapsed,
            fever_evt.event.params.get("onset_s", 300),
            fever_evt.event.params.get("plateau_s", 600),
            fever_evt.event.params.get("recovery_s", 300),
            fever_evt.event.params.get("peak_offset_c", 1.8),
        )

    body_temp = body_temperature_c(
        profile.baseline_temp_c,
        hour,
        sim.cfg.physiology.diurnal_amplitude,
        profile.temp_noise_std_c,
        state.rng,
        fever_offset_c=fever_offset,
    )
    # Clamp to plausible bounds (log warning instead of crash in sim loop)
    body_temp = max(35.0, min(43.0, body_temp))
    state.body_temp_c = body_temp

    # -- Behaviour --
    anomaly_active = has_any_anomaly(sim.event_state, aid)
    # Only transition every transition_interval_s
    if ss % sim.cfg.behaviour.transition_interval_s == 0:
        state.behaviour = behaviour_step(
            state.behaviour,
            sim.cfg.behaviour,
            hour,
            state.rng,
            anomaly_active=anomaly_active,
        )

    # -- Movement --
    is_isolated = is_event_active(sim.event_state, aid, EventType.SOCIAL_ISOLATION)
    is_breaching = is_event_active(sim.event_state, aid, EventType.GEOFENCE_BREACH)

    if is_breaching:
        # Drive animal outside polygon
        target = breach_excursion_target(
            state.position,
            sim.cfg.pasture_polygon,
        )
        state.position = move_toward(state.position, target, max_step_m=0.5)
    elif is_isolated:
        # Normal position + isolation drift
        iso_evt = get_active_event(sim.event_state, aid, EventType.SOCIAL_ISOLATION)
        iso_elapsed = (ss - iso_evt.activated_at) if iso_evt else 0
        iso_extra = isolation_extra_distance_m(float(iso_elapsed))
        state.position = animal_position(
            sim.centroid,
            state.behaviour,
            profile.preferred_bearing_rad,
            profile.preferred_offset_m,
            sim.cfg.movement,
            state.rng,
            walking_speed_range_mps=profile.walking_speed_range_mps,
            isolation_extra_m=iso_extra,
        )
    else:
        # Normal herd positioning
        state.position = animal_position(
            sim.centroid,
            state.behaviour,
            profile.preferred_bearing_rad,
            profile.preferred_offset_m,
            sim.cfg.movement,
            state.rng,
            walking_speed_range_mps=profile.walking_speed_range_mps,
        )

    # -- Battery --
    alert_active = anomaly_active or is_breaching
    state.battery = battery_step(state.battery, sim.cfg.battery, 1.0, alert_active)

    # -- Geofence --
    geofence_status = classify_geofence(state.position, sim.cfg.pasture_polygon)

    # -- Risk score --
    is_tampered = is_event_active(sim.event_state, aid, EventType.TAMPER)
    risk_inputs = RiskInputs(
        body_temp=body_temp,
        baseline_temp=profile.baseline_temp_c,
        thi=sim._thi,
        is_restless=(state.behaviour == Behaviour.RESTLESS),
        geofence_status=geofence_status,
        is_isolated=is_isolated,
        is_tampered=is_tampered,
    )
    risk_score = compute_risk_score(risk_inputs, sim.cfg.risk.severity)
    alert_band = classify_alert(
        risk_score,
        sim.cfg.risk.alert_bands.green_max,
        sim.cfg.risk.alert_bands.yellow_max,
    )

    # -- Event codes for status field --
    event_codes = get_event_codes_for_status(sim.event_state, aid)

    # Update sim_second on state
    state.sim_second = ss

    return AnimalTelemetry(
        animal_id=aid,
        is_physical=profile.is_physical,
        sim_second=ss,
        body_temp_c=body_temp,
        thi=sim._thi,
        behaviour=state.behaviour.value,
        latitude=state.position[0],
        longitude=state.position[1],
        risk_score=risk_score,
        alert_band=alert_band,
        geofence_status=geofence_status,
        battery_pct=state.battery.level_pct,
        event_codes=event_codes,
        dropped_out=state.battery.dropped_out,
    )


def run_simulation(
    sim: Simulator,
    duration_seconds: Optional[int] = None,
) -> None:
    """Run the simulation loop until stopped or duration reached.

    Args:
        sim: The simulator instance.
        duration_seconds: If set, stop after this many simulated seconds.
                         If None, run until quit command or KeyboardInterrupt.
    """
    logger.info(
        "Starting simulation: mode=%s, herd=%d, seed=%d, duration=%s",
        sim.clock.mode.value,
        sim.cfg.herd.n_total,
        sim.cfg.seed,
        f"{duration_seconds}s" if duration_seconds else "unlimited",
    )

    try:
        while sim.running:
            if duration_seconds and sim.clock.sim_second >= duration_seconds:
                logger.info("Duration limit reached (%ds), stopping.", duration_seconds)
                break

            tick(sim)

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — stopping simulation.")
    finally:
        sim.running = False
        logger.info(
            "Simulation ended: %d ticks, %d writes, %d sweeps",
            sim.clock.sim_second,
            sim.scheduler_state.total_writes,
            sim.scheduler_state.sweeps_completed,
        )
