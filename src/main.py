"""
main.py — CLI entry point for the Intelligent Cattle Fleet Management Platform.

Supported execution modes (AGENTS.md §6):
  1. High-Speed Dry Run:
     python src/main.py --mode dry-run --config config/default_config.yaml --duration-hours 24
  2. Offline Mode with Local Web HUD:
     python src/main.py --mode offline --config config/default_config.yaml --hud
  3. Live Mode (ThingSpeak uplink + Collar-1 sniffing):
     python src/main.py --mode live --config config/default_config.yaml --scenario config/scenarios/demo_scenario.json --hud
  4. Replay Mode:
     python src/main.py --mode replay --log-dir logs/run_20260820_001/

References:
  - Master PRD: Delivery Phases (P1: Digital-Twin Simulator)
  - AGENTS.md §6: Execution Modes & CLI Commands
  - ADR-012 (Dry-Run), ADR-018 (Logging & Replay), ADR-019 (ThingSpeak),
    ADR-020 (HUD), ADR-021 (CLI Entry Point)
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import logging
import os
import signal
import sys
import threading
import time
from itertools import groupby
from pathlib import Path
from queue import Queue
from typing import Optional, Sequence

from herd_simulator.config import (
    ConfigError,
    SimulatorConfig,
    load_config,
    load_env_credentials,
)
from herd_simulator.engine.live_cli import start_cli_thread
from herd_simulator.engine.scenario_runner import (
    Scenario,
    ScenarioError,
    load_scenario,
)
from herd_simulator.engine.simulator import (
    AnimalTelemetry,
    SimMode,
    Simulator,
    create_simulator,
    run_simulation,
)
from herd_simulator.env_loader import load_dotenv
from herd_simulator.services.api_server import (
    MAX_HISTORY_PER_ANIMAL,
    HudServer,
    create_hud_state,
    wire_api_server,
)
from herd_simulator.services.logger import (
    RunLogger,
    close_logger,
    create_run_logger,
    log_write_result,
    wire_logger,
    write_summary,
)
from herd_simulator.services.replay import (
    ReplayRow,
    load_manifest,
    load_replay,
)
from herd_simulator.services.thingspeak import (
    ThingSpeakClient,
    wire_thingspeak,
)

logger = logging.getLogger("herd_simulator")


def _build_parser() -> argparse.ArgumentParser:
    """Construct the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="herd-simulator",
        description="Intelligent Cattle Fleet Management — Phase 1 Digital-Twin Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/main.py --mode dry-run --duration-hours 24\n"
            "  python src/main.py --mode offline --hud\n"
            "  python src/main.py --mode live --scenario config/scenarios/demo_scenario.json --hud\n"
            "  python src/main.py --mode replay --log-dir logs/run_20260827_120000_abc12345/ --hud\n"
        ),
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["dry-run", "dry_run", "offline", "live", "replay"],
        help="Execution mode: dry-run, offline, live, or replay (default: offline)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/default_config.yaml",
        help="Path to YAML configuration file (default: config/default_config.yaml)",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Path to declarative scenario JSON file (optional)",
    )
    parser.add_argument(
        "--duration-hours",
        type=float,
        default=None,
        help="Stop simulation after N simulated hours (optional, unlimited if omitted)",
    )
    parser.add_argument(
        "--hud",
        action="store_true",
        default=False,
        help="Start local web HUD server on configured host:port",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Override HUD server port (default from config, typically 8000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override random seed in configuration",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Log directory to replay from (required for --mode replay) or override output directory",
    )
    parser.add_argument(
        "--playback-speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for replay mode (1.0 = real-time, 0 = fast-forward)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging output",
    )

    return parser


def _setup_logging(verbose: bool) -> None:
    """Configure console logging format and level."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _replay_row_to_telemetry(r: ReplayRow, alert_bands: tuple[int, int] = (39, 69)) -> AnimalTelemetry:
    """Convert a ReplayRow into an AnimalTelemetry instance."""
    green_max, yellow_max = alert_bands
    if r.risk_score <= green_max:
        alert_band = "green"
    elif r.risk_score <= yellow_max:
        alert_band = "yellow"
    else:
        alert_band = "red"

    dropped_out = r.battery_pct <= 0 or "DROPOUT" in r.event_codes

    return AnimalTelemetry(
        animal_id=r.animal_id,
        is_physical=r.is_physical,
        sim_second=r.sim_second,
        body_temp_c=r.body_temp_c,
        thi=r.thi,
        behaviour=r.behaviour,
        latitude=r.latitude,
        longitude=r.longitude,
        risk_score=r.risk_score,
        alert_band=alert_band,
        geofence_status=r.geofence_status,
        battery_pct=r.battery_pct,
        event_codes=list(r.event_codes),
        dropped_out=dropped_out,
    )


def _load_replay_config(config_path: str, log_dir: Path) -> SimulatorConfig:
    """Load the exact saved configuration when a replay snapshot exists."""
    config_snapshot = log_dir / "config.snapshot.json"
    if config_snapshot.exists():
        return load_config(config_snapshot)
    if Path(config_path).exists():
        return load_config(config_path)
    return load_config("config/default_config.yaml")


def _log_thingspeak_write_result(
    run_logger: RunLogger,
    animal_id: int,
    sim_second: int,
    outcome: str,
    status_code: Optional[int],
    attempts: int,
) -> None:
    """Preserve the client's (animal_id, sim_second) callback ordering."""
    log_write_result(
        run_logger,
        animal_id,
        sim_second,
        outcome,
        status_code,
        attempts,
    )


def run_replay(args: argparse.Namespace) -> int:
    """Execute replay mode from a prior run directory."""
    if not args.log_dir:
        logger.error("--log-dir is required when running in replay mode.")
        return 1

    log_dir = Path(args.log_dir)
    if not log_dir.exists() or not log_dir.is_dir():
        logger.error("Log directory not found: %s", log_dir)
        return 1

    telemetry_file = log_dir / "telemetry.csv"
    if not telemetry_file.exists():
        logger.error("telemetry.csv not found in log directory: %s", log_dir)
        return 1

    # Load manifest if present
    manifest = {}
    manifest_file = log_dir / "manifest.json"
    if manifest_file.exists():
        manifest = load_manifest(log_dir)

    run_id = manifest.get("run_id", log_dir.name)
    logger.info("Loaded replay for run: %s (dir: %s)", run_id, log_dir)

    try:
        cfg = _load_replay_config(args.config, log_dir)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("Replay configuration error: %s", exc)
        return 1

    if args.port is not None:
        cfg = dataclasses.replace(cfg, hud=dataclasses.replace(cfg.hud, port=args.port))

    # If HUD is requested, spin up HUD server and stream replay rows
    hud_server: Optional[HudServer] = None
    try:
        replay_iter = load_replay(log_dir)
        # Group rows by sim_second
        ticks = groupby(replay_iter, key=lambda r: r.sim_second)

        if args.hud:
            sim = create_simulator(cfg, SimMode.OFFLINE)
            hud_state = create_hud_state(sim, run_id)
            hud_server = HudServer(cfg.hud, hud_state)
            hud_server.start()
            logger.info("Replay HUD active at http://%s:%d", cfg.hud.host, hud_server.port or cfg.hud.port)

            stop_replay = threading.Event()

            def _sigint_handler(sig, frame):
                stop_replay.set()

            old_handler = signal.signal(signal.SIGINT, _sigint_handler)

            logger.info("Streaming telemetry replay to HUD (playback speed: %.1fx)...", args.playback_speed)
            last_wall = time.monotonic()

            for ss, group in ticks:
                if stop_replay.is_set():
                    logger.info("Replay stopped by user.")
                    break

                batch = [_replay_row_to_telemetry(r) for r in group]
                sim.clock.sim_second = ss

                with hud_state.lock:
                    hud_state.latest_telemetry = batch
                    for t in batch:
                        if t.animal_id not in hud_state.history:
                            hud_state.history[t.animal_id] = collections.deque(maxlen=MAX_HISTORY_PER_ANIMAL)
                        hud_state.history[t.animal_id].append(t)

                if args.playback_speed > 0:
                    delay = 1.0 / args.playback_speed
                    elapsed = time.monotonic() - last_wall
                    sleep_time = max(0.0, delay - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    last_wall = time.monotonic()

            signal.signal(signal.SIGINT, old_handler)
            logger.info("Replay stream completed.")

        else:
            row_count = 0
            tick_count = 0
            for ss, group in ticks:
                tick_count += 1
                for _ in group:
                    row_count += 1
            logger.info(
                "Replay summary for %s: %d ticks, %d telemetry rows verified.",
                run_id, tick_count, row_count,
            )

        return 0

    except Exception:
        logger.exception("Error during replay execution")
        return 1
    finally:
        if hud_server:
            hud_server.stop()


def run_simulator(args: argparse.Namespace) -> int:
    """Execute live, offline, or dry-run simulation mode."""
    # 1. Normalize mode string
    mode_str = args.mode.lower().replace("_", "-")
    if mode_str == "dry-run":
        sim_mode = SimMode.DRY_RUN
    elif mode_str == "live":
        sim_mode = SimMode.LIVE
    else:
        sim_mode = SimMode.OFFLINE

    # 2. Load .env file
    load_dotenv()

    # 3. Load & validate configuration
    try:
        cfg = load_config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    # 4. Apply CLI overrides
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)
    if args.port is not None:
        cfg = dataclasses.replace(cfg, hud=dataclasses.replace(cfg.hud, port=args.port))
    if args.log_dir is not None:
        cfg = dataclasses.replace(cfg, logging=dataclasses.replace(cfg.logging, log_dir=args.log_dir))

    # 5. Load scenario if provided
    scenario_events = None
    scenario_id = None
    if args.scenario:
        try:
            scenario: Scenario = load_scenario(
                args.scenario,
                valid_animal_ids=range(1, cfg.herd.n_total + 1),
            )
            scenario_events = scenario.events
            scenario_id = scenario.scenario_id
            logger.info("Loaded scenario '%s' (%d events)", scenario_id, len(scenario_events))
        except (ScenarioError, FileNotFoundError) as exc:
            logger.error("Scenario error: %s", exc)
            return 1

    # 6. Load credentials
    credentials = load_env_credentials()

    # 7. Calculate duration
    duration_s: Optional[int] = None
    if args.duration_hours is not None:
        if args.duration_hours <= 0:
            logger.error("--duration-hours must be positive.")
            return 1
        duration_s = int(args.duration_hours * 3600)

    # 8. Create Simulator
    cli_queue: Queue = Queue()
    sim = create_simulator(
        cfg=cfg,
        mode=sim_mode,
        scenario_events=scenario_events,
        cli_queue=cli_queue,
    )

    # 9. Create Run Logger & Wire
    run_logger: RunLogger = create_run_logger(
        cfg=cfg,
        mode=mode_str,
        profiles=sim.profiles,
        scenario_id=scenario_id,
    )
    wire_logger(sim, run_logger)

    # 10. Create ThingSpeak Client & Wire
    ts_client = ThingSpeakClient(
        cfg=cfg.thingspeak,
        credentials=credentials,
        mode=sim_mode,
    )
    wire_thingspeak(sim, ts_client)

    # Wire ThingSpeak write results to transmissions.jsonl
    def _on_ts_write_result(
        animal_id: int,
        ss: int,
        outcome: str,
        status_code: Optional[int],
        attempts: int,
    ) -> None:
        _log_thingspeak_write_result(
            run_logger,
            animal_id,
            ss,
            outcome,
            status_code,
            attempts,
        )

    ts_client.on_write_result = _on_ts_write_result
    ts_client.start()

    # 11. Optionally create HUD server & Wire
    hud_server: Optional[HudServer] = None
    if args.hud:
        hud_state = create_hud_state(
            sim=sim,
            run_id=run_logger.manifest.run_id,
            thingspeak_client=ts_client,
        )
        hud_server = HudServer(cfg.hud, hud_state)
        wire_api_server(sim, hud_state)
        hud_server.start()

    # 12. Start CLI input thread (interactive modes only)
    cli_stop: Optional[threading.Event] = None
    if sim_mode != SimMode.DRY_RUN and sys.stdin.isatty():
        cli_stop = threading.Event()
        start_cli_thread(cli_queue, cli_stop)
        logger.info("Interactive CLI prompt active (type 'help' for commands)")

    # 13. Graceful signal handling
    def _sig_handler(sig, frame):
        logger.info("Interrupt signal received — initiating clean shutdown...")
        sim.running = False

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    # 14. Banner
    print("\n" + "=" * 60)
    print(f" Intelligent Cattle Fleet Management — Digital Twin")
    print(f" Mode:           {mode_str.upper()}")
    print(f" Herd Size:      {cfg.herd.n_total} (ID 1 physical + {cfg.herd.n_sim} simulated)")
    print(f" Random Seed:    {cfg.seed}")
    print(f" Scenario:       {scenario_id or 'None (Autonomous Drift)'}")
    print(f" Duration:       {f'{args.duration_hours}h ({duration_s}s)' if duration_s else 'Unlimited'}")
    print(f" Log Directory:  {run_logger.run_dir}")
    if hud_server:
        print(f" Web HUD:        http://{cfg.hud.host}:{hud_server.port or cfg.hud.port}")
    if sim_mode == SimMode.LIVE:
        write_key_set = bool(credentials.get("THINGSPEAK_WRITE_API_KEY"))
        print(f" ThingSpeak:     Channel 2 POST (API key: {'SET' if write_key_set else 'MISSING'})")
    print("=" * 60 + "\n")

    # 15. Run simulation loop
    start_time = time.monotonic()
    try:
        run_simulation(sim, duration_seconds=duration_s)
    finally:
        elapsed = time.monotonic() - start_time
        logger.info("Shutting down services and flushing logs...")

        # Stop writers before closing files so callbacks cannot write through
        # closed buffered handles during shutdown.
        ts_client.stop()

        # Write summary and close logger
        write_summary(
            run_logger,
            sim_second=sim.clock.sim_second,
            total_writes=sim.scheduler_state.total_writes,
            sweeps_completed=sim.scheduler_state.sweeps_completed,
        )
        close_logger(run_logger)

        # Stop HUD and CLI
        if hud_server:
            hud_server.stop()
        if cli_stop:
            cli_stop.set()

        logger.info(
            "Run finished in %.2fs. Total ticks: %d, Transmissions: %d, Sweeps: %d",
            elapsed,
            sim.clock.sim_second,
            sim.scheduler_state.total_writes,
            sim.scheduler_state.sweeps_completed,
        )

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main CLI entry point function."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.verbose)

    mode = args.mode.lower().replace("_", "-")
    if mode == "replay":
        return run_replay(args)
    else:
        return run_simulator(args)


if __name__ == "__main__":
    sys.exit(main())
