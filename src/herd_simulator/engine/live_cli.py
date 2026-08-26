"""
live_cli.py — Threaded interactive CLI for live fault injection.

Runs in a separate daemon thread so `input()` never blocks the 1-second
tick loop. Parsed commands are pushed into a thread-safe queue that the
simulator drains each tick.

Supported commands (ADR-014 / autoplan DX review):
  fever <id>       — Inject fever on animal <id>
  heat <id>        — Inject heat stress on animal <id>
  breach <id>      — Inject geofence breach on animal <id>
  tamper <id>      — Inject collar tamper on animal <id>
  isolate <id>     — Inject social isolation on animal <id>
  dropout <id>     — Inject collar dropout on animal <id>
  clear <id>       — Clear ALL active events on animal <id>
  status           — Print summary of all active events
  pause            — Pause the simulation loop
  resume           — Resume the simulation loop
  quit / exit      — Stop the simulation
  help             — Show available commands

References:
  ADR-014: Dual Fault-Injection Engine (Declarative JSON + Live CLI)
  HerdSimulator PRD §6.6 (FR-31)
  Autoplan DX Review: "Live CLI `help` command must list all verbs"
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Optional


class CLICommandType(str, Enum):
    """Parsed CLI command types."""
    FEVER = "fever"
    HEAT = "heat"
    BREACH = "breach"
    TAMPER = "tamper"
    ISOLATE = "isolate"
    DROPOUT = "dropout"
    CLEAR = "clear"
    STATUS = "status"
    PAUSE = "pause"
    RESUME = "resume"
    QUIT = "quit"
    HELP = "help"


# Commands that require an animal ID argument
_ANIMAL_COMMANDS = {
    CLICommandType.FEVER,
    CLICommandType.HEAT,
    CLICommandType.BREACH,
    CLICommandType.TAMPER,
    CLICommandType.ISOLATE,
    CLICommandType.DROPOUT,
    CLICommandType.CLEAR,
}


@dataclass
class CLICommand:
    """A parsed CLI command ready for the simulator to process."""
    command: CLICommandType
    animal_id: Optional[int] = None
    raw_input: str = ""


# Map CLI verb → EventType string (for the simulator to translate)
CLI_TO_EVENT_TYPE = {
    CLICommandType.FEVER: "fever_onset",
    CLICommandType.HEAT: "heat_stress",
    CLICommandType.BREACH: "geofence_breach",
    CLICommandType.TAMPER: "tamper",
    CLICommandType.ISOLATE: "social_isolation",
    CLICommandType.DROPOUT: "collar_dropout",
}


HELP_TEXT = """\
╔══════════════════════════════════════════════════════════╗
║              Herd Simulator — Live CLI                   ║
╠══════════════════════════════════════════════════════════╣
║  fever <id>     Inject fever on animal <id>              ║
║  heat <id>      Inject heat stress on animal <id>        ║
║  breach <id>    Inject geofence breach on animal <id>    ║
║  tamper <id>    Inject collar tamper on animal <id>       ║
║  isolate <id>   Inject social isolation on animal <id>   ║
║  dropout <id>   Inject collar dropout on animal <id>     ║
║  clear <id>     Clear ALL events on animal <id>          ║
║  status         Show all active events                   ║
║  pause          Pause the simulation                     ║
║  resume         Resume the simulation                    ║
║  quit / exit    Stop the simulation                      ║
║  help           Show this help message                   ║
╚══════════════════════════════════════════════════════════╝
"""


def parse_command(raw: str) -> Optional[CLICommand]:
    """Parse a raw input string into a CLICommand, or None if invalid.

    Does NOT print errors — the caller (CLI thread) handles user feedback.
    """
    parts = raw.strip().lower().split()
    if not parts:
        return None

    verb = parts[0]

    # Handle aliases
    if verb in ("exit", "q"):
        verb = "quit"

    # Match verb
    try:
        cmd_type = CLICommandType(verb)
    except ValueError:
        return None

    # Commands requiring animal ID
    if cmd_type in _ANIMAL_COMMANDS:
        if len(parts) < 2:
            return None
        try:
            animal_id = int(parts[1])
        except ValueError:
            return None
        return CLICommand(command=cmd_type, animal_id=animal_id, raw_input=raw)

    return CLICommand(command=cmd_type, raw_input=raw)


def _cli_thread_fn(
    command_queue: Queue[CLICommand],
    stop_event: threading.Event,
    prompt: str = "herd> ",
) -> None:
    """Target function for the CLI input thread.

    Reads lines from stdin, parses them, and pushes valid commands into
    the thread-safe queue. Runs until `stop_event` is set or EOF.
    """
    try:
        while not stop_event.is_set():
            try:
                raw = input(prompt)
            except EOFError:
                break

            cmd = parse_command(raw)
            if cmd is None:
                if raw.strip():
                    print(f"  Unknown command: '{raw.strip()}'. Type 'help' for options.")
                continue

            if cmd.command == CLICommandType.HELP:
                print(HELP_TEXT)
                continue

            # Push to queue for the simulator to process
            command_queue.put(cmd)

            if cmd.command == CLICommandType.QUIT:
                break

    except KeyboardInterrupt:
        # Ctrl+C in the CLI thread — signal quit
        command_queue.put(CLICommand(command=CLICommandType.QUIT, raw_input="^C"))


def start_cli_thread(
    command_queue: Queue[CLICommand],
    stop_event: threading.Event,
) -> threading.Thread:
    """Start the CLI input thread (daemon — dies with the main process).

    Returns the thread handle (mainly for testing; normal code ignores it).
    """
    t = threading.Thread(
        target=_cli_thread_fn,
        args=(command_queue, stop_event),
        daemon=True,
        name="live-cli",
    )
    t.start()
    return t
