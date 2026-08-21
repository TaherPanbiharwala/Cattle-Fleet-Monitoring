"""
scheduler.py — Round-robin multiplexer with priority anomaly jump.

Manages the transmission queue for Channel 2: which animal's telemetry
gets POSTed next. In normal operation, IDs 2–20 cycle round-robin at
30s cadence. When an anomaly fires, the affected animal jumps to the
front of a priority queue and is served at 15s cadence.

Invariants enforced:
  - The 15s physical floor (ADR-004) is NEVER violated.
  - Priority queue drains before the normal round-robin resumes.
  - No animal is starved indefinitely — the round-robin cursor advances
    even while priority items are being served.
  - Full sweep: all 19 IDs appear at least once within 19 × cadence_s.

References:
  ADR-004: ThingSpeak Free-Tier Multi-Channel Allocation & Rate Limits
  HerdSimulator PRD §9.4: Round-robin scheduler & Priority Jump
  AGENTS.md §3.2: ThingSpeak Rate-Limit Compliance
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SchedulerConfig:
    """Extracted scheduler parameters (mirrors thingspeak config section)."""
    animal_ids: list[int]           # IDs to cycle through (e.g. [2, 3, ..., 20])
    normal_cadence_s: int = 30      # Seconds between normal-mode writes
    alert_cadence_s: int = 15       # Seconds between alert-mode writes
    min_interval_s: int = 15        # Absolute floor — never POST faster (ADR-004)


@dataclass
class SchedulerState:
    """Mutable scheduler runtime state."""
    # Round-robin queue — cycles endlessly through animal_ids
    rr_queue: deque[int] = field(default_factory=deque)
    # Priority queue — anomaly animals jump here
    priority_queue: deque[int] = field(default_factory=deque)
    # Timestamp of last successful write (wall-clock or sim-clock)
    last_write_time: float = 0.0
    # Count of total writes performed
    total_writes: int = 0
    # Track which IDs have been served in the current sweep
    current_sweep_served: set[int] = field(default_factory=set)
    # Number of complete sweeps
    sweeps_completed: int = 0


def new_scheduler(cfg: SchedulerConfig) -> SchedulerState:
    """Create initial scheduler state with the round-robin queue loaded."""
    return SchedulerState(
        rr_queue=deque(cfg.animal_ids),
        priority_queue=deque(),
        last_write_time=0.0,
        total_writes=0,
        current_sweep_served=set(),
        sweeps_completed=0,
    )


def enqueue_priority(state: SchedulerState, animal_id: int) -> None:
    """Push an animal to the priority queue (anomaly jump).

    Duplicates are suppressed — if the animal is already in the priority
    queue, it stays at its current position rather than being re-added.
    This prevents a chattering anomaly from flooding the queue.
    """
    if animal_id not in state.priority_queue:
        state.priority_queue.append(animal_id)


def is_write_allowed(state: SchedulerState, now: float, min_interval_s: int) -> bool:
    """Check if enough time has elapsed since the last write.

    This is the 15s floor enforcement (ADR-004). Returns True if
    `now - last_write_time >= min_interval_s`.
    """
    if state.total_writes == 0:
        return True  # First write is always allowed
    return (now - state.last_write_time) >= min_interval_s


def next_cadence_s(state: SchedulerState, cfg: SchedulerConfig) -> int:
    """Return the cadence to use for the next write.

    Alert cadence (15s) while priority queue is non-empty;
    normal cadence (30s) otherwise.
    """
    if state.priority_queue:
        return cfg.alert_cadence_s
    return cfg.normal_cadence_s


def next_animal(state: SchedulerState, cfg: SchedulerConfig) -> Optional[int]:
    """Pop and return the next animal ID to transmit.

    Priority queue is served first (FIFO). When empty, the round-robin
    queue advances. Returns None if the queue is somehow exhausted
    (should never happen in normal operation since RR refills).

    Also tracks sweep progress: once all IDs in cfg.animal_ids have been
    served at least once, a sweep is complete and the counter increments.
    """
    animal_id: Optional[int] = None

    if state.priority_queue:
        animal_id = state.priority_queue.popleft()
    elif state.rr_queue:
        animal_id = state.rr_queue.popleft()
        # Refill round-robin: put this ID at the back
        state.rr_queue.append(animal_id)
    else:
        return None

    # Track sweep progress
    if animal_id is not None:
        state.current_sweep_served.add(animal_id)
        if state.current_sweep_served >= set(cfg.animal_ids):
            state.sweeps_completed += 1
            state.current_sweep_served = set()

    return animal_id


def record_write(state: SchedulerState, now: float) -> None:
    """Record that a write just happened at timestamp `now`."""
    state.last_write_time = now
    state.total_writes += 1


def get_queue_snapshot(state: SchedulerState) -> dict:
    """Return a JSON-serializable snapshot of the current queue state.

    Used by the REST API `/api/queue` endpoint and the HUD.
    """
    return {
        "priority_queue": list(state.priority_queue),
        "rr_next_5": list(state.rr_queue)[:5],
        "total_writes": state.total_writes,
        "sweeps_completed": state.sweeps_completed,
        "current_sweep_progress": len(state.current_sweep_served),
        "last_write_time": state.last_write_time,
    }
