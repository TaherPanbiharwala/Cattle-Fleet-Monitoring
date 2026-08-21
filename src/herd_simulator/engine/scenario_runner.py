"""
scenario_runner.py — Declarative JSON scenario parser & event state manager.

Loads a JSON scenario file containing timed events, manages active event
state per animal, and provides query methods so the tick loop can ask
"is animal X currently in a fever / breach / isolation / etc?".

Supports 6 event types (ADR-014):
  - fever_onset:       Ramp body temp up (onset → plateau → recovery)
  - heat_stress:       Boost ambient temperature for THI spike
  - geofence_breach:   Drive animal outside pasture polygon
  - tamper:            Flag collar as tampered
  - social_isolation:  Flag animal as isolated (straggler drift)
  - collar_dropout:    Kill battery to 0%, cease transmissions

Events can overlap: fever + breach on the same animal composes correctly
because each event type is tracked independently. The risk engine's
product rule naturally handles multi-signal composition.

Event composition order (ADR-014):
  Physiology → Movement → Social State → Collar Faults → Risk → Transmission

This module owns the *state* of events (active/inactive, elapsed time,
parameters). The simulator tick loop owns the *composition order* — it
calls into physiology.py, movement.py, etc. with the event flags from here.

References:
  ADR-014: Dual Fault-Injection Engine (Declarative JSON + Live CLI)
  HerdSimulator PRD §6.6 (FR-15..FR-17, FR-31)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class EventType(str, Enum):
    """The 6 supported anomaly event types."""
    FEVER_ONSET = "fever_onset"
    HEAT_STRESS = "heat_stress"
    GEOFENCE_BREACH = "geofence_breach"
    TAMPER = "tamper"
    SOCIAL_ISOLATION = "social_isolation"
    COLLAR_DROPOUT = "collar_dropout"


# Default parameters for each event type (used when scenario JSON omits them)
_DEFAULT_PARAMS: dict[EventType, dict[str, Any]] = {
    EventType.FEVER_ONSET: {
        "peak_offset_c": 1.8,      # °C above baseline at plateau
        "onset_s": 300,            # 5 min ramp up
        "plateau_s": 600,          # 10 min hold
        "recovery_s": 300,         # 5 min ramp down
    },
    EventType.HEAT_STRESS: {
        "ambient_boost_c": 8.0,    # °C added to ambient temperature
    },
    EventType.GEOFENCE_BREACH: {
        "outward_m": 20.0,         # How far outside the boundary to push
    },
    EventType.TAMPER: {},          # Binary flag — no continuous params
    EventType.SOCIAL_ISOLATION: {
        "drift_rate_m_per_s": 0.15,
        "max_extra_m": 120.0,
    },
    EventType.COLLAR_DROPOUT: {},  # Binary flag — kills battery to 0
}


@dataclass
class ScenarioEvent:
    """A single parsed event from the scenario JSON."""
    sim_second: int                # When to activate (simulation seconds)
    animal_id: int                 # Which animal
    event_type: EventType          # What kind of anomaly
    params: dict[str, Any]         # Event-specific parameters
    event_id: Optional[str] = None # Unique ID for clearing via CLI/API


@dataclass
class ActiveEvent:
    """Runtime state for a currently active event on an animal."""
    event: ScenarioEvent
    activated_at: int              # sim_second when it was activated
    cleared: bool = False          # Set True by clear command


@dataclass
class EventState:
    """All active events, indexed by (animal_id, event_type)."""
    # (animal_id, EventType) → ActiveEvent
    active: dict[tuple[int, str], ActiveEvent] = field(default_factory=dict)
    # Monotonic counter for generating event IDs
    _next_id: int = field(default=1, repr=False)


class ScenarioError(Exception):
    """Invalid scenario file content."""


def load_scenario(path: str | Path) -> list[ScenarioEvent]:
    """Parse a scenario JSON file into a sorted list of ScenarioEvents.

    Expected format:
    [
      {"sim_second": 300, "animal_id": 5, "event": "fever_onset",
       "params": {"peak_offset_c": 1.8}},
      ...
    ]

    Events are sorted by sim_second for efficient processing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ScenarioError(f"Scenario must be a JSON array, got {type(raw).__name__}")

    events: list[ScenarioEvent] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ScenarioError(f"Event [{i}] must be a JSON object, got {type(entry).__name__}")

        # Required fields
        try:
            sim_second = int(entry["sim_second"])
            animal_id = int(entry["animal_id"])
            event_type_str = str(entry["event"])
        except KeyError as e:
            raise ScenarioError(f"Event [{i}] missing required field: {e}")
        except (ValueError, TypeError) as e:
            raise ScenarioError(f"Event [{i}] invalid field value: {e}")

        # Validate event type
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            valid = [e.value for e in EventType]
            raise ScenarioError(
                f"Event [{i}] unknown event type '{event_type_str}'. "
                f"Valid types: {valid}"
            )

        # Merge params with defaults
        user_params = entry.get("params", {})
        if not isinstance(user_params, dict):
            raise ScenarioError(f"Event [{i}] 'params' must be a dict, got {type(user_params).__name__}")
        params = {**_DEFAULT_PARAMS[event_type], **user_params}

        events.append(ScenarioEvent(
            sim_second=sim_second,
            animal_id=animal_id,
            event_type=event_type,
            params=params,
        ))

    # Sort by activation time
    events.sort(key=lambda e: e.sim_second)
    return events


def new_event_state() -> EventState:
    """Create empty event state for the start of a run."""
    return EventState()


def activate_event(
    state: EventState,
    animal_id: int,
    event_type: EventType,
    sim_second: int,
    params: Optional[dict[str, Any]] = None,
) -> str:
    """Activate an event on an animal. Returns the event_id.

    If an event of the same type is already active on this animal,
    it is replaced (the new one takes over).
    """
    event_id = f"evt-{state._next_id}"
    state._next_id += 1

    merged_params = {**_DEFAULT_PARAMS[event_type], **(params or {})}

    event = ScenarioEvent(
        sim_second=sim_second,
        animal_id=animal_id,
        event_type=event_type,
        params=merged_params,
        event_id=event_id,
    )

    key = (animal_id, event_type.value)
    state.active[key] = ActiveEvent(event=event, activated_at=sim_second)
    return event_id


def clear_event(state: EventState, animal_id: int, event_type: EventType) -> bool:
    """Clear (deactivate) a specific event type on an animal.

    Returns True if an event was found and cleared, False if none was active.
    """
    key = (animal_id, event_type.value)
    if key in state.active:
        state.active[key].cleared = True
        del state.active[key]
        return True
    return False


def clear_all_events(state: EventState, animal_id: int) -> int:
    """Clear all active events on a specific animal. Returns count cleared."""
    to_remove = [k for k in state.active if k[0] == animal_id]
    for k in to_remove:
        state.active[k].cleared = True
        del state.active[k]
    return len(to_remove)


def clear_event_by_id(state: EventState, event_id: str) -> bool:
    """Clear an event by its unique event_id (for REST API DELETE)."""
    for key, active in list(state.active.items()):
        if active.event.event_id == event_id:
            active.cleared = True
            del state.active[key]
            return True
    return False


def is_event_active(state: EventState, animal_id: int, event_type: EventType) -> bool:
    """Check if a specific event type is currently active on an animal."""
    key = (animal_id, event_type.value)
    return key in state.active and not state.active[key].cleared


def get_active_event(
    state: EventState,
    animal_id: int,
    event_type: EventType,
) -> Optional[ActiveEvent]:
    """Get the active event details for a specific (animal, type) pair."""
    key = (animal_id, event_type.value)
    active = state.active.get(key)
    if active and not active.cleared:
        return active
    return None


def get_all_active_for_animal(state: EventState, animal_id: int) -> list[ActiveEvent]:
    """Get all active events for a specific animal."""
    return [
        ae for (aid, _), ae in state.active.items()
        if aid == animal_id and not ae.cleared
    ]


def has_any_anomaly(state: EventState, animal_id: int) -> bool:
    """Check if an animal has ANY active anomaly (used for restless boost)."""
    return any(
        aid == animal_id and not ae.cleared
        for (aid, _), ae in state.active.items()
    )


def process_scheduled_events(
    state: EventState,
    scenario_events: list[ScenarioEvent],
    sim_second: int,
    scenario_cursor: int,
) -> tuple[int, list[str]]:
    """Activate any scenario events whose sim_second has arrived.

    Args:
        state: Current event state.
        scenario_events: Sorted list of all scenario events.
        sim_second: Current simulation second.
        scenario_cursor: Index into scenario_events of the next unprocessed event.

    Returns:
        (new_cursor, list_of_activated_event_ids)
    """
    activated: list[str] = []
    cursor = scenario_cursor

    while cursor < len(scenario_events):
        evt = scenario_events[cursor]
        if evt.sim_second > sim_second:
            break  # Events are sorted — no more to process this tick
        # Activate this event
        event_id = activate_event(
            state,
            evt.animal_id,
            evt.event_type,
            sim_second,
            evt.params,
        )
        activated.append(event_id)
        cursor += 1

    return cursor, activated


def get_event_codes_for_status(state: EventState, animal_id: int) -> list[str]:
    """Get ThingSpeak status field event codes for an animal.

    Returns a list like ["FEVER", "BREACH"] for the `status` field
    event encoding (ADR-005: `id=XX;evt=YY;src=ZZ`).
    """
    _TYPE_TO_CODE = {
        EventType.FEVER_ONSET: "FEVER",
        EventType.HEAT_STRESS: "HEAT",
        EventType.GEOFENCE_BREACH: "BREACH",
        EventType.TAMPER: "TAMPER",
        EventType.SOCIAL_ISOLATION: "ISOL",
        EventType.COLLAR_DROPOUT: "DROPOUT",
    }
    codes: list[str] = []
    for (aid, _), ae in state.active.items():
        if aid == animal_id and not ae.cleared:
            code = _TYPE_TO_CODE.get(ae.event.event_type, "UNKNOWN")
            codes.append(code)
    return codes
