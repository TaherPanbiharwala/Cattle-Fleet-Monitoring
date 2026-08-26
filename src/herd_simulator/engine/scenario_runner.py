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
product rule naturally handles multi-signal composition. The one thing
that must NOT overlap is the same event type with itself on the same
animal (Master PRD "Scenario contract") — that is rejected at load time,
not silently allowed.

Event composition order (ADR-014):
  Physiology → Movement → Social State → Collar Faults → Risk → Transmission

This module owns the *state* of events (active/inactive, elapsed time,
parameters, expiry). The simulator tick loop owns the *composition
order* — it calls into physiology.py, movement.py, etc. with the event
flags from here.

References:
  ADR-014: Dual Fault-Injection Engine (Declarative JSON + Live CLI)
  HerdSimulator PRD §6.6 (FR-15..FR-17, FR-31)
  Master PRD: "Scenario contract"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Optional

SCHEMA_VERSION = 1


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
    """A single parsed event from the scenario JSON.

    `duration_seconds` is required for scenario-file events (Master PRD:
    "Each event contains animal_id, type, start_sim_second,
    duration_seconds, and typed parameters") — the event auto-clears
    `duration_seconds` after `start_sim_second`. It is left `None` for
    events activated directly through the live CLI/API, which have no
    scripted duration and run until an explicit `clear` command instead.
    """
    animal_id: int                          # Which animal
    event_type: EventType                   # What kind of anomaly
    start_sim_second: int                   # When to activate (simulation seconds)
    params: dict[str, Any]                  # Event-specific parameters
    duration_seconds: Optional[int] = None  # How long it stays active; None = manual clear
    event_id: Optional[str] = None          # Unique ID for clearing via CLI/API


@dataclass
class Scenario:
    """A fully parsed scenario file: metadata plus its sorted events."""
    schema_version: int
    scenario_id: str
    seed: int
    events: list[ScenarioEvent]


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


def load_scenario(path: str | Path, valid_animal_ids: Optional[Iterable[int]] = None) -> Scenario:
    """Parse a scenario JSON file into a validated `Scenario`.

    Expected format (Master PRD "Scenario contract"):
        {
          "schema_version": 1,
          "scenario_id": "demo_scenario",
          "seed": 42,
          "events": [
            {"animal_id": 5, "type": "fever_onset", "start_sim_second": 300,
             "duration_seconds": 900, "params": {"peak_offset_c": 1.8}},
            ...
          ]
        }

    Validated before startup, per the Master PRD's binding rule ("Unknown
    types, duplicate event IDs, invalid cattle IDs, and non-positive
    duration fail before startup. The same event type cannot overlap
    itself for the same cow."):
      - `event.type` must be one of the 6 supported EventType values.
      - `event.event_id` (if supplied) must be unique across the file;
        omitted IDs are auto-generated from `scenario_id` + index.
      - `event.animal_id` must be in `valid_animal_ids`, when supplied by
        the caller (the herd's actual ID range) — skipped if not supplied,
        so callers without a config on hand can still parse a scenario.
      - `event.duration_seconds` must be a positive integer.
      - No two events of the same `type` on the same `animal_id` may have
        overlapping `[start_sim_second, start_sim_second + duration_seconds)`
        windows.

    Events are returned sorted by `start_sim_second`.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with open(path) as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ScenarioError(
            f"Scenario must be a JSON object with schema_version/scenario_id/seed/events, "
            f"got {type(raw).__name__}"
        )

    schema_version = raw.get("schema_version")
    if schema_version != SCHEMA_VERSION:
        raise ScenarioError(
            f"Unsupported schema_version {schema_version!r}, expected {SCHEMA_VERSION}"
        )

    scenario_id = raw.get("scenario_id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ScenarioError(f"scenario_id must be a non-empty string, got {scenario_id!r}")

    seed = raw.get("seed")
    if not isinstance(seed, int):
        raise ScenarioError(f"seed must be an integer, got {seed!r}")

    raw_events = raw.get("events")
    if not isinstance(raw_events, list):
        raise ScenarioError(f"events must be a JSON array, got {type(raw_events).__name__}")

    valid_ids = set(valid_animal_ids) if valid_animal_ids is not None else None

    events: list[ScenarioEvent] = []
    seen_ids: set[str] = set()
    for i, entry in enumerate(raw_events):
        if not isinstance(entry, dict):
            raise ScenarioError(f"Event [{i}] must be a JSON object, got {type(entry).__name__}")

        # Required fields
        try:
            animal_id = int(entry["animal_id"])
            event_type_str = str(entry["type"])
            start_sim_second = int(entry["start_sim_second"])
            duration_seconds = int(entry["duration_seconds"])
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
                f"Event [{i}] unknown event type '{event_type_str}'. Valid types: {valid}"
            )

        # Validate duration ("non-positive duration fail before startup")
        if duration_seconds <= 0:
            raise ScenarioError(
                f"Event [{i}] duration_seconds must be positive, got {duration_seconds}"
            )

        # Validate cattle ID ("invalid cattle IDs fail before startup")
        if valid_ids is not None and animal_id not in valid_ids:
            raise ScenarioError(
                f"Event [{i}] animal_id {animal_id} is not a member of this herd"
            )

        # Event ID: explicit or auto-generated, always unique ("duplicate
        # event IDs fail before startup")
        event_id = entry.get("event_id")
        if event_id is None:
            event_id = f"{scenario_id}-{i}"
        elif not isinstance(event_id, str):
            raise ScenarioError(f"Event [{i}] event_id must be a string, got {event_id!r}")
        if event_id in seen_ids:
            raise ScenarioError(f"Event [{i}] duplicate event_id '{event_id}'")
        seen_ids.add(event_id)

        # Merge params with defaults
        user_params = entry.get("params", {})
        if not isinstance(user_params, dict):
            raise ScenarioError(f"Event [{i}] 'params' must be a dict, got {type(user_params).__name__}")
        params = {**_DEFAULT_PARAMS[event_type], **user_params}

        events.append(ScenarioEvent(
            animal_id=animal_id,
            event_type=event_type,
            start_sim_second=start_sim_second,
            duration_seconds=duration_seconds,
            params=params,
            event_id=event_id,
        ))

    # Sort by activation time before the overlap check, so overlap windows
    # can be compared in a single forward pass per (animal, type) group.
    events.sort(key=lambda e: e.start_sim_second)
    _validate_no_self_overlap(events)

    return Scenario(schema_version=schema_version, scenario_id=scenario_id, seed=seed, events=events)


def _validate_no_self_overlap(events: list[ScenarioEvent]) -> None:
    """Master PRD: "The same event type cannot overlap itself for the
    same cow." Different event types on the same animal may overlap
    freely (that's the whole point of composable faults) — only a
    repeat of the *same* type on the *same* animal is rejected."""
    by_key: dict[tuple[int, EventType], list[ScenarioEvent]] = {}
    for evt in events:
        by_key.setdefault((evt.animal_id, evt.event_type), []).append(evt)

    for (animal_id, event_type), group in by_key.items():
        ordered = sorted(group, key=lambda e: e.start_sim_second)
        for prev, nxt in zip(ordered, ordered[1:]):
            prev_end = prev.start_sim_second + (prev.duration_seconds or 0)
            if nxt.start_sim_second < prev_end:
                raise ScenarioError(
                    f"Overlapping {event_type.value} events for animal {animal_id}: "
                    f"[{prev.start_sim_second}, {prev_end}) and "
                    f"[{nxt.start_sim_second}, {nxt.start_sim_second + (nxt.duration_seconds or 0)})"
                )


def new_event_state() -> EventState:
    """Create empty event state for the start of a run."""
    return EventState()


def activate_event(
    state: EventState,
    animal_id: int,
    event_type: EventType,
    sim_second: int,
    params: Optional[dict[str, Any]] = None,
    duration_seconds: Optional[int] = None,
    event_id: Optional[str] = None,
) -> str:
    """Activate an event on an animal. Returns the event_id.

    If an event of the same type is already active on this animal, it is
    replaced (the new one takes over) — this is the live/interactive
    activation path (CLI, REST API), where re-triggering the same fault
    is a normal "update it" gesture, not a scripted-timeline conflict.
    The "cannot overlap itself" rule is enforced separately, at scenario
    *load* time, across the static scripted timeline (see
    `_validate_no_self_overlap`).

    `duration_seconds=None` (the default, used by CLI/API callers) means
    the event runs until explicitly cleared. Scenario-sourced events pass
    their parsed `duration_seconds` so `expire_events` can auto-clear them.

    `event_id`, if provided, is used as-is (scenario events already carry
    a load-time-validated ID); otherwise one is auto-generated.
    """
    if event_id is None:
        event_id = f"evt-{state._next_id}"
        state._next_id += 1

    merged_params = {**_DEFAULT_PARAMS[event_type], **(params or {})}

    event = ScenarioEvent(
        animal_id=animal_id,
        event_type=event_type,
        start_sim_second=sim_second,
        duration_seconds=duration_seconds,
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


def expire_events(state: EventState, sim_second: int) -> list[str]:
    """Auto-clear any active event whose scripted duration has elapsed.

    Only events with a non-None `duration_seconds` are eligible — events
    activated live via CLI/API (`duration_seconds=None`) run until an
    explicit `clear` command, by design. Returns the event_ids cleared.

    Intended to be called once per tick, alongside `process_scheduled_events`.
    """
    expired: list[str] = []
    for key, ae in list(state.active.items()):
        duration = ae.event.duration_seconds
        if duration is not None and sim_second - ae.activated_at >= duration:
            ae.cleared = True
            del state.active[key]
            expired.append(ae.event.event_id or "")
    return expired


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
    """Activate any scenario events whose start_sim_second has arrived.

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
        if evt.start_sim_second > sim_second:
            break  # Events are sorted — no more to process this tick
        # Activate this event, carrying through its scripted duration and
        # load-time-validated event_id.
        event_id = activate_event(
            state,
            evt.animal_id,
            evt.event_type,
            sim_second,
            evt.params,
            duration_seconds=evt.duration_seconds,
            event_id=evt.event_id,
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
