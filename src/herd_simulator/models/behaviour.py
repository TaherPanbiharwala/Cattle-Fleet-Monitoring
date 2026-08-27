"""
behaviour.py — 5-state behavioural Markov state machine.

Implements the resting/grazing/ruminating/walking/restless state machine
from HerdSimulator PRD §9.1 and ADR-006. Code 5 (Other/Unknown) is
reserved for low-confidence ML output and is never assigned by this
state machine.

Transitions:
  - Follow the fixed graph mirrored in config `behaviour.base_transitions`
    (see `VALID_TRANSITIONS` below, matching the §9.1 state diagram).
  - Are weighted by time of day (FR-5): dawn/dusk favor grazing/walking,
    midday/night favor resting/ruminating.
  - Get an additive boost toward Restless when a fever/heat/isolation
    anomaly is active on the animal (FR-6). The boost only ever applies
    to the Resting -> Restless edge, since that is the only edge in the
    graph that reaches Restless — anomalies do not invent new edges.
  - Any transition outside the fixed graph is rejected rather than
    silently applied (Master PRD: "Invalid transitions are rejected").

References:
  ADR-006: 6-State Behavioural Classification & Safe Code Mapping
  Master PRD: "Animal profiles and behaviour"
  HerdSimulator PRD §6.2 (FR-4..FR-6), §9.1
"""

from __future__ import annotations

import random
from enum import IntEnum

from herd_simulator.config import BehaviourConfig


class Behaviour(IntEnum):
    """Behaviour codes (ADR-006 / AGENTS.md §4.1 field3)."""

    RESTING = 0
    GRAZING = 1
    RUMINATING = 2
    WALKING = 3
    RESTLESS = 4
    OTHER = 5  # reserved for low-confidence ML output; never emitted here


# Canonical valid transition graph — mirrors the §9.1 state diagram and the
# default_config.yaml `behaviour.base_transitions` rows exactly. Any target
# not listed here for a given source state is rejected by
# `validate_transition`, regardless of what a (possibly misconfigured) YAML
# file claims.
VALID_TRANSITIONS: dict[Behaviour, frozenset[Behaviour]] = {
    Behaviour.RESTING: frozenset({Behaviour.GRAZING, Behaviour.RESTLESS}),
    Behaviour.GRAZING: frozenset({Behaviour.RUMINATING, Behaviour.WALKING}),
    Behaviour.RUMINATING: frozenset({Behaviour.RESTING}),
    Behaviour.WALKING: frozenset({Behaviour.GRAZING}),
    Behaviour.RESTLESS: frozenset({Behaviour.RESTING}),
}

_NAME_TO_BEHAVIOUR = {b.name.lower(): b for b in Behaviour if b != Behaviour.OTHER}

# FR-5: hours where activity states (grazing/walking) are boosted, and hours
# where rest states (resting/ruminating) are boosted.
_ACTIVE_HOURS = frozenset({5, 6, 7, 17, 18, 19})  # dawn / dusk
_REST_HOURS = frozenset({0, 1, 2, 3, 11, 12, 13, 22, 23})  # midday / night
_TIME_OF_DAY_MULTIPLIER = 1.5

# FR-6: additive probability boost on the Resting -> Restless edge when a
# fever / heat-stress / isolation anomaly is active on this animal.
ANOMALY_RESTLESS_BOOST = 0.30


class InvalidTransitionError(Exception):
    """A proposed behaviour transition is not in the valid graph (ADR-006)."""


def validate_transition(current: Behaviour, target: Behaviour) -> None:
    """Raise InvalidTransitionError unless `target` is `current` (stay) or a
    listed edge out of `current`."""
    if target != current and target not in VALID_TRANSITIONS[current]:
        raise InvalidTransitionError(
            f"{current.name} -> {target.name} is not a valid transition "
            f"(allowed: {sorted(b.name for b in VALID_TRANSITIONS[current])})"
        )


def validate_config_transitions(cfg: BehaviourConfig) -> None:
    """Raise InvalidTransitionError if `cfg.base_transitions` declares any
    edge outside the canonical `VALID_TRANSITIONS` graph.

    Config loading (config.py) only checks that target names are *known
    states*; it does not know the state-diagram topology. Callers (the
    future engine startup path, and tests) should run this once per loaded
    config to catch a hand-edited YAML that adds an illegal edge.
    """
    for state in Behaviour:
        if state == Behaviour.OTHER:
            continue
        row = getattr(cfg.base_transitions, state.name.lower())
        for target_name in row:
            validate_transition(state, _NAME_TO_BEHAVIOUR[target_name])


def transition_weights(
    state: Behaviour,
    cfg: BehaviourConfig,
    hour_of_day: float,
    anomaly_active: bool = False,
) -> dict[Behaviour, float]:
    """Return {target_state: probability} of leaving `state` this tick.

    The remaining probability mass (1 - sum(weights)) is the chance of
    staying in `state`. Weights are time-of-day adjusted and, only for the
    Resting state, anomaly-boosted toward Restless.
    """
    row = getattr(cfg.base_transitions, state.name.lower())
    hour = int(hour_of_day) % 24

    weights: dict[Behaviour, float] = {}
    for target_name, prob in row.items():
        target = _NAME_TO_BEHAVIOUR[target_name]
        adjusted = prob
        if hour in _ACTIVE_HOURS and target in (Behaviour.GRAZING, Behaviour.WALKING):
            adjusted *= _TIME_OF_DAY_MULTIPLIER
        elif hour in _REST_HOURS and target in (Behaviour.RESTING, Behaviour.RUMINATING):
            adjusted *= _TIME_OF_DAY_MULTIPLIER
        weights[target] = adjusted

    if anomaly_active and Behaviour.RESTLESS in VALID_TRANSITIONS[state]:
        weights[Behaviour.RESTLESS] = weights.get(Behaviour.RESTLESS, 0.0) + ANOMALY_RESTLESS_BOOST

    total = sum(weights.values())
    if total > 1.0:
        weights = {target: prob / total for target, prob in weights.items()}

    return weights


def step(
    state: Behaviour,
    cfg: BehaviourConfig,
    hour_of_day: float,
    rng: random.Random,
    anomaly_active: bool = False,
) -> Behaviour:
    """Evaluate one transition_interval_s tick and return the resulting
    state (which may be unchanged).

    Draws exactly one uniform sample from `rng` so that, for a fixed rng
    stream, the sequence of states produced is fully deterministic
    (AGENTS.md golden rule 4).
    """
    weights = transition_weights(state, cfg, hour_of_day, anomaly_active)
    roll = rng.random()
    cumulative = 0.0
    for target, prob in weights.items():
        cumulative += prob
        if roll < cumulative:
            return target
    return state
