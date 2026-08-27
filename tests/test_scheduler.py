"""
test_scheduler.py — Tests for the round-robin multiplexer & priority queue.

Validates:
  - 15s floor enforcement (ADR-004)
  - Round-robin cycling (no starvation)
  - Priority jump semantics
  - Full sweep accounting
  - Queue snapshot serialization
"""

from __future__ import annotations

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

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


@pytest.fixture
def cfg() -> SchedulerConfig:
    return SchedulerConfig(
        animal_ids=list(range(2, 21)),  # IDs 2..20 = 19 animals
        normal_cadence_s=30,
        alert_cadence_s=15,
        min_interval_s=15,
    )


@pytest.fixture
def sched(cfg: SchedulerConfig) -> SchedulerState:
    return new_scheduler(cfg)


# ===================================================================
# 15-Second Floor Enforcement (ADR-004)
# ===================================================================

class TestFloorEnforcement:
    """The scheduler must NEVER allow a write faster than min_interval_s."""

    def test_first_write_always_allowed(self, sched: SchedulerState):
        assert is_write_allowed(sched, now=0.0, min_interval_s=15)

    def test_write_blocked_before_interval(self, sched: SchedulerState):
        record_write(sched, now=100.0)
        assert not is_write_allowed(sched, now=110.0, min_interval_s=15)

    def test_write_allowed_at_exact_interval(self, sched: SchedulerState):
        record_write(sched, now=100.0)
        assert is_write_allowed(sched, now=115.0, min_interval_s=15)

    def test_write_allowed_after_interval(self, sched: SchedulerState):
        record_write(sched, now=100.0)
        assert is_write_allowed(sched, now=200.0, min_interval_s=15)


# ===================================================================
# Round-Robin Cycling
# ===================================================================

class TestRoundRobin:
    """All 19 simulated IDs must cycle without starvation."""

    def test_full_sweep_covers_all_ids(self, sched: SchedulerState, cfg: SchedulerConfig):
        """One full round-robin sweep serves every ID exactly once."""
        served: list[int] = []
        for _ in range(19):
            aid = next_animal(sched, cfg)
            assert aid is not None
            served.append(aid)
        assert set(served) == set(range(2, 21))

    def test_sweep_counter_increments(self, sched: SchedulerState, cfg: SchedulerConfig):
        assert sched.sweeps_completed == 0
        for _ in range(19):
            next_animal(sched, cfg)
        assert sched.sweeps_completed == 1

    def test_two_sweeps(self, sched: SchedulerState, cfg: SchedulerConfig):
        for _ in range(38):
            next_animal(sched, cfg)
        assert sched.sweeps_completed == 2

    def test_rr_order_is_deterministic(self, cfg: SchedulerConfig):
        s1 = new_scheduler(cfg)
        s2 = new_scheduler(cfg)
        seq1 = [next_animal(s1, cfg) for _ in range(19)]
        seq2 = [next_animal(s2, cfg) for _ in range(19)]
        assert seq1 == seq2


# ===================================================================
# Priority Queue
# ===================================================================

class TestPriorityQueue:
    """Priority items are served before round-robin."""

    def test_priority_animal_served_first(self, sched: SchedulerState, cfg: SchedulerConfig):
        enqueue_priority(sched, 15)
        aid = next_animal(sched, cfg)
        assert aid == 15

    def test_priority_drains_before_rr(self, sched: SchedulerState, cfg: SchedulerConfig):
        enqueue_priority(sched, 10)
        enqueue_priority(sched, 12)
        assert next_animal(sched, cfg) == 10
        assert next_animal(sched, cfg) == 12
        # Next should be from round-robin
        rr_first = next_animal(sched, cfg)
        assert rr_first == 2  # Head of the RR queue

    def test_duplicate_priority_suppressed(self, sched: SchedulerState):
        enqueue_priority(sched, 7)
        enqueue_priority(sched, 7)
        assert len(sched.priority_queue) == 1

    def test_priority_does_not_duplicate_existing(self, sched: SchedulerState):
        enqueue_priority(sched, 5)
        enqueue_priority(sched, 5)
        enqueue_priority(sched, 5)
        assert list(sched.priority_queue).count(5) == 1


# ===================================================================
# Cadence Selection
# ===================================================================

class TestCadenceSelection:
    """Alert cadence while priority queue is non-empty, normal otherwise."""

    def test_normal_cadence_when_empty(self, sched: SchedulerState, cfg: SchedulerConfig):
        assert next_cadence_s(sched, cfg) == 30

    def test_alert_cadence_when_priority(self, sched: SchedulerState, cfg: SchedulerConfig):
        enqueue_priority(sched, 5)
        assert next_cadence_s(sched, cfg) == 15


# ===================================================================
# Queue Snapshot
# ===================================================================

class TestQueueSnapshot:
    """Snapshot for REST API / HUD rendering."""

    def test_snapshot_is_dict(self, sched: SchedulerState):
        snap = get_queue_snapshot(sched)
        assert isinstance(snap, dict)
        assert "priority_queue" in snap
        assert "rr_next_5" in snap
        assert "total_writes" in snap

    def test_snapshot_reflects_writes(self, sched: SchedulerState):
        record_write(sched, 100.0)
        snap = get_queue_snapshot(sched)
        assert snap["total_writes"] == 1

    def test_snapshot_rr_next_5(self, sched: SchedulerState):
        snap = get_queue_snapshot(sched)
        assert snap["rr_next_5"] == [2, 3, 4, 5, 6]
