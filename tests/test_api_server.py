"""Tests for REST API server, HUD state, and callback wiring."""

from __future__ import annotations

import collections
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Optional
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from herd_simulator.config import HudConfig, load_config
from herd_simulator.engine.scenario_runner import EventType, activate_event
from herd_simulator.engine.simulator import (
    AnimalTelemetry,
    SimMode,
    Simulator,
    create_simulator,
    tick,
)
from herd_simulator.services.api_server import (
    DEFAULT_HISTORY_LIMIT,
    HISTORY_LIMIT_MAX,
    MAX_HISTORY_PER_ANIMAL,
    SCHEMA_VERSION,
    HudRequestHandler,
    HudServer,
    HudState,
    create_hud_state,
    wire_api_server,
    _clear_event_with_info,
    _telemetry_to_dict,
    _base_envelope,
    _error_body,
)


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_telemetry(**overrides) -> AnimalTelemetry:
    defaults = dict(
        animal_id=5,
        is_physical=False,
        sim_second=300,
        body_temp_c=39.1,
        thi=72.5,
        behaviour=1,
        latitude=12.9716,
        longitude=79.1589,
        risk_score=25,
        alert_band="green",
        geofence_status=0,
        battery_pct=95.3,
        event_codes=[],
        dropped_out=False,
    )
    defaults.update(overrides)
    return AnimalTelemetry(**defaults)


def _make_hud_config(**overrides) -> HudConfig:
    defaults = dict(host="127.0.0.1", port=0, poll_interval_ms=2000)
    defaults.update(overrides)
    return HudConfig(**defaults)


@pytest.fixture
def cfg():
    return load_config(
        os.path.join(os.path.dirname(__file__), "..", "config", "default_config.yaml")
    )


@pytest.fixture
def sim(cfg):
    return create_simulator(cfg, SimMode.DRY_RUN)


@pytest.fixture
def hud_state(sim):
    return create_hud_state(sim, "test-run-001")


def _cfg_with_port(cfg, port: int):
    """Return config with a specific HUD port."""
    import dataclasses
    return dataclasses.replace(cfg, hud=HudConfig(host="127.0.0.1", port=port, poll_interval_ms=2000))


class _ServerFixture:
    """Manages a running HudServer for integration tests."""

    def __init__(self, cfg, sim, hud_state):
        self.sim = sim
        self.hud_state = hud_state
        self.server = HudServer(cfg.hud, hud_state)
        self.server.start()
        self.port = self.server.port
        self.base_url = f"http://127.0.0.1:{self.port}"

    def get(self, path: str) -> tuple[int, dict[str, Any]]:
        try:
            req = urllib.request.Request(self.base_url + path)
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return resp.status, body
        except urllib.error.HTTPError as e:
            body = json.loads(e.read().decode("utf-8"))
            return e.code, body

    def post(self, path: str, data: dict) -> tuple[int, dict[str, Any]]:
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def delete(self, path: str) -> tuple[int, Optional[dict[str, Any]]]:
        req = urllib.request.Request(self.base_url + path, method="DELETE")
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                raw = resp.read()
                body = json.loads(raw.decode("utf-8")) if raw else None
                return resp.status, body
        except urllib.error.HTTPError as e:
            raw = e.read()
            body = json.loads(raw.decode("utf-8")) if raw else None
            return e.code, body

    def stop(self):
        self.server.stop()


@pytest.fixture
def server(cfg, sim, hud_state):
    cfg_mod = _cfg_with_port(cfg, 0)
    wire_api_server(sim, hud_state)
    tick(sim)
    srv = _ServerFixture(cfg_mod, sim, hud_state)
    yield srv
    srv.stop()


# =======================================================================
# Unit tests: telemetry serialization
# =======================================================================

class TestTelemetryToDict:

    def test_all_fields_present(self):
        t = _make_telemetry()
        d = _telemetry_to_dict(t)
        assert d["animal_id"] == 5
        assert d["is_physical"] is False
        assert d["sim_second"] == 300
        assert d["behaviour"] == 1
        assert d["risk_score"] == 25
        assert d["alert_band"] == "green"
        assert d["geofence_status"] == 0
        assert d["dropped_out"] is False
        assert d["stale"] is False
        assert isinstance(d["event_codes"], list)

    def test_floats_rounded(self):
        t = _make_telemetry(body_temp_c=39.123456, thi=72.98765, latitude=12.97164321)
        d = _telemetry_to_dict(t)
        assert d["body_temp_c"] == 39.12
        assert d["thi"] == 72.99
        assert d["latitude"] == 12.971643


# =======================================================================
# Unit tests: HudState creation
# =======================================================================

class TestHudState:

    def test_create_hud_state_defaults(self, sim):
        hs = create_hud_state(sim, "run-1")
        assert hs.sim is sim
        assert hs.run_id == "run-1"
        assert hs.latest_telemetry == []
        assert hs.history == {}
        assert hs.thingspeak_client is None

    def test_create_hud_state_with_ts_client(self, sim):
        fake_client = object()
        hs = create_hud_state(sim, "run-2", thingspeak_client=fake_client)
        assert hs.thingspeak_client is fake_client

    def test_history_deque_maxlen(self, hud_state):
        dq = collections.deque(maxlen=MAX_HISTORY_PER_ANIMAL)
        hud_state.history[5] = dq
        for i in range(MAX_HISTORY_PER_ANIMAL + 100):
            dq.append(_make_telemetry(sim_second=i))
        assert len(dq) == MAX_HISTORY_PER_ANIMAL


# =======================================================================
# Unit tests: error body format
# =======================================================================

class TestErrorBody:

    def test_basic(self):
        b = _error_body("TEST_CODE", "Something went wrong.")
        assert b["code"] == "TEST_CODE"
        assert b["message"] == "Something went wrong."
        assert "details" not in b

    def test_with_details(self):
        b = _error_body("ERR", "Bad", details={"key": "val"})
        assert b["details"] == {"key": "val"}


# =======================================================================
# Unit tests: base envelope
# =======================================================================

class TestBaseEnvelope:

    def test_contains_required_fields(self, hud_state):
        env = _base_envelope(hud_state)
        assert env["schema_version"] == SCHEMA_VERSION
        assert env["run_id"] == "test-run-001"
        assert "sim_second" in env


# =======================================================================
# Unit tests: clear event helper
# =======================================================================

class TestClearEventWithInfo:

    def test_clears_existing_event(self, sim, hud_state):
        eid = activate_event(
            sim.event_state, 5, EventType.FEVER_ONSET, sim.clock.sim_second,
        )
        result = _clear_event_with_info(hud_state, eid)
        assert result is not None
        assert result[0] == eid
        assert result[1] == 5
        assert result[2] == "fever_onset"
        assert len(sim.event_state.active) == 0

    def test_returns_none_for_missing(self, hud_state):
        result = _clear_event_with_info(hud_state, "nonexistent")
        assert result is None


# =======================================================================
# Wiring tests
# =======================================================================

class TestWireApiServer:

    def test_chains_on_tick_complete(self, sim, hud_state):
        calls = []
        sim.on_tick_complete = lambda t, ss: calls.append(("orig", ss))
        wire_api_server(sim, hud_state)

        batch = [_make_telemetry(animal_id=i) for i in range(2, 21)]
        sim.on_tick_complete(batch, 42)

        assert calls == [("orig", 42)]
        assert len(hud_state.latest_telemetry) == 19

    def test_no_existing_callback(self, sim, hud_state):
        assert sim.on_tick_complete is None
        wire_api_server(sim, hud_state)
        batch = [_make_telemetry(animal_id=2)]
        sim.on_tick_complete(batch, 1)
        assert hud_state.latest_telemetry == batch

    def test_history_appended_per_animal(self, sim, hud_state):
        wire_api_server(sim, hud_state)
        for ss in range(5):
            batch = [_make_telemetry(animal_id=i, sim_second=ss) for i in (2, 3, 4)]
            sim.on_tick_complete(batch, ss)
        assert len(hud_state.history[2]) == 5
        assert len(hud_state.history[3]) == 5
        assert len(hud_state.history[4]) == 5

    def test_latest_telemetry_replaced_each_tick(self, sim, hud_state):
        wire_api_server(sim, hud_state)
        batch1 = [_make_telemetry(sim_second=1)]
        batch2 = [_make_telemetry(sim_second=2)]
        sim.on_tick_complete(batch1, 1)
        sim.on_tick_complete(batch2, 2)
        assert hud_state.latest_telemetry is batch2

    def test_callback_exception_does_not_break_chain(self, sim, hud_state):
        calls = []
        sim.on_tick_complete = lambda t, ss: calls.append(ss)
        wire_api_server(sim, hud_state)

        orig_lock = hud_state.lock
        hud_state.lock = None  # will cause AttributeError in the try block

        sim.on_tick_complete([_make_telemetry()], 99)
        assert calls == [99]
        hud_state.lock = orig_lock


# =======================================================================
# Integration tests: HudServer lifecycle
# =======================================================================

class TestHudServerLifecycle:

    def test_start_stop(self, cfg, hud_state):
        cfg_mod = _cfg_with_port(cfg, 0)
        srv = HudServer(cfg_mod.hud, hud_state)
        srv.start()
        assert srv.port is not None
        assert srv.port > 0
        srv.stop()

    def test_port_property_none_before_start(self, cfg, hud_state):
        cfg_mod = _cfg_with_port(cfg, 0)
        srv = HudServer(cfg_mod.hud, hud_state)
        assert srv.port is None


# =======================================================================
# Integration tests: API endpoints
# =======================================================================

class TestHealthEndpoint:

    def test_returns_200(self, server):
        code, data = server.get("/api/health")
        assert code == 200

    def test_required_fields(self, server):
        _, data = server.get("/api/health")
        assert data["schema_version"] == SCHEMA_VERSION
        assert "run_id" in data
        assert "sim_second" in data
        assert "uptime_seconds" in data
        assert "sim_mode" in data
        assert "herd_size" in data
        assert "active_events" in data
        assert "queue_depth" in data
        assert "memory_rss_mb" in data
        assert "paused" in data

    def test_herd_size_is_20(self, server):
        _, data = server.get("/api/health")
        assert data["herd_size"] == 20


class TestStateEndpoint:

    def test_returns_200(self, server):
        code, data = server.get("/api/state")
        assert code == 200

    def test_required_fields(self, server):
        _, data = server.get("/api/state")
        assert data["schema_version"] == SCHEMA_VERSION
        assert "run_id" in data
        assert "sim_second" in data
        assert "ambient_temp_c" in data
        assert "humidity_pct" in data
        assert "thi" in data
        assert "centroid" in data
        assert "pasture_polygon" in data
        assert "animals" in data
        assert "active_events" in data

    def test_animals_list_has_20(self, server):
        _, data = server.get("/api/state")
        assert len(data["animals"]) == 20

    def test_pasture_polygon_has_vertices(self, server):
        _, data = server.get("/api/state")
        assert len(data["pasture_polygon"]) >= 3

    def test_centroid_is_two_element_list(self, server):
        _, data = server.get("/api/state")
        assert len(data["centroid"]) == 2

    def test_animal_fields(self, server):
        _, data = server.get("/api/state")
        a = data["animals"][0]
        for field in (
            "animal_id", "is_physical", "sim_second", "body_temp_c",
            "thi", "behaviour", "latitude", "longitude", "risk_score",
            "alert_band", "geofence_status", "battery_pct", "event_codes",
            "dropped_out", "stale",
        ):
            assert field in a, f"Missing field: {field}"


class TestHistoryEndpoint:

    def test_returns_records(self, server):
        code, data = server.get("/api/history?id=2&limit=5")
        assert code == 200
        assert "records" in data
        assert "count" in data
        assert data["animal_id"] == 2

    def test_default_limit(self, server):
        for _ in range(5):
            tick(server.sim)
        code, data = server.get("/api/history?id=2")
        assert code == 200
        assert data["count"] <= DEFAULT_HISTORY_LIMIT

    def test_missing_id_returns_400(self, server):
        code, data = server.get("/api/history")
        assert code == 400
        assert data["code"] == "MISSING_PARAMETER"

    def test_invalid_id_returns_400(self, server):
        code, data = server.get("/api/history?id=abc")
        assert code == 400
        assert data["code"] == "INVALID_PARAMETER"

    def test_out_of_range_id_returns_400(self, server):
        code, data = server.get("/api/history?id=99")
        assert code == 400
        assert data["code"] == "INVALID_PARAMETER"

    def test_out_of_range_limit_returns_400(self, server):
        code, data = server.get(f"/api/history?id=2&limit={HISTORY_LIMIT_MAX + 1}")
        assert code == 400
        assert data["code"] == "INVALID_PARAMETER"

    def test_zero_limit_returns_400(self, server):
        code, data = server.get("/api/history?id=2&limit=0")
        assert code == 400
        assert data["code"] == "INVALID_PARAMETER"


class TestQueueEndpoint:

    def test_returns_200(self, server):
        code, data = server.get("/api/queue")
        assert code == 200

    def test_has_scheduler_fields(self, server):
        _, data = server.get("/api/queue")
        assert "priority_queue" in data
        assert "rr_next_5" in data
        assert "total_writes" in data
        assert "sweeps_completed" in data


class TestCreateEventEndpoint:

    def test_valid_event_returns_201(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 5,
            "type": "fever_onset",
        })
        assert code == 201
        assert "event_id" in data
        assert data["animal_id"] == 5
        assert data["event_type"] == "fever_onset"

    def test_event_mutation_waits_for_simulator_state_lock(self, server):
        """HTTP injection cannot modify events while a tick owns state."""
        entered_activate_event = threading.Event()
        results: list[int] = []
        original_activate_event = activate_event

        def observe_activation(*args, **kwargs):
            entered_activate_event.set()
            return original_activate_event(*args, **kwargs)

        def post_event():
            results.append(server.post("/api/events", {
                "animal_id": 5,
                "type": "fever_onset",
            })[0])

        with patch(
            "herd_simulator.services.api_server.activate_event",
            side_effect=observe_activation,
        ):
            with server.sim.state_lock:
                worker = threading.Thread(target=post_event)
                worker.start()
                assert not entered_activate_event.wait(0.1)
            worker.join(timeout=2)

        assert not worker.is_alive()
        assert entered_activate_event.is_set()
        assert results == [201]

    def test_response_has_envelope(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 3,
            "type": "tamper",
        })
        assert data["schema_version"] == SCHEMA_VERSION
        assert "run_id" in data
        assert "sim_second" in data

    def test_invalid_type_returns_422(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 5,
            "type": "invalid_type",
        })
        assert code == 422
        assert data["code"] == "INVALID_EVENT_TYPE"
        assert "supported_types" in data["details"]

    def test_missing_type_returns_422(self, server):
        code, data = server.post("/api/events", {"animal_id": 5})
        assert code == 422

    def test_invalid_animal_id_returns_422(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 99,
            "type": "fever_onset",
        })
        assert code == 422
        assert data["code"] == "INVALID_ANIMAL_ID"

    def test_missing_animal_id_returns_422(self, server):
        code, data = server.post("/api/events", {"type": "fever_onset"})
        assert code == 422

    def test_malformed_json_returns_400(self, server):
        req = urllib.request.Request(
            server.base_url + "/api/events",
            data=b"not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.status
                data = json.loads(resp.read())
        except urllib.error.HTTPError as e:
            code = e.code
            data = json.loads(e.read())
        assert code == 400
        assert data["code"] == "INVALID_JSON"

    def test_duration_seconds_is_rejected_for_live_event(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 7,
            "type": "heat_stress",
            "duration_seconds": 600,
        })
        assert code == 422
        assert data["code"] == "DURATION_NOT_ALLOWED"

    def test_invalid_duration_returns_422(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 7,
            "type": "heat_stress",
            "duration_seconds": -1,
        })
        assert code == 422
        assert data["code"] == "DURATION_NOT_ALLOWED"

    def test_with_params(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 5,
            "type": "fever_onset",
            "params": {"peak_offset_c": 2.5},
        })
        assert code == 201

    def test_event_appears_in_state(self, server):
        code, data = server.post("/api/events", {
            "animal_id": 10,
            "type": "geofence_breach",
        })
        event_id = data["event_id"]
        _, state = server.get("/api/state")
        assert event_id in state["active_events"]


class TestDeleteEventEndpoint:

    def test_delete_existing_returns_204(self, server):
        _, create_data = server.post("/api/events", {
            "animal_id": 5,
            "type": "fever_onset",
        })
        event_id = create_data["event_id"]
        code, _ = server.delete(f"/api/events/{event_id}")
        assert code == 204

    def test_delete_nonexistent_returns_404(self, server):
        code, data = server.delete("/api/events/nonexistent-id")
        assert code == 404
        assert data["code"] == "EVENT_NOT_FOUND"

    def test_event_gone_after_delete(self, server):
        _, create_data = server.post("/api/events", {
            "animal_id": 5,
            "type": "tamper",
        })
        event_id = create_data["event_id"]
        server.delete(f"/api/events/{event_id}")
        _, state = server.get("/api/state")
        assert event_id not in state["active_events"]


# =======================================================================
# Integration tests: static file serving
# =======================================================================

class TestStaticServing:

    def test_index_html(self, server):
        req = urllib.request.Request(server.base_url + "/")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert "text/html" in resp.headers.get("Content-Type", "")

    def test_app_js(self, server):
        req = urllib.request.Request(server.base_url + "/app.js")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert "javascript" in resp.headers.get("Content-Type", "")

    def test_style_css(self, server):
        req = urllib.request.Request(server.base_url + "/style.css")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200
            assert "css" in resp.headers.get("Content-Type", "")

    def test_leaflet_js(self, server):
        req = urllib.request.Request(server.base_url + "/leaflet/leaflet.js")
        with urllib.request.urlopen(req, timeout=5) as resp:
            assert resp.status == 200

    def test_not_found(self, server):
        code, data = server.get("/nonexistent.txt")
        assert code == 404

    def test_directory_traversal_blocked(self, server):
        try:
            req = urllib.request.Request(server.base_url + "/../../../etc/passwd")
            with urllib.request.urlopen(req, timeout=5) as resp:
                code = resp.status
        except urllib.error.HTTPError as e:
            code = e.code
        assert code in (403, 404)


# =======================================================================
# JSON contract tests
# =======================================================================

class TestJsonContract:

    def test_snake_case_keys(self, server):
        _, data = server.get("/api/state")
        all_keys = set()
        _collect_keys(data, all_keys)
        for k in all_keys:
            assert "_" in k or k.islower() or k.isdigit() or k.startswith("evt-"), \
                f"Non-snake_case key: {k}"

    def test_schema_version_in_health(self, server):
        _, data = server.get("/api/health")
        assert data["schema_version"] == SCHEMA_VERSION

    def test_schema_version_in_state(self, server):
        _, data = server.get("/api/state")
        assert data["schema_version"] == SCHEMA_VERSION

    def test_schema_version_in_queue(self, server):
        _, data = server.get("/api/queue")
        assert data["schema_version"] == SCHEMA_VERSION

    def test_schema_version_in_history(self, server):
        _, data = server.get("/api/history?id=2")
        assert data["schema_version"] == SCHEMA_VERSION

    def test_run_id_in_responses(self, server):
        for path in ("/api/health", "/api/state", "/api/queue", "/api/history?id=2"):
            _, data = server.get(path)
            assert "run_id" in data, f"Missing run_id in {path}"

    def test_sim_second_in_responses(self, server):
        for path in ("/api/health", "/api/state", "/api/queue", "/api/history?id=2"):
            _, data = server.get(path)
            assert "sim_second" in data, f"Missing sim_second in {path}"


def _collect_keys(obj: Any, keys: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            keys.add(k)
            _collect_keys(v, keys)
    elif isinstance(obj, list):
        for item in obj:
            _collect_keys(item, keys)
