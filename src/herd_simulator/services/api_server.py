"""
api_server.py — Local REST API and static file server for the HUD.

Runs a stdlib-only threaded HTTP server that exposes simulation state
via JSON endpoints and serves the Leaflet.js web dashboard.

Threading model:
  - The HTTP server runs in a daemon thread (same pattern as ThingSpeakClient).
  - Simulator state is read and mutated under ``Simulator.state_lock``.
  - Mutable HUD-specific state (history buffer, latest snapshot) is protected
    by a single threading.Lock.
  - Event injection calls activate_event()/clear directly from the handler
    thread under the simulator lock.

Wiring:
  wire_api_server() MUST be called AFTER wire_logger() and wire_thingspeak().

References:
  ADR-011: Local Web Visualizer HUD Architecture (Zero-Dependency)
  AGENTS.md §7: Local REST API Endpoints
  Master PRD: "Local API and HUD"
  HerdSimulator PRD FR-34
"""

from __future__ import annotations

import collections
import dataclasses
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

from herd_simulator.config import HudConfig
from herd_simulator.engine.scenario_runner import (
    EventType,
    activate_event,
)
from herd_simulator.engine.scheduler import (
    enqueue_priority,
    get_queue_snapshot,
)
from herd_simulator.engine.simulator import (
    AnimalTelemetry,
    Simulator,
)

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_HISTORY_PER_ANIMAL = 10_000
DEFAULT_HISTORY_LIMIT = 100
HISTORY_LIMIT_MAX = 10_000

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

_CONTENT_TYPES: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".png": "image/png",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
}

_SUPPORTED_EVENT_TYPES = [et.value for et in EventType]


# -----------------------------------------------------------------------
# Shared HUD state
# -----------------------------------------------------------------------

@dataclasses.dataclass
class HudState:
    """Mutable state shared between the tick-loop callback and HTTP handlers."""

    sim: Simulator
    run_id: str
    start_time: float
    latest_telemetry: list[AnimalTelemetry] = dataclasses.field(
        default_factory=list,
    )
    history: dict[int, collections.deque[AnimalTelemetry]] = dataclasses.field(
        default_factory=dict,
    )
    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
    thingspeak_client: Any = None


def create_hud_state(
    sim: Simulator,
    run_id: str,
    thingspeak_client: Any = None,
) -> HudState:
    return HudState(
        sim=sim,
        run_id=run_id,
        start_time=time.monotonic(),
        thingspeak_client=thingspeak_client,
    )


# -----------------------------------------------------------------------
# JSON helpers
# -----------------------------------------------------------------------

def _telemetry_to_dict(t: AnimalTelemetry) -> dict[str, Any]:
    return {
        "animal_id": t.animal_id,
        "is_physical": t.is_physical,
        "sim_second": t.sim_second,
        "body_temp_c": round(t.body_temp_c, 2),
        "thi": round(t.thi, 2),
        "behaviour": t.behaviour,
        "latitude": round(t.latitude, 6),
        "longitude": round(t.longitude, 6),
        "risk_score": t.risk_score,
        "alert_band": t.alert_band,
        "geofence_status": t.geofence_status,
        "battery_pct": round(t.battery_pct, 2),
        "event_codes": t.event_codes,
        "dropped_out": t.dropped_out,
        "stale": t.stale,
    }


def _base_envelope(hs: HudState) -> dict[str, Any]:
    with hs.sim.state_lock:
        sim_second = hs.sim.clock.sim_second
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": hs.run_id,
        "sim_second": sim_second,
    }


def _error_body(
    code: str,
    message: str,
    details: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if details is not None:
        body["details"] = details
    return body


# -----------------------------------------------------------------------
# Event helpers
# -----------------------------------------------------------------------

def _clear_event_with_info(
    hs: HudState,
    event_id: str,
) -> Optional[tuple[str, int, str]]:
    """Clear an event by ID, returning (event_id, animal_id, event_type) if found."""
    state = hs.sim.event_state
    for key, active in list(state.active.items()):
        if active.event.event_id == event_id:
            aid, etype = key
            active.cleared = True
            del state.active[key]
            return (event_id, aid, etype)
    return None


# -----------------------------------------------------------------------
# Request handler
# -----------------------------------------------------------------------

class HudRequestHandler(BaseHTTPRequestHandler):
    """Dispatches /api/* to JSON handlers, everything else to static files."""

    server: HudHTTPServer  # type: ignore[assignment]

    @property
    def hs(self) -> HudState:
        return self.server.hud_state

    # -- Routing ----------------------------------------------------------

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        try:
            if path == "/api/health":
                self._handle_health()
            elif path == "/api/state":
                self._handle_state()
            elif path == "/api/history":
                self._handle_history(parse_qs(parsed.query))
            elif path == "/api/queue":
                self._handle_queue()
            else:
                self._serve_static(parsed.path)
        except Exception:
            _log.exception("Unhandled error in GET %s", self.path)
            self._send_json(
                _error_body("INTERNAL_ERROR", "An unexpected error occurred."),
                500,
            )

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path == "/api/events":
                self._handle_create_event()
            else:
                self._send_json(
                    _error_body("NOT_FOUND", f"No route for POST {path}"),
                    404,
                )
        except Exception:
            _log.exception("Unhandled error in POST %s", self.path)
            self._send_json(
                _error_body("INTERNAL_ERROR", "An unexpected error occurred."),
                500,
            )

    def do_DELETE(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        try:
            if path.startswith("/api/events/"):
                event_id = path[len("/api/events/"):]
                if event_id:
                    self._handle_delete_event(event_id)
                else:
                    self._send_json(
                        _error_body("MISSING_PARAMETER", "Event ID is required in the URL path."),
                        400,
                    )
            else:
                self._send_json(
                    _error_body("NOT_FOUND", f"No route for DELETE {path}"),
                    404,
                )
        except Exception:
            _log.exception("Unhandled error in DELETE %s", self.path)
            self._send_json(
                _error_body("INTERNAL_ERROR", "An unexpected error occurred."),
                500,
            )

    # -- API handlers -----------------------------------------------------

    def _handle_health(self) -> None:
        sim = self.hs.sim
        try:
            import resource
            mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        except (ImportError, AttributeError):
            mem_mb = 0.0

        with sim.state_lock:
            sim_mode = sim.clock.mode.value
            paused = sim.paused
            herd_size = sim.cfg.herd.n_total
            active_events = len(sim.event_state.active)
            queue_depth = len(sim.scheduler_state.priority_queue) + len(sim.scheduler_state.rr_queue)

        data = _base_envelope(self.hs)
        data.update({
            "uptime_seconds": round(time.monotonic() - self.hs.start_time, 1),
            "sim_mode": sim_mode,
            "paused": paused,
            "herd_size": herd_size,
            "active_events": active_events,
            "queue_depth": queue_depth,
            "memory_rss_mb": round(mem_mb, 1),
        })
        self._send_json(data)

    def _handle_state(self) -> None:
        sim = self.hs.sim
        with sim.state_lock:
            sim_second = sim.clock.sim_second
            active_events: dict[str, Any] = {}
            for (aid, etype), ae in sim.event_state.active.items():
                eid = ae.event.event_id or ""
                active_events[eid] = {
                    "animal_id": aid,
                    "event_type": etype,
                    "activated_at": ae.activated_at,
                    "elapsed_s": sim_second - ae.activated_at,
                }
            ambient_temp_c = round(sim._ambient_temp_c, 2)
            humidity_pct = round(sim._humidity_pct, 2)
            thi = round(sim._thi, 2)
            centroid = [round(sim.centroid[0], 6), round(sim.centroid[1], 6)]
            pasture_polygon = [
                [round(lat, 6), round(lon, 6)]
                for lat, lon in sim.cfg.pasture_polygon
            ]
        with self.hs.lock:
            animals = [_telemetry_to_dict(t) for t in self.hs.latest_telemetry]

        data = _base_envelope(self.hs)
        data.update({
            "ambient_temp_c": ambient_temp_c,
            "humidity_pct": humidity_pct,
            "thi": thi,
            "centroid": centroid,
            "pasture_polygon": pasture_polygon,
            "animals": animals,
            "active_events": active_events,
        })
        self._send_json(data)

    def _handle_history(self, qs: dict[str, list[str]]) -> None:
        raw_id = qs.get("id", [None])[0]  # type: ignore[list-item]
        if raw_id is None:
            self._send_json(
                _error_body("MISSING_PARAMETER", "Query parameter 'id' is required."),
                400,
            )
            return

        try:
            animal_id = int(raw_id)
        except (ValueError, TypeError):
            self._send_json(
                _error_body(
                    "INVALID_PARAMETER",
                    f"'id' must be an integer, got '{raw_id}'.",
                ),
                400,
            )
            return

        if animal_id < 1 or animal_id > self.hs.sim.cfg.herd.n_total:
            self._send_json(
                _error_body(
                    "INVALID_PARAMETER",
                    f"'id' must be between 1 and {self.hs.sim.cfg.herd.n_total}.",
                ),
                400,
            )
            return

        raw_limit = qs.get("limit", [str(DEFAULT_HISTORY_LIMIT)])[0]
        try:
            limit = int(raw_limit)
        except (ValueError, TypeError):
            self._send_json(
                _error_body(
                    "INVALID_PARAMETER",
                    f"'limit' must be an integer, got '{raw_limit}'.",
                ),
                400,
            )
            return

        if limit < 1 or limit > HISTORY_LIMIT_MAX:
            self._send_json(
                _error_body(
                    "INVALID_PARAMETER",
                    f"'limit' must be between 1 and {HISTORY_LIMIT_MAX}.",
                ),
                400,
            )
            return

        with self.hs.lock:
            dq = self.hs.history.get(animal_id)
            if dq is None:
                records: list[dict[str, Any]] = []
            else:
                tail = list(dq)[-limit:]
                records = [_telemetry_to_dict(t) for t in tail]

        data = _base_envelope(self.hs)
        data.update({
            "animal_id": animal_id,
            "count": len(records),
            "records": records,
        })
        self._send_json(data)

    def _handle_queue(self) -> None:
        with self.hs.sim.state_lock:
            snapshot = get_queue_snapshot(self.hs.sim.scheduler_state)

        data = _base_envelope(self.hs)
        data.update(snapshot)

        if self.hs.thingspeak_client is not None:
            try:
                data["thingspeak_quota"] = self.hs.thingspeak_client.get_quota_snapshot()
            except Exception:
                pass

        self._send_json(data)

    def _handle_create_event(self) -> None:
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len == 0:
            self._send_json(
                _error_body("INVALID_JSON", "Request body is empty."),
                400,
            )
            return

        try:
            body = json.loads(self.rfile.read(content_len).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            self._send_json(
                _error_body("INVALID_JSON", f"Malformed JSON: {exc}"),
                400,
            )
            return

        if not isinstance(body, dict):
            self._send_json(
                _error_body("INVALID_JSON", "Request body must be a JSON object."),
                400,
            )
            return

        raw_aid = body.get("animal_id")
        if raw_aid is None or not isinstance(raw_aid, int):
            self._send_json(
                _error_body("INVALID_PARAMETER", "'animal_id' is required and must be an integer."),
                422,
            )
            return

        n_total = self.hs.sim.cfg.herd.n_total
        if raw_aid < 1 or raw_aid > n_total:
            self._send_json(
                _error_body(
                    "INVALID_ANIMAL_ID",
                    f"'animal_id' must be between 1 and {n_total}.",
                    {"valid_range": [1, n_total]},
                ),
                422,
            )
            return

        raw_type = body.get("type")
        if raw_type is None or not isinstance(raw_type, str):
            self._send_json(
                _error_body(
                    "INVALID_EVENT_TYPE",
                    "'type' is required and must be a string.",
                    {"supported_types": _SUPPORTED_EVENT_TYPES},
                ),
                422,
            )
            return

        try:
            evt_type = EventType(raw_type)
        except ValueError:
            self._send_json(
                _error_body(
                    "INVALID_EVENT_TYPE",
                    f"Event type '{raw_type}' is not supported.",
                    {"supported_types": _SUPPORTED_EVENT_TYPES},
                ),
                422,
            )
            return

        params = body.get("params")
        if params is not None and not isinstance(params, dict):
            self._send_json(
                _error_body("INVALID_PARAMETER", "'params' must be an object if provided."),
                422,
            )
            return

        duration = body.get("duration_seconds")
        if duration is not None:
            self._send_json(
                _error_body(
                    "DURATION_NOT_ALLOWED",
                    "Live API events run until explicitly cleared; omit 'duration_seconds'.",
                ),
                422,
            )
            return

        sim = self.hs.sim
        with sim.state_lock:
            event_id = activate_event(
                sim.event_state,
                raw_aid,
                evt_type,
                sim.clock.sim_second,
                params=params,
            )
            if raw_aid >= 2:
                enqueue_priority(sim.scheduler_state, raw_aid)

        if sim.on_event_activated:
            sim.on_event_activated(event_id, raw_aid, evt_type.value)

        resp = _base_envelope(self.hs)
        resp.update({
            "event_id": event_id,
            "animal_id": raw_aid,
            "event_type": evt_type.value,
        })
        self._send_json(resp, 201)

    def _handle_delete_event(self, event_id: str) -> None:
        with self.hs.sim.state_lock:
            result = _clear_event_with_info(self.hs, event_id)

        if result is None:
            self._send_json(
                _error_body("EVENT_NOT_FOUND", f"No active event with id '{event_id}'."),
                404,
            )
            return

        _, aid, etype = result
        if self.hs.sim.on_event_cleared:
            self.hs.sim.on_event_cleared(event_id, aid, etype)

        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- Static file serving ----------------------------------------------

    def _serve_static(self, url_path: str) -> None:
        if url_path in ("", "/"):
            url_path = "/index.html"

        rel = url_path.lstrip("/")
        file_path = (WEB_DIR / rel).resolve()

        if not str(file_path).startswith(str(WEB_DIR)):
            self._send_json(
                _error_body("FORBIDDEN", "Access denied."),
                403,
            )
            return

        if not file_path.is_file():
            self._send_json(
                _error_body("NOT_FOUND", f"File not found: {rel}"),
                404,
            )
            return

        suffix = file_path.suffix.lower()
        content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        if suffix == ".html":
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    # -- Response helpers -------------------------------------------------

    def _send_json(self, data: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(data, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        _log.debug("HUD HTTP: %s", format % args)


# -----------------------------------------------------------------------
# Server
# -----------------------------------------------------------------------

class HudHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer subclass that carries HudState."""

    hud_state: HudState


class HudServer:
    """Lifecycle manager for the HUD HTTP server thread."""

    def __init__(self, cfg: HudConfig, hud_state: HudState) -> None:
        self._cfg = cfg
        self._hud_state = hud_state
        self._server: Optional[HudHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._server = HudHTTPServer(
            (self._cfg.host, self._cfg.port),
            HudRequestHandler,
        )
        self._server.hud_state = self._hud_state
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="hud-server",
        )
        self._thread.start()
        _log.info(
            "HUD server started on http://%s:%d",
            self._cfg.host,
            self._cfg.port,
        )

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server = None
            _log.info("HUD server stopped")

    @property
    def port(self) -> Optional[int]:
        """Actual bound port (useful when configured port is 0)."""
        if self._server:
            return self._server.server_address[1]
        return None


# -----------------------------------------------------------------------
# Wiring
# -----------------------------------------------------------------------

def wire_api_server(sim: Simulator, hud_state: HudState) -> None:
    """Register HUD state-update callbacks on the simulator.

    MUST be called AFTER wire_logger() and wire_thingspeak() to chain
    on_tick_complete correctly.
    """
    existing_on_tick_complete = sim.on_tick_complete

    def _on_tick_complete(
        telemetry: list[AnimalTelemetry],
        ss: int,
    ) -> None:
        if existing_on_tick_complete:
            existing_on_tick_complete(telemetry, ss)
        try:
            with hud_state.lock:
                hud_state.latest_telemetry = telemetry
                for t in telemetry:
                    if t.animal_id not in hud_state.history:
                        hud_state.history[t.animal_id] = collections.deque(
                            maxlen=MAX_HISTORY_PER_ANIMAL,
                        )
                    hud_state.history[t.animal_id].append(t)
        except Exception:
            _log.exception("HUD state update failed (simulation continues)")

    sim.on_tick_complete = _on_tick_complete
