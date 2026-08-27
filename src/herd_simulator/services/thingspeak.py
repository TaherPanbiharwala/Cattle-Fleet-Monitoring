"""
thingspeak.py — ThingSpeak Channel 2 POST writer and Channel 1 GET sniffer.

Background-threaded HTTP integration so the 1-second tick loop never blocks
on network I/O.  All writes go through a bounded queue; the writer thread
drains it with exponential backoff on failures.  Quota enforcement disables
writes before the configurable annual ceiling is reached.

References:
  ADR-004: ThingSpeak free-tier channel allocation & rate limits
  ADR-010: Collar-1 sniffing & synchronization
  Master PRD §"Scheduler", §"P1: Digital-Twin Simulator"
  AGENTS.md §3 golden rules 2 & 5, §8
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

from herd_simulator.config import ThingSpeakConfig
from herd_simulator.engine.simulator import AnimalTelemetry, SimMode, Simulator
from herd_simulator.utils.geo import Coord

_log = logging.getLogger(__name__)

THINGSPEAK_UPDATE_URL = "https://api.thingspeak.com/update"
THINGSPEAK_FEED_URL = "https://api.thingspeak.com/channels/{channel_id}/feeds/last.json"
MAX_RETRY_ATTEMPTS = 4
WRITE_QUEUE_MAXSIZE = 100
HTTP_TIMEOUT_S = 10


# -----------------------------------------------------------------------
# Data types
# -----------------------------------------------------------------------

@dataclass
class QuotaState:
    annual_limit: int
    warning_pct: int
    annual_count: int = 0
    daily_count: int = 0
    day_start_monotonic: float = field(default_factory=time.monotonic)
    warned: bool = False
    disabled: bool = False


@dataclass
class BackoffState:
    consecutive_failures: int = 0


@dataclass(frozen=True)
class SniffedFix:
    latitude: float
    longitude: float
    created_at_iso: str
    fetched_at_monotonic: float


@dataclass(frozen=True)
class WriteRequest:
    animal_id: int
    sim_second: int
    fields: dict[str, str]


# -----------------------------------------------------------------------
# Pure functions
# -----------------------------------------------------------------------

def format_status_field(animal_id: int, event_codes: list[str]) -> str:
    parts = [f"id={animal_id:02d}"]
    if event_codes:
        parts.append(f"evt={'|'.join(event_codes)}")
    parts.append("src=RULE")
    return ";".join(parts)


def telemetry_to_fields(t: AnimalTelemetry) -> dict[str, str]:
    return {
        "field1": f"{t.body_temp_c:.2f}",
        "field2": f"{t.thi:.2f}",
        "field3": str(t.behaviour),
        "field4": f"{t.latitude:.6f}",
        "field5": f"{t.longitude:.6f}",
        "field6": str(t.risk_score),
        "field7": str(t.geofence_status),
        "field8": str(int(t.battery_pct)),
        "status": format_status_field(t.animal_id, t.event_codes),
    }


def format_post_body(fields: dict[str, str], write_api_key: str) -> bytes:
    payload = {"api_key": write_api_key, **fields}
    return urllib.parse.urlencode(payload).encode("ascii")


def parse_channel1_response(raw_json: bytes) -> Optional[tuple[float, float, str]]:
    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, ValueError):
        return None
    try:
        lat = float(data["field4"])
        lon = float(data["field5"])
        created = str(data["created_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return (lat, lon, created)


def check_quota(state: QuotaState) -> tuple[bool, Optional[str]]:
    if time.monotonic() - state.day_start_monotonic >= 86400:
        state.daily_count = 0
        state.day_start_monotonic = time.monotonic()

    if state.disabled:
        return (False, f"quota disabled: {state.annual_count}/{state.annual_limit} annual writes")

    if state.annual_count >= state.annual_limit:
        state.disabled = True
        return (False, f"quota ceiling reached: {state.annual_count}/{state.annual_limit}")

    warning_threshold = int(state.annual_limit * state.warning_pct / 100)
    if state.annual_count >= warning_threshold and not state.warned:
        state.warned = True
        return (True, f"quota warning: {state.annual_count}/{state.annual_limit} ({state.warning_pct}% threshold)")

    return (True, None)


def record_quota_write(state: QuotaState) -> None:
    state.annual_count += 1
    state.daily_count += 1


def next_backoff_delay(state: BackoffState, base_s: int, max_s: int) -> float:
    delay = min(base_s * (2 ** state.consecutive_failures), max_s)
    state.consecutive_failures += 1
    return float(delay)


def reset_backoff(state: BackoffState) -> None:
    state.consecutive_failures = 0


# -----------------------------------------------------------------------
# HTTP primitives
# -----------------------------------------------------------------------

def _http_post(url: str, body: bytes, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, data=body, method="POST",
        headers={"User-Agent": "HerdSimulator/1.0"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout_s)
    return (resp.status, resp.read())


def _http_get(url: str, timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(
        url, method="GET",
        headers={"User-Agent": "HerdSimulator/1.0"},
    )
    resp = urllib.request.urlopen(req, timeout=timeout_s)
    return (resp.status, resp.read())


# -----------------------------------------------------------------------
# ThingSpeakClient
# -----------------------------------------------------------------------

class ThingSpeakClient:
    def __init__(
        self,
        cfg: ThingSpeakConfig,
        credentials: dict[str, str],
        mode: SimMode,
    ) -> None:
        self._cfg = cfg
        self._mode = mode
        self._write_api_key = credentials.get("THINGSPEAK_WRITE_API_KEY", "")
        self._read_api_key = credentials.get("THINGSPEAK_READ_API_KEY", "")

        self._write_queue: queue.Queue[WriteRequest] = queue.Queue(maxsize=WRITE_QUEUE_MAXSIZE)
        self._stop_event = threading.Event()

        self._quota = QuotaState(
            annual_limit=cfg.annual_write_limit,
            warning_pct=cfg.quota_warning_pct,
        )
        self._backoff = BackoffState()

        self._latest_fix: Optional[SniffedFix] = None
        self._fix_lock = threading.Lock()

        self._writer_thread: Optional[threading.Thread] = None
        self._sniffer_thread: Optional[threading.Thread] = None

        self.on_write_result: Optional[Callable[..., None]] = None

    def start(self) -> None:
        if self._mode != SimMode.LIVE:
            _log.info("ThingSpeak client not started (mode=%s)", self._mode.value)
            return

        if self._write_api_key:
            self._writer_thread = threading.Thread(
                target=self._writer_loop, name="ts-writer", daemon=True,
            )
            self._writer_thread.start()
            _log.info("ThingSpeak writer thread started (key=set)")
        else:
            _log.warning("ThingSpeak writer not started (THINGSPEAK_WRITE_API_KEY missing)")

        if self._read_api_key and self._cfg.channel_1_id:
            self._sniffer_thread = threading.Thread(
                target=self._sniffer_loop, name="ts-sniffer", daemon=True,
            )
            self._sniffer_thread.start()
            _log.info("ThingSpeak sniffer thread started (channel_1_id=%s)", self._cfg.channel_1_id)
        else:
            _log.info("ThingSpeak sniffer not started (key=%s, channel_1_id=%s)",
                       "set" if self._read_api_key else "missing",
                       self._cfg.channel_1_id or "empty")

    def stop(self) -> None:
        self._stop_event.set()

    def enqueue_write(self, t: AnimalTelemetry) -> None:
        if self._mode != SimMode.LIVE:
            return
        fields = telemetry_to_fields(t)
        req = WriteRequest(animal_id=t.animal_id, sim_second=t.sim_second, fields=fields)
        try:
            self._write_queue.put_nowait(req)
        except queue.Full:
            _log.warning("Write queue full — dropping write for animal %d at ss=%d", t.animal_id, t.sim_second)
            self._notify_write_result(t.animal_id, t.sim_second, "dropped", attempts=0)

    def get_collar1_fix(self) -> Optional[SniffedFix]:
        with self._fix_lock:
            return self._latest_fix

    def get_quota_snapshot(self) -> dict:
        return {
            "annual_count": self._quota.annual_count,
            "annual_limit": self._quota.annual_limit,
            "daily_count": self._quota.daily_count,
            "warning_pct": self._quota.warning_pct,
            "warned": self._quota.warned,
            "disabled": self._quota.disabled,
        }

    # --- Writer thread ---

    def _writer_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                req = self._write_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            self._execute_write(req)

    def _execute_write(self, req: WriteRequest) -> None:
        allowed, msg = check_quota(self._quota)
        if msg:
            _log.warning(msg)
        if not allowed:
            self._notify_write_result(req.animal_id, req.sim_second, "quota_disabled", attempts=0)
            return

        body = format_post_body(req.fields, self._write_api_key)
        for attempt in range(MAX_RETRY_ATTEMPTS):
            if self._stop_event.is_set():
                return
            try:
                status, resp_body = _http_post(THINGSPEAK_UPDATE_URL, body, HTTP_TIMEOUT_S)
            except (urllib.error.URLError, OSError, TimeoutError) as exc:
                delay = next_backoff_delay(self._backoff, self._cfg.backoff_base_s, self._cfg.backoff_max_s)
                _log.warning("POST failed for animal %d (attempt %d/%d): %s — backoff %.1fs",
                             req.animal_id, attempt + 1, MAX_RETRY_ATTEMPTS, exc, delay)
                self._stop_event.wait(delay)
                continue

            entry_id = resp_body.strip()
            if status == 200 and entry_id != b"0":
                record_quota_write(self._quota)
                reset_backoff(self._backoff)
                _log.debug("POST success for animal %d at ss=%d (entry=%s)",
                           req.animal_id, req.sim_second, entry_id.decode("ascii", errors="replace"))
                self._notify_write_result(req.animal_id, req.sim_second, "success",
                                          status_code=status, attempts=attempt + 1)
                return

            delay = next_backoff_delay(self._backoff, self._cfg.backoff_base_s, self._cfg.backoff_max_s)
            _log.warning("POST rejected for animal %d (status=%d, body=%s, attempt %d/%d) — backoff %.1fs",
                         req.animal_id, status, entry_id[:20], attempt + 1, MAX_RETRY_ATTEMPTS, delay)
            self._stop_event.wait(delay)

        reset_backoff(self._backoff)
        _log.error("POST exhausted retries for animal %d at ss=%d", req.animal_id, req.sim_second)
        self._notify_write_result(req.animal_id, req.sim_second, "retry_exhausted",
                                  attempts=MAX_RETRY_ATTEMPTS)

    # --- Sniffer thread ---

    def _sniffer_loop(self) -> None:
        while not self._stop_event.is_set():
            fix = self._fetch_channel1()
            if fix is not None:
                with self._fix_lock:
                    self._latest_fix = fix
            self._stop_event.wait(self._cfg.channel_1_sniff_interval_s)

    def _fetch_channel1(self) -> Optional[SniffedFix]:
        url = THINGSPEAK_FEED_URL.format(channel_id=self._cfg.channel_1_id)
        url += f"?api_key={self._read_api_key}"
        try:
            status, body = _http_get(url, HTTP_TIMEOUT_S)
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            _log.debug("Channel 1 sniff failed: %s", exc)
            return None
        if status != 200:
            _log.debug("Channel 1 sniff returned status %d", status)
            return None
        parsed = parse_channel1_response(body)
        if parsed is None:
            return None
        lat, lon, created = parsed
        return SniffedFix(
            latitude=lat,
            longitude=lon,
            created_at_iso=created,
            fetched_at_monotonic=time.monotonic(),
        )

    # --- Internal helpers ---

    def _notify_write_result(
        self,
        animal_id: int,
        sim_second: int,
        outcome: str,
        status_code: Optional[int] = None,
        attempts: int = 0,
    ) -> None:
        if self.on_write_result is not None:
            self.on_write_result(animal_id, sim_second, outcome, status_code, attempts)


# -----------------------------------------------------------------------
# Wiring
# -----------------------------------------------------------------------

def wire_thingspeak(sim: Simulator, client: ThingSpeakClient) -> None:
    """Register ThingSpeak callbacks on the simulator.

    MUST be called AFTER wire_logger() to chain on_transmit correctly.
    """
    existing_on_transmit = sim.on_transmit

    def _on_transmit(t: AnimalTelemetry) -> None:
        if existing_on_transmit:
            existing_on_transmit(t)
        client.enqueue_write(t)

    sim.on_transmit = _on_transmit
