"""Tests for ThingSpeak client, quota enforcement, and Channel 1 sniffer."""

from __future__ import annotations

import json
import queue
import threading
import time
import urllib.error
from io import BytesIO
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from herd_simulator.config import ThingSpeakConfig
from herd_simulator.engine.simulator import AnimalTelemetry, SimMode
from herd_simulator.services.thingspeak import (
    THINGSPEAK_UPDATE_URL,
    BackoffState,
    QuotaState,
    SniffedFix,
    ThingSpeakClient,
    WriteRequest,
    check_quota,
    format_post_body,
    format_status_field,
    next_backoff_delay,
    parse_channel1_response,
    record_quota_write,
    reset_backoff,
    telemetry_to_fields,
    wire_thingspeak,
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


def _make_ts_config(**overrides) -> ThingSpeakConfig:
    defaults = dict(
        channel_2_id="12345",
        write_cadence_s=30,
        breach_cadence_s=15,
        min_interval_s=15,
        channel_1_id="67890",
        channel_1_sniff_interval_s=60,
        channel_1_stale_threshold_s=120,
        backoff_base_s=2,
        backoff_max_s=16,
        annual_write_limit=3_000_000,
        quota_warning_pct=90,
    )
    defaults.update(overrides)
    return ThingSpeakConfig(**defaults)


def _make_credentials(write_key: str = "TESTKEY", read_key: str = "READKEY") -> dict[str, str]:
    return {
        "THINGSPEAK_WRITE_API_KEY": write_key,
        "THINGSPEAK_READ_API_KEY": read_key,
        "THINGSPEAK_CONFIG_READ_API_KEY": "",
    }


def _mock_response(body: bytes, status: int = 200):
    resp = MagicMock()
    resp.read.return_value = body
    resp.status = status
    resp.__enter__ = MagicMock(return_value=resp)
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# =======================================================================
# Pure function tests
# =======================================================================


class TestFormatStatusField:
    def test_no_events(self):
        assert format_status_field(7, []) == "id=07;src=RULE"

    def test_single_event(self):
        assert format_status_field(7, ["FEVER"]) == "id=07;evt=FEVER;src=RULE"

    def test_multiple_events(self):
        assert format_status_field(14, ["FEVER", "BREACH"]) == "id=14;evt=FEVER|BREACH;src=RULE"

    def test_single_digit_padded(self):
        assert format_status_field(3, []) == "id=03;src=RULE"

    def test_physical_collar_id(self):
        assert format_status_field(1, []) == "id=01;src=RULE"

    def test_two_digit_id(self):
        assert format_status_field(14, []) == "id=14;src=RULE"

    def test_twenty(self):
        assert format_status_field(20, []) == "id=20;src=RULE"


class TestTelemetryToFields:
    def test_all_fields_present(self):
        t = _make_telemetry()
        fields = telemetry_to_fields(t)
        for i in range(1, 9):
            assert f"field{i}" in fields
        assert "status" in fields

    def test_field_values_are_strings(self):
        fields = telemetry_to_fields(_make_telemetry())
        for v in fields.values():
            assert isinstance(v, str)

    def test_behaviour_is_integer_code(self):
        fields = telemetry_to_fields(_make_telemetry(behaviour=0))
        assert fields["field3"] == "0"

    def test_battery_is_integer(self):
        fields = telemetry_to_fields(_make_telemetry(battery_pct=99.7))
        assert fields["field8"] == "99"

    def test_field_mapping_matches_contract(self):
        t = _make_telemetry(
            body_temp_c=38.60, thi=72.50, behaviour=1,
            latitude=12.971600, longitude=79.158900,
            risk_score=25, geofence_status=0, battery_pct=95.0,
        )
        fields = telemetry_to_fields(t)
        assert fields["field1"] == "38.60"
        assert fields["field2"] == "72.50"
        assert fields["field3"] == "1"
        assert fields["field4"] == "12.971600"
        assert fields["field5"] == "79.158900"
        assert fields["field6"] == "25"
        assert fields["field7"] == "0"
        assert fields["field8"] == "95"

    def test_status_with_events(self):
        t = _make_telemetry(animal_id=7, event_codes=["FEVER"])
        fields = telemetry_to_fields(t)
        assert fields["status"] == "id=07;evt=FEVER;src=RULE"


class TestFormatPostBody:
    def test_url_encoding(self):
        fields = {"field1": "38.6", "field2": "72.5"}
        body = format_post_body(fields, "KEY123")
        decoded = body.decode("ascii")
        assert "api_key=KEY123" in decoded
        assert "field1=38.6" in decoded
        assert "field2=72.5" in decoded

    def test_api_key_included(self):
        body = format_post_body({"field1": "1.0"}, "SECRETKEY")
        assert b"api_key=SECRETKEY" in body

    def test_all_fields_in_body(self):
        fields = telemetry_to_fields(_make_telemetry())
        body = format_post_body(fields, "K").decode("ascii")
        for key in fields:
            assert key in body


class TestParseChannel1Response:
    def test_valid_response(self):
        data = json.dumps({
            "field4": "12.9716",
            "field5": "79.1589",
            "created_at": "2026-08-27T10:00:00Z",
        }).encode()
        result = parse_channel1_response(data)
        assert result is not None
        lat, lon, created = result
        assert abs(lat - 12.9716) < 1e-6
        assert abs(lon - 79.1589) < 1e-6
        assert created == "2026-08-27T10:00:00Z"

    def test_missing_field4(self):
        data = json.dumps({"field5": "79.0", "created_at": "t"}).encode()
        assert parse_channel1_response(data) is None

    def test_missing_field5(self):
        data = json.dumps({"field4": "12.0", "created_at": "t"}).encode()
        assert parse_channel1_response(data) is None

    def test_empty_response(self):
        assert parse_channel1_response(b"") is None

    def test_malformed_json(self):
        assert parse_channel1_response(b"not json") is None

    def test_null_field_values(self):
        data = json.dumps({"field4": None, "field5": "79.0", "created_at": "t"}).encode()
        assert parse_channel1_response(data) is None

    def test_non_numeric_field(self):
        data = json.dumps({"field4": "abc", "field5": "79.0", "created_at": "t"}).encode()
        assert parse_channel1_response(data) is None


# -----------------------------------------------------------------------
# Quota tests
# -----------------------------------------------------------------------


class TestQuota:
    def test_under_limit_allowed(self):
        q = QuotaState(annual_limit=100, warning_pct=90, annual_count=50)
        allowed, msg = check_quota(q)
        assert allowed is True
        assert msg is None

    def test_at_warning_threshold(self):
        q = QuotaState(annual_limit=100, warning_pct=90, annual_count=90)
        allowed, msg = check_quota(q)
        assert allowed is True
        assert msg is not None
        assert "warning" in msg

    def test_warning_fires_once(self):
        q = QuotaState(annual_limit=100, warning_pct=90, annual_count=90)
        _, msg1 = check_quota(q)
        assert msg1 is not None
        _, msg2 = check_quota(q)
        assert msg2 is None

    def test_at_limit_disabled(self):
        q = QuotaState(annual_limit=100, warning_pct=90, annual_count=100)
        allowed, msg = check_quota(q)
        assert allowed is False
        assert "ceiling" in msg or "disabled" in msg

    def test_stays_disabled_after_ceiling(self):
        q = QuotaState(annual_limit=100, warning_pct=90, annual_count=100)
        check_quota(q)
        allowed, _ = check_quota(q)
        assert allowed is False

    def test_daily_counter_resets(self):
        q = QuotaState(annual_limit=1_000_000, warning_pct=90,
                       annual_count=50, daily_count=50,
                       day_start_monotonic=time.monotonic() - 86401)
        check_quota(q)
        assert q.daily_count == 0

    def test_record_write_increments(self):
        q = QuotaState(annual_limit=100, warning_pct=90)
        record_quota_write(q)
        assert q.annual_count == 1
        assert q.daily_count == 1
        record_quota_write(q)
        assert q.annual_count == 2
        assert q.daily_count == 2


# -----------------------------------------------------------------------
# Backoff tests
# -----------------------------------------------------------------------


class TestBackoff:
    def test_exponential_growth(self):
        b = BackoffState()
        delays = [next_backoff_delay(b, 2, 16) for _ in range(4)]
        assert delays == [2.0, 4.0, 8.0, 16.0]

    def test_capped_at_max(self):
        b = BackoffState()
        for _ in range(10):
            d = next_backoff_delay(b, 2, 16)
        assert d == 16.0

    def test_reset_clears_state(self):
        b = BackoffState(consecutive_failures=5)
        reset_backoff(b)
        assert b.consecutive_failures == 0
        assert next_backoff_delay(b, 2, 16) == 2.0

    def test_first_delay_is_base(self):
        b = BackoffState()
        assert next_backoff_delay(b, 3, 30) == 3.0


# =======================================================================
# HTTP mock tests
# =======================================================================


class TestHttpPost:
    @patch("herd_simulator.services.thingspeak.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(b"12345", 200)
        from herd_simulator.services.thingspeak import _http_post
        status, body = _http_post("http://example.com", b"data", 10)
        assert status == 200
        assert body == b"12345"

    @patch("herd_simulator.services.thingspeak.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        from herd_simulator.services.thingspeak import _http_post
        with pytest.raises(TimeoutError):
            _http_post("http://example.com", b"data", 10)

    @patch("herd_simulator.services.thingspeak.urllib.request.urlopen")
    def test_network_error(self, mock_urlopen):
        mock_urlopen.side_effect = urllib.error.URLError("connection refused")
        from herd_simulator.services.thingspeak import _http_post
        with pytest.raises(urllib.error.URLError):
            _http_post("http://example.com", b"data", 10)


class TestHttpGet:
    @patch("herd_simulator.services.thingspeak.urllib.request.urlopen")
    def test_success(self, mock_urlopen):
        mock_urlopen.return_value = _mock_response(b'{"field4":"12.0"}', 200)
        from herd_simulator.services.thingspeak import _http_get
        status, body = _http_get("http://example.com", 10)
        assert status == 200

    @patch("herd_simulator.services.thingspeak.urllib.request.urlopen")
    def test_timeout(self, mock_urlopen):
        mock_urlopen.side_effect = TimeoutError("timed out")
        from herd_simulator.services.thingspeak import _http_get
        with pytest.raises(TimeoutError):
            _http_get("http://example.com", 10)


# =======================================================================
# ThingSpeakClient integration tests
# =======================================================================


class TestClientModeGating:
    def test_dry_run_skips_enqueue(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.DRY_RUN)
        client.enqueue_write(_make_telemetry())
        assert client._write_queue.empty()

    def test_offline_skips_enqueue(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.OFFLINE)
        client.enqueue_write(_make_telemetry())
        assert client._write_queue.empty()

    def test_live_enqueues(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.enqueue_write(_make_telemetry())
        assert not client._write_queue.empty()

    def test_start_noop_in_dry_run(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.DRY_RUN)
        client.start()
        assert client._writer_thread is None
        assert client._sniffer_thread is None


class TestClientStartStop:
    def test_start_no_write_key(self):
        cfg = _make_ts_config()
        creds = _make_credentials(write_key="")
        client = ThingSpeakClient(cfg, creds, SimMode.LIVE)
        client.start()
        assert client._writer_thread is None
        client.stop()

    def test_start_no_read_key_skips_sniffer(self):
        cfg = _make_ts_config()
        creds = _make_credentials(read_key="")
        client = ThingSpeakClient(cfg, creds, SimMode.LIVE)
        client.start()
        assert client._sniffer_thread is None
        client.stop()

    def test_start_no_channel1_id_skips_sniffer(self):
        cfg = _make_ts_config(channel_1_id="")
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()
        assert client._sniffer_thread is None
        client.stop()

    def test_quota_snapshot(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        snap = client.get_quota_snapshot()
        assert snap["annual_limit"] == 3_000_000
        assert snap["annual_count"] == 0
        assert snap["disabled"] is False


class TestWriterThread:
    @patch("herd_simulator.services.thingspeak._http_post")
    def test_enqueue_and_post(self, mock_post):
        mock_post.return_value = (200, b"99999")
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry(animal_id=5))
        time.sleep(0.5)
        client.stop()

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == THINGSPEAK_UPDATE_URL
        body = call_args[0][1].decode("ascii")
        assert "api_key=TESTKEY" in body
        assert "field1=" in body

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_success_increments_quota(self, mock_post):
        mock_post.return_value = (200, b"99999")
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry())
        time.sleep(0.5)
        client.stop()

        assert client._quota.annual_count == 1
        assert client._quota.daily_count == 1

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_retry_on_failure_then_succeed(self, mock_post):
        mock_post.side_effect = [
            urllib.error.URLError("fail"),
            (200, b"99999"),
        ]
        cfg = _make_ts_config(backoff_base_s=1, backoff_max_s=1)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry())
        time.sleep(3.0)
        client.stop()

        assert mock_post.call_count == 2
        assert client._quota.annual_count == 1

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_max_retries_exhausted(self, mock_post):
        mock_post.side_effect = urllib.error.URLError("always fail")
        cfg = _make_ts_config(backoff_base_s=1, backoff_max_s=1)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry())
        time.sleep(6.0)
        client.stop()

        assert mock_post.call_count == 4
        assert client._quota.annual_count == 0

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_thingspeak_rate_limit_retries(self, mock_post):
        mock_post.side_effect = [
            (200, b"0"),
            (200, b"12345"),
        ]
        cfg = _make_ts_config(backoff_base_s=1, backoff_max_s=1)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry())
        time.sleep(3.0)
        client.stop()

        assert mock_post.call_count == 2
        assert client._quota.annual_count == 1

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_quota_blocks_write(self, mock_post):
        cfg = _make_ts_config(annual_write_limit=0)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()

        client.enqueue_write(_make_telemetry())
        time.sleep(0.5)
        client.stop()

        mock_post.assert_not_called()

    @patch("herd_simulator.services.thingspeak._http_post")
    def test_write_result_callback(self, mock_post):
        mock_post.return_value = (200, b"99999")
        results = []
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.on_write_result = lambda *args: results.append(args)
        client.start()

        client.enqueue_write(_make_telemetry(animal_id=7, sim_second=100))
        time.sleep(0.5)
        client.stop()

        assert len(results) == 1
        assert results[0][0] == 7
        assert results[0][1] == 100
        assert results[0][2] == "success"

    def test_queue_full_drops(self):
        cfg = _make_ts_config()
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        for i in range(105):
            client.enqueue_write(_make_telemetry(sim_second=i))
        assert client._write_queue.qsize() == 100


class TestSnifferThread:
    @patch("herd_simulator.services.thingspeak._http_get")
    def test_sniffer_fetches_and_updates(self, mock_get):
        resp = json.dumps({
            "field4": "12.9716", "field5": "79.1589",
            "created_at": "2026-08-27T10:00:00Z",
        }).encode()
        mock_get.return_value = (200, resp)

        cfg = _make_ts_config(channel_1_sniff_interval_s=60)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()
        time.sleep(0.5)
        client.stop()

        fix = client.get_collar1_fix()
        assert fix is not None
        assert abs(fix.latitude - 12.9716) < 1e-6
        assert abs(fix.longitude - 79.1589) < 1e-6

    @patch("herd_simulator.services.thingspeak._http_get")
    def test_sniffer_survives_error(self, mock_get):
        mock_get.side_effect = urllib.error.URLError("network down")
        cfg = _make_ts_config(channel_1_sniff_interval_s=60)
        client = ThingSpeakClient(cfg, _make_credentials(), SimMode.LIVE)
        client.start()
        time.sleep(0.5)
        client.stop()

        assert client.get_collar1_fix() is None


# =======================================================================
# Wiring tests
# =======================================================================


class TestWireThingspeak:
    def test_chains_on_transmit(self):
        calls = {"logger": 0, "thingspeak": 0}

        class FakeSimulator:
            on_transmit = None

        class FakeClient:
            def enqueue_write(self, t):
                calls["thingspeak"] += 1

        sim = FakeSimulator()
        sim.on_transmit = lambda t: calls.__setitem__("logger", calls["logger"] + 1)

        client = FakeClient()
        existing = sim.on_transmit

        def _on_transmit(t):
            if existing:
                existing(t)
            client.enqueue_write(t)

        sim.on_transmit = _on_transmit

        sim.on_transmit(_make_telemetry())
        assert calls["logger"] == 1
        assert calls["thingspeak"] == 1

    def test_no_existing_callback(self):
        calls = []

        class FakeSimulator:
            on_transmit = None

        class FakeClient:
            def enqueue_write(self, t):
                calls.append(t.animal_id)

        sim = FakeSimulator()
        client = FakeClient()

        existing = sim.on_transmit

        def _on_transmit(t):
            if existing:
                existing(t)
            client.enqueue_write(t)

        sim.on_transmit = _on_transmit

        sim.on_transmit(_make_telemetry(animal_id=9))
        assert calls == [9]


# =======================================================================
# log_write_result tests
# =======================================================================


class TestLogWriteResult:
    @staticmethod
    def _dummy_logging_cfg():
        from herd_simulator.config import LoggingConfig
        return LoggingConfig(
            log_dir="logs", ground_truth_enabled=False,
            telemetry_csv_enabled=False, events_jsonl_enabled=False,
            buffer_size=10,
        )

    def test_writes_http_result_record(self, tmp_path):
        from herd_simulator.services.logger import RunLogger, log_write_result, _new_buffered_writer, _flush_writer

        writer = _new_buffered_writer(tmp_path / "tx.jsonl", 10)
        rl = RunLogger(
            run_dir=tmp_path,
            manifest={},
            telemetry_writer=None,
            events_writer=None,
            transmissions_writer=writer,
            ground_truth_writer=None,
            logging_cfg=self._dummy_logging_cfg(),
            profiles={},
        )
        log_write_result(rl, 5, 300, "success", status_code=200, attempts=1)
        _flush_writer(writer)

        lines = (tmp_path / "tx.jsonl").read_text().strip().split("\n")
        record = json.loads(lines[0])
        assert record["type"] == "http_result"
        assert record["animal_id"] == 5
        assert record["outcome"] == "success"
        assert record["http_status"] == 200
        assert record["attempts"] == 1

    def test_no_status_code(self, tmp_path):
        from herd_simulator.services.logger import RunLogger, log_write_result, _new_buffered_writer, _flush_writer

        writer = _new_buffered_writer(tmp_path / "tx.jsonl", 10)
        rl = RunLogger(
            run_dir=tmp_path,
            manifest={},
            telemetry_writer=None,
            events_writer=None,
            transmissions_writer=writer,
            ground_truth_writer=None,
            logging_cfg=self._dummy_logging_cfg(),
            profiles={},
        )
        log_write_result(rl, 5, 300, "quota_disabled")
        _flush_writer(writer)

        record = json.loads((tmp_path / "tx.jsonl").read_text().strip())
        assert "http_status" not in record
        assert "attempts" not in record

    def test_noop_when_writer_none(self):
        from herd_simulator.services.logger import RunLogger, log_write_result

        rl = RunLogger(
            run_dir=None,
            manifest={},
            telemetry_writer=None,
            events_writer=None,
            transmissions_writer=None,
            ground_truth_writer=None,
            logging_cfg=self._dummy_logging_cfg(),
            profiles={},
        )
        log_write_result(rl, 5, 300, "success")
