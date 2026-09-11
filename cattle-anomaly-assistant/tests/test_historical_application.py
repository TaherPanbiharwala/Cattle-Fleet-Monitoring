from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from shared.schemas import AnomalyRecord, HistoricalBehaviorContext
from stage1_anomaly_detection.behavior_classifier.historical_demo import historical_demo_anomaly_records
from stage2_rag_assistant.config.settings import FallbackConfig, GenerationConfig, KBConfig, LLMConfig, RouterConfig, Stage2Config
from stage2_rag_assistant.historical_application import (
    CONTROL_RECORD_COUNT,
    EXPECTED_MODEL_SHA256,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_METRICS_SHA256,
    FLAGGED_RECORD_COUNT,
    HistoricalApplicationError,
    preflight,
    run,
    select_historical_cusum_windows,
)
from stage2_rag_assistant.pipeline.generator import _build_user_prompt
from stage2_rag_assistant.pipeline.retriever import RetrievalResult


def _summary() -> dict:
    return {
        "schema_version": 1,
        "source_kind": "wasp_public_historical_dataset",
        "context_relation": "cross_dataset_historical_demo",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "manifest_sha256": "b" * 64,
        "behavior_state": "grazing",
        "behavior_state_confidence": 0.8,
        "behavior_state_distribution": {"walking": 0.1, "grazing": 0.6, "resting": 0.2, "miscellaneous": 0.1},
        "observed_windows": 100,
        "source_csv_files": 4,
        "confidence_kind": "uncalibrated_max_class_probability",
        "source_sha256": "c" * 64,
        "window_exclusions": {"source_rows": 500, "valid_rows": 500, "timestamp_breaks": 0, "cadence_breaks": 0, "short_segment_rows": 0},
    }


def _row(index: int, *, flagged: bool) -> dict:
    cow = f"T{(index % 10) + 1:02d}"
    day = f"2023-08-{(index // 10) + 1:02d}"
    return {
        "schema_version": 1,
        "cow_id": cow,
        "window_id": f"{cow}:{day}",
        "date": day,
        "timestamp": f"{day}T00:00:00-05:00",
        "detector_state": "monitoring",
        "cbt_c": 38.5,
        "cbt_deviation_sigma": 2.6 if flagged else 0.2,
        "cbt_cusum_value": 5.2 if flagged else 0.4,
        "lying_time_pct_24h": 40.0,
        "lying_time_deviation_sigma": 0.1,
        "thi": 68.0,
        "activity_magnitude_deviation_sigma": 0.0,
        "anomaly_flag": flagged,
        "anomaly_score": 0.52 if flagged else 0.04,
        "driving_signals": ["cbt"] if flagged else [],
        "injected_signals": [],
    }


def _rows() -> list[dict]:
    return [*(_row(index, flagged=True) for index in range(FLAGGED_RECORD_COUNT)), *(_row(index + 20, flagged=False) for index in range(CONTROL_RECORD_COUNT))]


def _config(db_path: Path) -> Stage2Config:
    return Stage2Config(
        llm=LLMConfig(provider="openrouter", model="test-model"),
        router=RouterConfig(max_tokens=32, timeout_s=1),
        generation=GenerationConfig(max_tokens=64, timeout_s=1, max_retries=0),
        fallback=FallbackConfig(tau=0.7, high_severity_margin=0.3),
        kb=KBConfig(db_path=str(db_path)),
    )


def _write_cusum_input(tmp_path: Path) -> Path:
    output = tmp_path / "cusum"
    output.mkdir()
    source = output / "anomaly_windows.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in _rows()), encoding="utf-8")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "artifact": "mmcows_daily_personal_baseline_spc_cusum",
                "config": {"timezone": "America/Chicago", "baseline_days": 7, "coverage_minimum": 0.75, "cusum_k": 0.5, "cusum_h": 5.0},
            }
        ),
        encoding="utf-8",
    )
    return source


def test_cross_dataset_records_require_dedicated_context_and_never_reuse_daily_fields() -> None:
    summary = _summary()
    context = HistoricalBehaviorContext(**{key: summary[key] for key in (
        "schema_version", "source_kind", "context_relation", "model_sha256", "manifest_sha256", "behavior_state",
        "behavior_state_confidence", "behavior_state_distribution", "observed_windows", "source_csv_files", "confidence_kind",
    )})
    base = {
        "cow_id": "T01",
        "timestamp": "2023-08-02T05:00:00Z",
        "window_id": "T01:2023-08-01",
        "behavior_context_source": "wasp_public_historical_dataset",
        "behavior_context_relation": "cross_dataset_historical_demo",
        "behavior_model_sha256": EXPECTED_MODEL_SHA256,
        "historical_behavior_context": context,
        "cbt_c": 38.5,
        "cbt_deviation_sigma": 2.0,
        "cbt_cusum_value": 5.0,
        "lying_time_pct_24h": 40.0,
        "lying_time_deviation_sigma": 0.1,
        "thi": 68.0,
        "anomaly_flag": True,
        "anomaly_score": 0.5,
        "driving_signals": ["cbt"],
    }
    assert AnomalyRecord.model_validate(base).behavior_state is None
    with pytest.raises(ValidationError, match="cow-day behavior_state"):
        AnomalyRecord.model_validate({**base, "behavior_state": "grazing"})
    with pytest.raises(ValidationError, match="require historical_behavior_context"):
        AnomalyRecord.model_validate({key: value for key, value in base.items() if key != "historical_behavior_context"})

    record = historical_demo_anomaly_records(
        [_row(1, flagged=True)],
        deployment_id="mmcows-public-2023",
        timezone="America/Chicago",
        behavior_summary=summary,
    )[0]
    assert {"behavior_state", "behavior_state_confidence", "behavior_state_distribution_24h"}.isdisjoint(record)


def test_selection_keeps_exact_flagged_and_distributed_controls() -> None:
    selected = select_historical_cusum_windows(_rows())
    assert len(selected) == 24
    assert sum(row["anomaly_flag"] for row in selected) == 11
    assert sum(not row["anomaly_flag"] for row in selected) == 13
    assert len({row["cow_id"] for row in selected if not row["anomaly_flag"]}) == 10


def test_checked_in_friend_model_is_pinned_by_exact_hash() -> None:
    from stage2_rag_assistant.historical_application import _validate_pinned_model

    assert _validate_pinned_model() == {
        "model_sha256": EXPECTED_MODEL_SHA256,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "metrics_sha256": EXPECTED_METRICS_SHA256,
    }


def test_wasp_time_breaks_never_create_a_window_across_a_gap(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from stage1_anomaly_detection.behavior_classifier.historical_demo import SENSOR_COLUMNS, _event_windows

    path = tmp_path / "recording.csv"
    # The real WASP files use an unzoned recording clock. It is sufficient
    # for cadence validation because it never enters a MmCows join/output.
    first = datetime(2023, 1, 1)
    rows = []
    for index in range(50):
        rows.append(["%s" % (first + timedelta(seconds=index / 10)).isoformat(), *(["1.0"] * len(SENSOR_COLUMNS))])
    # This source gap must end the previous segment rather than be bridged.
    for index in range(50):
        rows.append(["%s" % (first + timedelta(seconds=10 + index / 10)).isoformat(), *(["1.0"] * len(SENSOR_COLUMNS))])
    path.write_text(",".join(("Time", *SENSOR_COLUMNS)) + "\n" + "\n".join(",".join(row) for row in rows) + "\n", encoding="utf-8")

    windows, diagnostics = _event_windows(path)
    assert len(windows) == 2
    assert diagnostics["cadence_breaks"] == 1
    assert diagnostics["valid_rows"] == 100


def test_historical_context_is_not_sent_to_the_llm_as_cow_specific_evidence() -> None:
    record = historical_demo_anomaly_records([_row(1, flagged=True)], deployment_id="mmcows-public-2023", timezone="America/Chicago", behavior_summary=_summary())[0]
    prompt = _build_user_prompt(AnomalyRecord.model_validate(record), RetrievalResult(matched_categories=[], is_empty=True), query=type("Query", (), {"raw_text": None})())
    assert "WASP" not in prompt
    assert "historical aggregate" not in prompt.lower()
    assert "Driving signals: cbt" in prompt


def test_historical_context_disclaimer_is_fixed_in_final_explanations(tmp_kb_db: Path, fake_llm_client) -> None:
    from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

    record = historical_demo_anomaly_records(
        [_row(1, flagged=True)],
        deployment_id="mmcows-public-2023",
        timezone="America/Chicago",
        behavior_summary=_summary(),
    )[0]
    response = run_pipeline(record, None, config=_config(tmp_kb_db), llm_client=fake_llm_client())
    assert response.historical_context_disclaimer == (
        "Historical aggregate from the separate WASP public dataset; it is not a measurement of this "
        "MmCows cow or day and is not evidence for this anomaly indicator."
    )


def test_preflight_has_no_output_and_run_publishes_exactly_five_derived_artifacts(tmp_path: Path, tmp_kb_db: Path, fake_llm_client, monkeypatch: pytest.MonkeyPatch) -> None:
    import stage2_rag_assistant.historical_application as app

    source = _write_cusum_input(tmp_path)
    monkeypatch.setattr(app, "_validate_pinned_model", lambda: {"model_sha256": EXPECTED_MODEL_SHA256, "manifest_sha256": "b" * 64, "metrics_sha256": "d" * 64})
    monkeypatch.setattr(app, "historical_behavior_summary", lambda **_: _summary())
    config = _config(tmp_kb_db)
    summary, selected, _ = preflight(wasp_dataset_dir=tmp_path / "wasp", cusum_windows_jsonl=source, pipeline_config=config)
    assert summary["selected_record_count"] == 24
    assert len(selected) == 24
    assert not (tmp_path / "preflight-output").exists()

    result = run(
        wasp_dataset_dir=tmp_path / "wasp",
        cusum_windows_jsonl=source,
        output_dir=tmp_path / "published",
        pipeline_config=config,
        llm_client=fake_llm_client(),
    )
    assert result["record_count"] == 24
    assert {path.name for path in (tmp_path / "published").iterdir()} == {
        "anomaly_records.jsonl", "explanations.jsonl", "historical_behavior_summary.json", "audit.jsonl", "run_manifest.json"
    }
    assert "raw_text" not in (tmp_path / "published" / "audit.jsonl").read_text(encoding="utf-8")


def test_run_rejects_an_existing_output_before_calling_a_provider(tmp_path: Path, tmp_kb_db: Path) -> None:
    output = tmp_path / "published"
    output.mkdir()
    with pytest.raises(HistoricalApplicationError, match="OUTPUT_EXISTS"):
        run(
            wasp_dataset_dir=tmp_path / "wasp",
            cusum_windows_jsonl=tmp_path / "missing.jsonl",
            output_dir=output,
            pipeline_config=_config(tmp_kb_db),
        )


def test_provider_failure_uses_exit_code_three_and_redacts_details(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys) -> None:
    import stage2_rag_assistant.historical_application as app

    monkeypatch.setattr(app, "load_config", lambda _: _config(tmp_path / "unused.sqlite3"))
    monkeypatch.setattr(
        app,
        "run",
        lambda **_: (_ for _ in ()).throw(app.HistoricalApplicationError("OPENROUTER_AUTH_FAILED", "OpenRouter rejected the configured credentials.", {})),
    )
    assert app.main(["run", "--wasp-dataset-dir", "/a", "--cusum-windows-jsonl", "/b", "--output-dir", "/c"]) == 3
    assert "OPENROUTER_AUTH_FAILED" in capsys.readouterr().err
