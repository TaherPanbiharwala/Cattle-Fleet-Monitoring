from __future__ import annotations

import json
from pathlib import Path

from shared.schemas import GoldenCase
from stage2_rag_assistant.calibration.tune_threshold import run
from stage2_rag_assistant.config.settings import (
    DEFAULT_CONFIG_PATH,
    FallbackConfig,
    GenerationConfig,
    KBConfig,
    LLMConfig,
    RouterConfig,
    Stage2Config,
)
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record

TAU = 0.6
MARGIN = 0.3

# Scripted per-cow (confidence, llm_asserts_anomaly). query_text is None for
# every case, so the router never runs — only the generator's calls are
# scripted here, keyed by cow_id parsed out of its user_prompt.
_SCRIPT = {
    "cow-02": (0.85, True),
    "cow-01": (0.85, True),  # agree, confident stage1 -> llm_grounded, bucket 0.8-0.9, correct
    "cow-04": (0.85, True),
    "cow-03": (0.85, False),  # disagree, stage1 NOT confident (score 0.55) -> not high-severity -> llm_grounded, incorrect
    "cow-06": (0.85, True),
    "cow-05": (0.45, True),  # agree, but confidence < tau -> fallback_stage1_output, excluded
    "cow-08": (0.85, True),
    "cow-07": (0.65, False),  # agree (both False), confident stage1 -> llm_grounded, bucket 0.6-0.7, correct
}


def _responder(system_prompt: str, user_prompt: str) -> str:
    if "classify" in system_prompt.lower():
        return json.dumps({"intent": "explain_anomaly"})
    cow_id = user_prompt.split("Cow ", 1)[1].split(",", 1)[0]
    confidence, llm_asserts_anomaly = _SCRIPT[cow_id]
    return json.dumps(
        {
            "anomaly_summary": "t",
            "llm_asserts_anomaly": llm_asserts_anomaly,
            "confidence": confidence,
            "rationale": "t",
            "contributing_signals": [],
            "suggested_next_step": "monitor",
        }
    )


def _case(case_id: str, cow_id: str, *, stage1_flag: bool, stage1_score: float, gold_flag: bool) -> GoldenCase:
    record = generate_mock_record(cow_id, anomaly_flag=stage1_flag).model_copy(
        update={"driving_signals": ["cbt"], "anomaly_flag": stage1_flag, "anomaly_score": stage1_score}
    )
    return GoldenCase(
        case_id=case_id,
        category="injected",
        source="synthetic_injection",
        injection_type=None,
        input_record=record,
        query_text=None,
        gold_anomaly_flag=gold_flag,
        gold_driving_signals=["cbt"],
        gold_key_facts=[],
        reviewed_by="mechanical",
    )


def _write_eight_case_golden_set(path: Path) -> None:
    # Odd-numbered ("case-01/03/05/07") sort into the calibration split (even
    # index after sorting), even-numbered into the held-out test split — see
    # split_calibration_test's even/odd interleave.
    calibration_cases = [
        _case("case-01", "cow-01", stage1_flag=True, stage1_score=0.9, gold_flag=True),
        _case("case-03", "cow-03", stage1_flag=True, stage1_score=0.55, gold_flag=True),
        _case("case-05", "cow-05", stage1_flag=True, stage1_score=0.9, gold_flag=True),
        _case("case-07", "cow-07", stage1_flag=False, stage1_score=0.1, gold_flag=False),
    ]
    filler_test_cases = [
        _case(f"case-{i:02d}", f"cow-{i:02d}", stage1_flag=True, stage1_score=0.9, gold_flag=True)
        for i in (2, 4, 6, 8)
    ]
    with path.open("w") as f:
        for case in calibration_cases + filler_test_cases:
            f.write(case.model_dump_json() + "\n")


def _pipeline_config(db_path) -> Stage2Config:
    return Stage2Config(
        llm=LLMConfig(provider="fake", model="fake-model"),
        router=RouterConfig(max_tokens=32, timeout_s=10.0),
        generation=GenerationConfig(max_tokens=512, timeout_s=20.0, max_retries=1),
        fallback=FallbackConfig(tau=TAU, high_severity_margin=MARGIN),
        kb=KBConfig(db_path=str(db_path)),
    )


def _run_scripted(tmp_kb_db, fake_llm_client, tmp_path, out_dir_name: str = "reports") -> dict:
    golden_path = tmp_path / "golden.jsonl"
    _write_eight_case_golden_set(golden_path)
    client = fake_llm_client(responder=_responder)
    return run(
        [golden_path],
        pipeline_config=_pipeline_config(tmp_kb_db),
        llm_client=client,
        out_dir=tmp_path / out_dir_name,
    )


def test_calibration_split_sizes_and_exclusion(tmp_kb_db, fake_llm_client, tmp_path):
    report = _run_scripted(tmp_kb_db, fake_llm_client, tmp_path)

    assert report["total_golden_cases"] == 8
    assert report["calibration_split_size"] == 4
    assert report["held_out_split_size"] == 4
    assert set(case_id.removeprefix("legacy:") for case_id in report["held_out_scenario_ids"]) == {"case-02", "case-04", "case-06", "case-08"}
    assert report["excluded_by_path_taken"] == {"fallback_stage1_output": 1}


def test_bucket_stats_match_the_scripted_scenario(tmp_kb_db, fake_llm_client, tmp_path):
    report = _run_scripted(tmp_kb_db, fake_llm_client, tmp_path)

    stats_by_bucket = {b["bucket"]: b for b in report["bucket_stats"]}
    assert len(report["bucket_stats"]) == 10  # all deciles present, even empty ones

    bucket_08 = stats_by_bucket["0.8-0.9"]
    assert bucket_08["count"] == 2 and bucket_08["correct_count"] == 1 and bucket_08["accuracy"] == 0.5

    bucket_06 = stats_by_bucket["0.6-0.7"]
    assert bucket_06["count"] == 1 and bucket_06["correct_count"] == 1 and bucket_06["accuracy"] == 1.0

    other_buckets = [b for label, b in stats_by_bucket.items() if label not in ("0.8-0.9", "0.6-0.7")]
    assert all(b["count"] == 0 for b in other_buckets)

    overall = report["overall_llm_accuracy"]
    assert overall["count"] == 3 and overall["correct_count"] == 2


def test_report_has_fixed_keys_with_no_stage1_accuracy_field(tmp_kb_db, fake_llm_client, tmp_path):
    report = _run_scripted(tmp_kb_db, fake_llm_client, tmp_path)

    expected_keys = {
        "generated_at",
        "header",
        "scope_note",
        "demo_mode",
        "demo_mode_banner",
        "configured_tau",
        "tau_policy",
        "total_golden_cases",
        "calibration_split_size",
        "held_out_split_size",
        "calibration_scenario_ids",
        "held_out_scenario_ids",
        "excluded_by_path_taken",
        "held_out_excluded_by_path_taken",
        "bucket_stats",
        "overall_llm_accuracy",
        "held_out_workflow_result",
        "raw_pre_gate",
        "per_case",
        "held_out_per_case",
        "_json_path",
        "_md_path",
    }
    assert set(report.keys()) == expected_keys
    assert "stage1" not in json.dumps(list(report.keys())).lower()
    for case_row in report["per_case"]:
        assert set(case_row.keys()) == {
            "case_id",
            "scenario_id",
            "bucket",
            "confidence",
            "llm_asserts_anomaly",
            "gold_anomaly_flag",
            "correct",
        }


def test_report_files_are_written(tmp_kb_db, fake_llm_client, tmp_path):
    report = _run_scripted(tmp_kb_db, fake_llm_client, tmp_path)

    json_path = Path(report["_json_path"])
    md_path = Path(report["_md_path"])
    assert json_path.exists() and json_path.parent == tmp_path / "reports"
    assert md_path.exists()
    assert json.loads(json_path.read_text())["calibration_split_size"] == 4
    assert "scenario-held-out" in md_path.read_text().lower()


def test_run_never_writes_near_the_real_config(tmp_kb_db, fake_llm_client, tmp_path):
    before = DEFAULT_CONFIG_PATH.read_bytes()
    _run_scripted(tmp_kb_db, fake_llm_client, tmp_path)
    after = DEFAULT_CONFIG_PATH.read_bytes()
    assert before == after


def test_missing_golden_file_raises_loudly(tmp_kb_db, fake_llm_client, tmp_path):
    client = fake_llm_client(responder=_responder)
    missing_path = tmp_path / "does_not_exist.jsonl"
    try:
        run([missing_path], pipeline_config=_pipeline_config(tmp_kb_db), llm_client=client, out_dir=tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "does_not_exist.jsonl" in str(exc)
