from __future__ import annotations

import json
from pathlib import Path

from shared.schemas import GoldenCase
from stage2_rag_assistant.config.settings import (
    FallbackConfig,
    GenerationConfig,
    KBConfig,
    LLMConfig,
    RouterConfig,
    Stage2Config,
)
from stage2_rag_assistant.eval.eval_config import EvalConfig, JudgeEmbeddingConfig, JudgeLLMConfig
from stage2_rag_assistant.eval.run_eval import run
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record


def _write_small_golden_set(path: Path) -> None:
    llm_grounded_record = generate_mock_record("cow-e2e-1", anomaly_flag=True).model_copy(
        update={"driving_signals": ["lying_time"], "anomaly_score": 0.8}
    )
    fallback_record = generate_mock_record("cow-e2e-2", anomaly_flag=True).model_copy(
        update={"driving_signals": ["totally_unknown_signal"]}
    )
    cases = [
        GoldenCase(
            case_id="test-llm-grounded",
            category="injected",
            source="synthetic_injection",
            injection_type="mock_lying_time_deviation",
            input_record=llm_grounded_record,
            query_text=None,
            gold_anomaly_flag=True,
            gold_driving_signals=["lying_time"],
            gold_key_facts=["Sustained increases in lying time correlate with early-stage lameness."],
            reviewed_by="mechanical",
        ),
        GoldenCase(
            case_id="test-fallback",
            category="adversarial",
            source="synthetic_adversarial",
            injection_type=None,
            input_record=fallback_record,
            query_text=None,
            gold_anomaly_flag=True,
            gold_driving_signals=["totally_unknown_signal"],
            gold_key_facts=[],
            reviewed_by="mtz",
        ),
    ]
    with path.open("w") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")


def _configs(db_path) -> tuple[Stage2Config, EvalConfig]:
    pipeline_config = Stage2Config(
        llm=LLMConfig(provider="fake", model="fake-model"),
        router=RouterConfig(max_tokens=32, timeout_s=10.0),
        generation=GenerationConfig(max_tokens=512, timeout_s=20.0, max_retries=1),
        fallback=FallbackConfig(tau=0.6, high_severity_margin=0.3),
        kb=KBConfig(db_path=str(db_path)),
    )
    eval_config = EvalConfig(
        judge_llm=JudgeLLMConfig(provider="fake", model="fake-judge"),
        judge_embedding=JudgeEmbeddingConfig(provider="fake", model="fake-embedding"),
    )
    return pipeline_config, eval_config


async def test_run_eval_scores_llm_grounded_and_excludes_fallback(tmp_kb_db, tmp_path):
    golden_path = tmp_path / "small_golden_set.jsonl"
    _write_small_golden_set(golden_path)
    pipeline_config, eval_config = _configs(tmp_kb_db)

    report = await run(
        [golden_path],
        pipeline_config=pipeline_config,
        eval_config=eval_config,
        out_dir=tmp_path / "reports",
    )

    assert report["total_cases"] == 2
    assert report["excluded_by_path_taken"] == {"fallback_insufficient_data": 1}
    assert report["aggregates"]["total_cases_scored"] == 1

    per_case_by_id = {c["case_id"]: c for c in report["per_case"]}
    assert per_case_by_id["test-llm-grounded"]["scored"] is True
    assert per_case_by_id["test-llm-grounded"]["path_taken"] == "llm_grounded"
    assert per_case_by_id["test-fallback"]["scored"] is False
    assert per_case_by_id["test-fallback"]["path_taken"] == "fallback_insufficient_data"


async def test_run_eval_report_has_the_layer2_only_header_and_demo_banner(tmp_kb_db, tmp_path):
    golden_path = tmp_path / "small_golden_set.jsonl"
    _write_small_golden_set(golden_path)
    pipeline_config, eval_config = _configs(tmp_kb_db)

    report = await run(
        [golden_path], pipeline_config=pipeline_config, eval_config=eval_config, out_dir=tmp_path / "reports"
    )

    assert "Layer 2" in report["header"]
    assert "Layer 1" in report["header"]
    assert report["demo_mode"] is True
    assert "DEMO MODE" in report["demo_mode_banner"]


async def test_run_eval_writes_json_and_markdown_files(tmp_kb_db, tmp_path):
    golden_path = tmp_path / "small_golden_set.jsonl"
    _write_small_golden_set(golden_path)
    pipeline_config, eval_config = _configs(tmp_kb_db)
    out_dir = tmp_path / "reports"

    report = await run([golden_path], pipeline_config=pipeline_config, eval_config=eval_config, out_dir=out_dir)

    json_path = Path(report["_json_path"])
    md_path = Path(report["_md_path"])
    assert json_path.exists() and json_path.parent == out_dir
    assert md_path.exists()
    assert json.loads(json_path.read_text())["total_cases"] == 2
    assert "Layer 2" in md_path.read_text()


async def test_run_eval_raises_loudly_on_a_missing_golden_file(tmp_kb_db, tmp_path):
    pipeline_config, eval_config = _configs(tmp_kb_db)
    missing_path = tmp_path / "does_not_exist.jsonl"

    try:
        await run([missing_path], pipeline_config=pipeline_config, eval_config=eval_config, out_dir=tmp_path)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "does_not_exist.jsonl" in str(exc)
