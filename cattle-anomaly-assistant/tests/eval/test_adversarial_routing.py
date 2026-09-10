"""Runs the hand-authored adversarial golden cases through the real
orchestrator, mirroring tests/stage2/test_orchestrator_end_to_end.py's
style. These are routing/edge-condition regression fixtures, not Ragas
scoring fixtures — see stage2_rag_assistant/eval/run_eval.py for why only
llm_grounded responses get scored at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from shared.schemas import AnomalyExplanationQuery, GoldenCase
from stage2_rag_assistant.config.settings import (
    FallbackConfig,
    GenerationConfig,
    KBConfig,
    LLMConfig,
    RouterConfig,
    Stage2Config,
)
from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

GOLDEN_PATH = Path(__file__).parents[1].parent / "stage2_rag_assistant" / "eval" / "golden" / "adversarial_cases.jsonl"

_GENERATED_RESPONSE = json.dumps(
    {
        "anomaly_summary": "summary",
        "llm_asserts_anomaly": True,
        "confidence": 0.8,
        "rationale": "rationale",
        "contributing_signals": [],
        "suggested_next_step": "monitor",
    }
)


def _load_cases() -> list[GoldenCase]:
    with GOLDEN_PATH.open() as f:
        return [GoldenCase.model_validate_json(line) for line in f]


def _config(db_path) -> Stage2Config:
    return Stage2Config(
        llm=LLMConfig(provider="fake", model="fake-model"),
        router=RouterConfig(max_tokens=32, timeout_s=10.0),
        generation=GenerationConfig(max_tokens=512, timeout_s=20.0, max_retries=1),
        fallback=FallbackConfig(tau=0.6, high_severity_margin=0.3),
        kb=KBConfig(db_path=str(db_path)),
    )


def _case_by_id(case_id: str) -> GoldenCase:
    for case in _load_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def _query_for(case: GoldenCase) -> AnomalyExplanationQuery | None:
    """None (system-auto) stays None; any query_text (including "") gets
    wrapped so the router actually sees it."""
    if case.query_text is None:
        return None
    return AnomalyExplanationQuery(
        query_id=f"eval-{case.case_id}",
        cow_id=case.input_record.cow_id,
        raw_text=case.query_text,
        submitted_by="vet",
        timestamp=datetime.now(timezone.utc),
    )


def test_all_seven_adversarial_cases_are_present():
    assert len(_load_cases()) == 7


@pytest.mark.parametrize(
    "case_id,router_response",
    [
        ("adv-out-of-scope-treatment-request", '{"intent": "out_of_scope"}'),
        ("adv-out-of-scope-unrelated-question", '{"intent": "out_of_scope"}'),
    ],
)
def test_out_of_scope_cases_never_reach_generation(tmp_kb_db, fake_llm_client, case_id, router_response):
    case = _case_by_id(case_id)
    client = fake_llm_client(responses=[router_response])
    response = run_pipeline(case.input_record, _query_for(case), config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_insufficient_data"
    assert len(client.call_log) == 1  # router only — generator never called


def test_unmapped_driving_signal_short_circuits_before_generation(tmp_kb_db, fake_llm_client):
    case = _case_by_id("adv-unmapped-driving-signal")
    client = fake_llm_client(responses=[])
    response = run_pipeline(case.input_record, _query_for(case), config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "fallback_insufficient_data"
    assert client.call_log == []


@pytest.mark.parametrize(
    "case_id",
    [
        "adv-activity-signal-flagged-but-field-missing",
        "adv-herd-isolation-flagged-but-score-missing",
        "adv-noisy-recent-anomaly-history-not-referenced",
    ],
)
def test_missing_field_and_history_cases_still_reach_llm_grounded(tmp_kb_db, fake_llm_client, case_id):
    case = _case_by_id(case_id)
    client = fake_llm_client(responses=[_GENERATED_RESPONSE])
    response = run_pipeline(case.input_record, _query_for(case), config=_config(tmp_kb_db), llm_client=client)
    assert response.path_taken == "llm_grounded"


def test_empty_string_query_text_triggers_a_router_call_unlike_none(tmp_kb_db, fake_llm_client):
    case = _case_by_id("adv-empty-string-query-text-not-treated-as-auto")
    assert case.query_text == ""
    client = fake_llm_client(responses=['{"intent": "explain_anomaly"}', _GENERATED_RESPONSE])
    response = run_pipeline(case.input_record, _query_for(case), config=_config(tmp_kb_db), llm_client=client)
    assert len(client.call_log) == 2  # router call + generator call, unlike None's zero-router-call path
    assert response.path_taken == "llm_grounded"


def test_generator_prompt_never_includes_recent_anomaly_history_content(tmp_kb_db, fake_llm_client):
    case = _case_by_id("adv-noisy-recent-anomaly-history-not-referenced")
    assert case.input_record.recent_anomaly_history  # sanity: the fixture actually has history populated
    client = fake_llm_client(responses=[_GENERATED_RESPONSE])
    run_pipeline(case.input_record, _query_for(case), config=_config(tmp_kb_db), llm_client=client)
    prompt = client.call_log[0]["user_prompt"]
    assert "recent_anomaly_history" not in prompt
    assert "2026-05-30" not in prompt
