from __future__ import annotations

import pytest
from pydantic import ValidationError

from shared.schemas import AnomalyExplanationQuery, GoldenCase
from stage2_rag_assistant.golden_loading import DEFAULT_GOLDEN_FILES, load_cases, query_for
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record


def _case(query_text: str | None) -> GoldenCase:
    record = generate_mock_record("cow-001")
    return GoldenCase(
        case_id="c-1",
        category="injected",
        source="synthetic_injection",
        injection_type=None,
        input_record=record,
        query_text=query_text,
        gold_anomaly_flag=True,
        gold_driving_signals=record.driving_signals,
        gold_key_facts=[],
        reviewed_by="mechanical",
    )


def test_query_for_returns_none_for_system_auto_case():
    assert query_for(_case(None)) is None


def test_query_for_wraps_query_text():
    query = query_for(_case("why was this cow flagged?"))
    assert isinstance(query, AnomalyExplanationQuery)
    assert query.raw_text == "why was this cow flagged?"


def test_load_cases_raises_loudly_on_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    with pytest.raises(FileNotFoundError, match="does_not_exist.jsonl"):
        load_cases([missing])


def test_load_default_golden_files_returns_all_24_real_cases():
    cases = load_cases(DEFAULT_GOLDEN_FILES)
    assert len(cases) == 24
    assert all(isinstance(c, GoldenCase) for c in cases)


def test_load_cases_validates_each_line(tmp_path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text('{"case_id": "c-1"}\n')  # missing every other required field
    with pytest.raises(ValidationError):
        load_cases([bad_file])
