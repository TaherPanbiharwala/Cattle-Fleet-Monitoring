from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from shared.schemas import AnomalyExplanationQuery
from stage2_rag_assistant.mock.generate_mock_records import generate_mock_record
from stage2_rag_assistant.pipeline.record_assembler import assemble


def test_dict_and_object_record_input_assemble_identically():
    """FR-2: the concrete proof the Section 6 contract is parallel-safe — a
    raw dict (Stage 1's real JSON payload shape) and an already-constructed
    AnomalyRecord (the mock generator's Python objects) go through the
    identical call.
    """
    record = generate_mock_record("cow-001")
    from_object = assemble(record, None)
    from_dict = assemble(record.model_dump(mode="json"), None)
    assert from_object.record.model_dump() == from_dict.record.model_dump()


def test_malformed_record_dict_raises_validation_error():
    """FR-1: never proceed on partial/malformed data."""
    with pytest.raises(ValidationError):
        assemble({"cow_id": "cow-001"}, None)  # missing every other required field


def test_none_query_synthesizes_a_system_auto_query_with_no_raw_text():
    """This is what makes 'system-triggered auto-explanation of a Stage-1
    flag always routes to explain_anomaly' (PRD Section 7) a real code path.
    """
    record = generate_mock_record("cow-002")
    assembled = assemble(record, None)
    assert assembled.query.raw_text is None
    assert assembled.query.submitted_by == "system_auto"
    assert assembled.query.cow_id == record.cow_id


def test_explicit_query_object_is_passed_through():
    record = generate_mock_record("cow-003")
    query = AnomalyExplanationQuery(
        query_id="q-explicit",
        cow_id=record.cow_id,
        raw_text="why was she flagged?",
        submitted_by="farmer",
        timestamp=datetime.now(timezone.utc),
    )
    assembled = assemble(record, query)
    assert assembled.query is query


def test_query_dict_is_validated_too():
    record = generate_mock_record("cow-004")
    with pytest.raises(ValidationError):
        assemble(record, {"query_id": "q-1"})  # missing required fields
