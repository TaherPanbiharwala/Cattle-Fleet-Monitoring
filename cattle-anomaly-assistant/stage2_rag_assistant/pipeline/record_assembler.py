"""Stage 1 of 5. See LLM_Diagnostic_Assistant_PRD.md Section 7, FR-1/FR-2. No LLM call."""

from __future__ import annotations

from shared.schemas import AnomalyExplanationQuery, AnomalyRecord
from shared.schemas._base import StrictModel


class AssembledInput(StrictModel):
    record: AnomalyRecord
    query: AnomalyExplanationQuery


def assemble(
    record_data: dict | AnomalyRecord,
    query_data: dict | AnomalyExplanationQuery | None,
) -> AssembledInput:
    """FR-1: a malformed record_data dict raises pydantic.ValidationError
    unwrapped — AnomalyRecord's own StrictModel already does the rejection
    work (M0); this stage is the pipeline's entry point, not re-validation.

    FR-2: record_data may be a raw dict (Stage 1's real JSON payload) or an
    already-constructed AnomalyRecord (the mock generator's Python objects) —
    both go through this identical call, which is the concrete proof the
    Section 6 contract is parallel-safe.

    query_data=None synthesizes a system_auto query with raw_text=None,
    which is what makes "system-triggered auto-explanation of a Stage-1 flag
    always routes to explain_anomaly" (Section 7) a real code path rather
    than a special case the router has to know about.
    """
    record = record_data if isinstance(record_data, AnomalyRecord) else AnomalyRecord.model_validate(record_data)

    if query_data is None:
        query = AnomalyExplanationQuery(
            query_id=f"auto-{record.window_id}",
            cow_id=record.cow_id,
            raw_text=None,
            submitted_by="system_auto",
            timestamp=record.timestamp,
        )
    elif isinstance(query_data, AnomalyExplanationQuery):
        query = query_data
    else:
        query = AnomalyExplanationQuery.model_validate(query_data)

    return AssembledInput(record=record, query=query)
