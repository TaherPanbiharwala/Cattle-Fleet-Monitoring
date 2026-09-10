"""Optional free text alongside an AnomalyRecord. See LLM_Diagnostic_Assistant_PRD.md Section 6.3."""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime

from ._base import StrictModel

SubmittedBy = Literal["farmer", "vet", "system_auto"]


class AnomalyExplanationQuery(StrictModel):
    query_id: str
    cow_id: str
    raw_text: str | None = None
    submitted_by: SubmittedBy
    timestamp: AwareDatetime
