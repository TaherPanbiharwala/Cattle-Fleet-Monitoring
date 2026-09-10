"""Stage 2's output. See LLM_Diagnostic_Assistant_PRD.md Section 6.4.

No `suspected_condition` field, and `suggested_next_step` is deliberately generic
(monitor / check / escalate) rather than disease-specific treatment. That is the
anomaly-explanation framing carried into the schema itself, not just PRD prose
around it — do not add a diagnosis-shaped field here later without re-reading
PRD Section 5's non-goals first.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field

from ._base import StrictModel

PathTaken = Literal["llm_grounded", "fallback_stage1_output", "fallback_insufficient_data"]


class CitedFact(StrictModel):
    fact_id: str
    source: str
    text: str


class Stage1OutputSummary(StrictModel):
    anomaly_flag: bool
    anomaly_score: float = Field(ge=0, le=1)
    driving_signals: list[str] = Field(default_factory=list)


class AnomalyExplanationResponse(StrictModel):
    query_id: str
    cow_id: str
    path_taken: PathTaken

    anomaly_summary: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str
    cited_facts: list[CitedFact] = Field(default_factory=list)
    contributing_signals: list[str] = Field(default_factory=list)
    suggested_next_step: str

    stage1_output: Stage1OutputSummary
    disagreement_flag: bool

    latency_ms: int = Field(ge=0)
    model_version: str
    timestamp: AwareDatetime
