"""Builds the final, frozen AnomalyExplanationResponse for every pipeline
path. See LLM_Diagnostic_Assistant_PRD.md Section 6.4, FR-13.
"""

from __future__ import annotations

from datetime import datetime, timezone

from shared.schemas import (
    HISTORICAL_BEHAVIOR_DISCLAIMER,
    AnomalyExplanationResponse,
    AnomalyRecord,
    Stage1OutputSummary,
)
from stage2_rag_assistant.pipeline.fallback_gate import GateDecision
from stage2_rag_assistant.pipeline.generator import GeneratedContent
from stage2_rag_assistant.pipeline.retriever import RetrievalResult


def _stage1_output(record: AnomalyRecord) -> Stage1OutputSummary:
    """FR-13: every response includes this, built once from the assembled
    record, regardless of which path the query took."""
    return Stage1OutputSummary(
        anomaly_flag=record.anomaly_flag,
        anomaly_score=record.anomaly_score,
        driving_signals=record.driving_signals,
    )


def _base_kwargs(query_id: str, cow_id: str, record: AnomalyRecord, *, model_version: str, latency_ms: int) -> dict:
    return dict(
        query_id=query_id,
        cow_id=cow_id,
        stage1_output=_stage1_output(record),
        latency_ms=latency_ms,
        model_version=model_version,
        timestamp=datetime.now(timezone.utc),
        historical_context_disclaimer=(
            HISTORICAL_BEHAVIOR_DISCLAIMER if record.historical_behavior_context is not None else None
        ),
    )


def build_out_of_scope(
    query_id: str, cow_id: str, record: AnomalyRecord, *, model_version: str, latency_ms: int
) -> AnomalyExplanationResponse:
    """PRD Section 7's diagram routes 'out of scope' to a node with no edge
    to the final response, and the frozen PathTaken literal has no
    out_of_scope value. Resolved in the M2a plan: map to
    fallback_insufficient_data, distinguishable from a real empty-retrieval
    case only via `rationale` — there's no better option without changing
    the frozen contract.
    """
    return AnomalyExplanationResponse(
        **_base_kwargs(query_id, cow_id, record, model_version=model_version, latency_ms=latency_ms),
        path_taken="fallback_insufficient_data",
        anomaly_summary=None,
        confidence=None,
        rationale="Query was classified out of scope for this assistant — ask about this cow's "
        "flagged anomaly, or consult a vet directly for diagnosis/treatment questions.",
        cited_facts=[],
        contributing_signals=[],
        suggested_next_step="monitor",
        disagreement_flag=False,
    )


def build_insufficient_data(
    query_id: str, cow_id: str, record: AnomalyRecord, *, reason: str, model_version: str, latency_ms: int
) -> AnomalyExplanationResponse:
    return AnomalyExplanationResponse(
        **_base_kwargs(query_id, cow_id, record, model_version=model_version, latency_ms=latency_ms),
        path_taken="fallback_insufficient_data",
        anomaly_summary=None,
        confidence=None,
        rationale=f"Insufficient grounded data to answer ({reason}).",
        cited_facts=[],
        contributing_signals=[],
        suggested_next_step="monitor",
        disagreement_flag=False,
    )


def build(
    query_id: str,
    cow_id: str,
    record: AnomalyRecord,
    retrieval: RetrievalResult,
    generated: GeneratedContent,
    decision: GateDecision,
    *,
    model_version: str,
    latency_ms: int,
) -> AnomalyExplanationResponse:
    """On a fallback_stage1_output decision, don't trust the LLM's prose —
    build anomaly_summary/suggested_next_step from Stage 1's own numbers
    instead. `generated` is still logged by the orchestrator via audit_log
    for later inspection either way, just not treated as authoritative here.
    """
    base = _base_kwargs(query_id, cow_id, record, model_version=model_version, latency_ms=latency_ms)
    cited_facts = [fact for category in retrieval.matched_categories for fact in category.cited_facts]

    if decision.path_taken == "llm_grounded":
        return AnomalyExplanationResponse(
            **base,
            path_taken="llm_grounded",
            anomaly_summary=generated.anomaly_summary,
            confidence=generated.confidence,
            rationale=generated.rationale,
            cited_facts=cited_facts,
            contributing_signals=generated.contributing_signals,
            suggested_next_step=generated.suggested_next_step,
            disagreement_flag=decision.disagreement_flag,
        )

    driving = ", ".join(record.driving_signals) or "no specific signal"
    return AnomalyExplanationResponse(
        **base,
        path_taken="fallback_stage1_output",
        anomaly_summary=f"This cow's {driving} deviated from her own baseline "
        f"(Stage 1 anomaly score {record.anomaly_score:.2f}).",
        confidence=None,
        rationale=f"Falling back to Stage 1's own determination ({decision.reason}).",
        cited_facts=cited_facts,
        contributing_signals=record.driving_signals,
        suggested_next_step="flag for a vet visit if this persists" if record.anomaly_score >= 0.7 else "monitor",
        disagreement_flag=decision.disagreement_flag,
    )
