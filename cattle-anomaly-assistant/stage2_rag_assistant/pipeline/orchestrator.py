"""Wires all 5 pipeline stages end to end. See LLM_Diagnostic_Assistant_PRD.md
Section 7's mermaid diagram — this module IS that diagram, executed.
"""

from __future__ import annotations

import time

from shared.schemas import AnomalyExplanationQuery, AnomalyExplanationResponse, AnomalyRecord
from stage2_rag_assistant.config.settings import Stage2Config
from stage2_rag_assistant.llm.client import LLMClient
from stage2_rag_assistant.pipeline import (
    audit_log,
    fallback_gate,
    generator,
    intent_router,
    record_assembler,
    response_builder,
    retriever,
)


def run_pipeline(
    record_data: dict | AnomalyRecord,
    query_data: dict | AnomalyExplanationQuery | None,
    *,
    config: Stage2Config,
    llm_client: LLMClient,
) -> AnomalyExplanationResponse:
    """Always returns a schema-valid AnomalyExplanationResponse; a malformed
    input record is the one case that raises (pydantic.ValidationError) —
    FR-1's "never proceed on partial data" read as fail loudly, not silently
    degrade.
    """
    start = time.monotonic()

    assembled = record_assembler.assemble(record_data, query_data)
    query_id, cow_id = assembled.query.query_id, assembled.record.cow_id
    audit_log.log_event(
        query_id=query_id, stage="record_assembler", cow_id=cow_id, window_id=assembled.record.window_id
    )

    intent = intent_router.classify_intent(assembled.query, llm_client=llm_client, config=config.router)
    audit_log.log_event(query_id=query_id, stage="intent_router", intent=intent)
    if intent == "out_of_scope":
        return response_builder.build_out_of_scope(
            query_id, cow_id, assembled.record, model_version=config.llm.model, latency_ms=_elapsed_ms(start)
        )

    retrieval = retriever.retrieve_facts(
        assembled.record.driving_signals, assembled.query.raw_text, db_path=config.kb.db_path
    )
    audit_log.log_event(
        query_id=query_id,
        stage="retriever",
        matched_categories=[c.name for c in retrieval.matched_categories],
        is_empty=retrieval.is_empty,
    )
    if retrieval.is_empty:
        return response_builder.build_insufficient_data(
            query_id,
            cow_id,
            assembled.record,
            reason="empty_retrieval",
            model_version=config.llm.model,
            latency_ms=_elapsed_ms(start),
        )

    generated = generator.generate_explanation(
        assembled.record, retrieval, assembled.query, llm_client=llm_client, config=config.generation
    )
    audit_log.log_event(
        query_id=query_id,
        stage="generator",
        succeeded=generated is not None,
        generated=generated.model_dump() if generated is not None else None,
    )
    if generated is None:
        return response_builder.build_insufficient_data(
            query_id,
            cow_id,
            assembled.record,
            reason="generation_retry_exhausted",
            model_version=config.llm.model,
            latency_ms=_elapsed_ms(start),
        )

    decision = fallback_gate.evaluate_gate(
        confidence=generated.confidence,
        llm_asserts_anomaly=generated.llm_asserts_anomaly,
        stage1_anomaly_flag=assembled.record.anomaly_flag,
        stage1_anomaly_score=assembled.record.anomaly_score,
        tau=config.fallback.tau,
        high_severity_margin=config.fallback.high_severity_margin,
    )
    audit_log.log_event(query_id=query_id, stage="fallback_gate", path_taken=decision.path_taken, reason=decision.reason)

    return response_builder.build(
        query_id,
        cow_id,
        assembled.record,
        retrieval,
        generated,
        decision,
        model_version=config.llm.model,
        latency_ms=_elapsed_ms(start),
    )


def _elapsed_ms(start: float) -> int:
    return max(1, round((time.monotonic() - start) * 1000))
