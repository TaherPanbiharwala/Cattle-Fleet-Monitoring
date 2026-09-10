from __future__ import annotations

import pytest

from stage2_rag_assistant.pipeline.fallback_gate import evaluate_gate

TAU = 0.6
MARGIN = 0.3  # "confident" means anomaly_score <= 0.2 or >= 0.8


@pytest.mark.parametrize(
    "confidence, llm_asserts_anomaly, stage1_flag, stage1_score, expected_path, expected_reason",
    [
        # agree, confident LLM, stage1 confident anomaly -> trust the LLM
        (0.9, True, True, 0.9, "llm_grounded", "none"),
        # agree, but LLM confidence below tau -> fall back on confidence alone
        (0.4, True, True, 0.9, "fallback_stage1_output", "confidence_below_tau"),
        # disagreement, but stage1 was NOT confident (near the 0.5 boundary) -> expected noise, trust the LLM
        (0.9, True, False, 0.55, "llm_grounded", "none"),
        # disagreement, and stage1 WAS confident -> hard override regardless of high LLM confidence
        (0.95, True, False, 0.05, "fallback_stage1_output", "high_severity_disagreement"),
        (0.95, False, True, 0.95, "fallback_stage1_output", "high_severity_disagreement"),
        # exact boundary: stage1_score sits exactly at the "confident" edge -> counts as confident
        (0.95, True, False, 0.2, "fallback_stage1_output", "high_severity_disagreement"),
        # just inside the "not confident" zone -> not high severity, falls through to the confidence check
        (0.95, True, False, 0.21, "llm_grounded", "none"),
    ],
)
def test_gate_decisions(confidence, llm_asserts_anomaly, stage1_flag, stage1_score, expected_path, expected_reason):
    decision = evaluate_gate(
        confidence=confidence,
        llm_asserts_anomaly=llm_asserts_anomaly,
        stage1_anomaly_flag=stage1_flag,
        stage1_anomaly_score=stage1_score,
        tau=TAU,
        high_severity_margin=MARGIN,
    )
    assert decision.path_taken == expected_path
    assert decision.reason == expected_reason


def test_disagreement_flag_is_independent_of_severity():
    """Mild disagreement, stage1 unconfident -> still routed llm_grounded,
    but disagreement_flag must still be True (it's a raw signal, not gated)."""
    decision = evaluate_gate(
        confidence=0.9,
        llm_asserts_anomaly=True,
        stage1_anomaly_flag=False,
        stage1_anomaly_score=0.5,
        tau=TAU,
        high_severity_margin=MARGIN,
    )
    assert decision.path_taken == "llm_grounded"
    assert decision.disagreement_flag is True


def test_agreement_never_sets_disagreement_flag():
    decision = evaluate_gate(
        confidence=0.9,
        llm_asserts_anomaly=True,
        stage1_anomaly_flag=True,
        stage1_anomaly_score=0.9,
        tau=TAU,
        high_severity_margin=MARGIN,
    )
    assert decision.disagreement_flag is False
