"""Stage 5 of 5. See LLM_Diagnostic_Assistant_PRD.md Section 10, FR-11/FR-12.
Pure and deterministic — no LLM call.

Only two of Section 10's three trigger conditions live here. The third
(empty retrieval) is handled one stage earlier, in the orchestrator, right
after retriever.retrieve_facts() — per the Section 7 diagram, an empty
retrieval routes straight to fallback_insufficient_data without Constrained
Generation ever running, so this stage never sees that case.

Takes plain values rather than a GeneratedContent/AnomalyRecord object on
purpose: the gate's job is narrowly about confidence-vs-tau and
disagreement-vs-severity, nothing else those objects carry.
"""

from __future__ import annotations

from typing import Literal

from shared.schemas._base import StrictModel

PathTaken = Literal["llm_grounded", "fallback_stage1_output"]


class GateDecision(StrictModel):
    path_taken: PathTaken
    disagreement_flag: bool
    reason: str


def evaluate_gate(
    *,
    confidence: float,
    llm_asserts_anomaly: bool,
    stage1_anomaly_flag: bool,
    stage1_anomaly_score: float,
    tau: float,
    high_severity_margin: float,
) -> GateDecision:
    """FR-12 operationalization (the PRD states the requirement, not a
    formula — this is the resolution from the M2a plan): a disagreement
    between the LLM's own boolean assertion and Stage 1's flag is
    "high severity" only when Stage 1 itself was confident in its call
    (its anomaly_score was not near the 0.5 decision boundary). A
    disagreement against an uncertain Stage 1 call is expected noise, not a
    safety violation worth a hard override.

    disagreement_flag reflects the raw disagreement regardless of severity,
    so disagreement_flag=True with path_taken="llm_grounded" is a valid,
    meaningful combination: mild disagreement, Stage 1 wasn't confident
    either, the LLM's grounded answer is still trusted.

    FR-12 (checked first, independent of FR-11 — Section 10: "any one [ of the
    trigger conditions ] is sufficient", so it isn't short-circuited by the
    confidence check below):
    """
    disagreement = llm_asserts_anomaly != stage1_anomaly_flag
    stage1_was_confident = (
        stage1_anomaly_score <= 0.5 - high_severity_margin or stage1_anomaly_score >= 0.5 + high_severity_margin
    )
    high_severity = disagreement and stage1_was_confident

    if high_severity:
        return GateDecision(
            path_taken="fallback_stage1_output",
            disagreement_flag=disagreement,
            reason="high_severity_disagreement",
        )

    # FR-11: confidence < tau.
    if confidence < tau:
        return GateDecision(
            path_taken="fallback_stage1_output",
            disagreement_flag=disagreement,
            reason="confidence_below_tau",
        )

    return GateDecision(path_taken="llm_grounded", disagreement_flag=disagreement, reason="none")
