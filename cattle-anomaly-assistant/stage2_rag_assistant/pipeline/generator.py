"""Stage 4 of 5. See LLM_Diagnostic_Assistant_PRD.md Section 7-8, FR-8/FR-9/FR-10.

The empty-retrieval case is NOT handled here — per the Section 7 diagram, the
orchestrator checks retrieval.is_empty and routes to fallback_insufficient_data
one stage earlier, before this function is ever called.
"""

from __future__ import annotations

from pydantic import Field, ValidationError

from shared.schemas import AnomalyExplanationQuery, AnomalyRecord
from shared.schemas._base import StrictModel
from stage2_rag_assistant.config.settings import GenerationConfig
from stage2_rag_assistant.llm.client import LLMClient, LLMOutputError, complete_json
from stage2_rag_assistant.pipeline.retriever import RetrievalResult

_SYSTEM_PROMPT = (
    "You are an assistant that explains a dairy cow's detected behavioral or "
    "physiological anomaly to a farmer or vet. Answer ONLY using the retrieved "
    "knowledge-base facts and the record's own numeric fields given in the user "
    "message below — never invent facts, never reference raw sensor data beyond "
    "what's given, and never provide a disease diagnosis or treatment "
    "recommendation. This system detects anomalies relative to a cow's own "
    "baseline; it does not diagnose illness."
)


class GeneratedContent(StrictModel):
    """Internal-only structured LLM output, not part of shared/schemas/ — the
    orchestrator/response_builder map this into the frozen
    AnomalyExplanationResponse. llm_asserts_anomaly exists purely so the
    Fallback Gate can compare a clean boolean against Stage 1's own flag,
    instead of trying to parse agreement/disagreement out of free text.
    """

    anomaly_summary: str
    llm_asserts_anomaly: bool
    confidence: float = Field(ge=0, le=1)
    rationale: str
    contributing_signals: list[str] = Field(default_factory=list)
    suggested_next_step: str


def generate_explanation(
    record: AnomalyRecord,
    retrieval: RetrievalResult,
    query: AnomalyExplanationQuery,
    *,
    llm_client: LLMClient,
    config: GenerationConfig,
) -> GeneratedContent | None:
    """FR-9: one retry (config.max_retries) on malformed/schema-invalid
    output; still failing after that returns None, which the orchestrator
    maps to fallback_insufficient_data. This function never raises on a bad
    LLM response — a raised exception here would mean a real bug, not an
    expected failure mode.
    """
    user_prompt = _build_user_prompt(record, retrieval, query)

    for _ in range(config.max_retries + 1):
        try:
            raw = complete_json(
                llm_client,
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                max_tokens=config.max_tokens,
                timeout_s=config.timeout_s,
            )
            return GeneratedContent.model_validate(raw)
        except (LLMOutputError, ValidationError):
            continue
    return None


def _build_user_prompt(record: AnomalyRecord, retrieval: RetrievalResult, query: AnomalyExplanationQuery) -> str:
    lines: list[str] = [
        f"Cow {record.cow_id}, window {record.window_id}.",
        f"Stage 1's own anomaly determination: flag={record.anomaly_flag}, score={record.anomaly_score:.2f}.",
        f"Driving signals: {', '.join(record.driving_signals) or 'none'}.",
        "Record's own relevant numbers (do not use anything beyond these):",
    ]

    if "cbt" in record.driving_signals:
        lines.append(
            f"  - core body temperature {record.cbt_c}°C, deviation {record.cbt_deviation_sigma:.2f} "
            "sigma from this cow's own baseline"
        )
    if "lying_time" in record.driving_signals:
        lines.append(
            f"  - lying time {record.lying_time_pct_24h:.1f}% of the last 24h, deviation "
            f"{record.lying_time_deviation_sigma:.2f} sigma from baseline"
        )
    if "activity_magnitude" in record.driving_signals and record.activity_magnitude_deviation_sigma is not None:
        lines.append(f"  - activity magnitude deviation {record.activity_magnitude_deviation_sigma:.2f} sigma")
    if "herd_isolation" in record.driving_signals and record.herd_isolation_score is not None:
        lines.append(f"  - herd isolation score {record.herd_isolation_score:.2f} (0-1 scale)")
    if "behavior_state" in record.driving_signals:
        dist = record.behavior_state_distribution_24h
        lines.append(
            "  - 24h behavior distribution: walking="
            f"{dist.walking:.2f}, grazing={dist.grazing:.2f}, resting={dist.resting:.2f}, "
            f"miscellaneous={dist.miscellaneous:.2f}"
        )
    lines.append(f"  - ambient THI (context only, not itself a deviation signal): {record.thi:.1f}")

    if retrieval.matched_categories:
        lines.append("\nRetrieved knowledge-base facts (answer ONLY using these plus the numbers above):")
        for category in retrieval.matched_categories:
            lines.append(f"- {category.name}: {category.description.strip()}")
            for threshold in category.thresholds:
                lines.append(
                    f"    threshold: {threshold.signal_name} {threshold.threshold_direction} "
                    f"{threshold.threshold_value} (sustained_days={threshold.sustained_days}) "
                    f"— {threshold.notes or ''}"
                )
            for fact in category.cited_facts:
                lines.append(f"    citation [{fact.fact_id}]: {fact.text}")

    if len(record.driving_signals) > 1:
        lines.append(
            "\nIMPORTANT: more than one signal drove this anomaly "
            f"({', '.join(record.driving_signals)}). Reason across ALL of them together, not just "
            "the strongest one — for example, a core-body-temperature deviation reads differently "
            "alongside elevated ambient THI (an environmental explanation is available) than without "
            "it (it points toward the animal specifically)."
        )

    if query.raw_text:
        lines.append(f'\nThe farmer/vet asked: "{query.raw_text}"')

    lines.append(
        "\nRespond with exactly one JSON object with keys: anomaly_summary (string, plain language), "
        "llm_asserts_anomaly (boolean — your own independent judgment of whether this is a real "
        "anomaly, used only for an internal consistency check), confidence (float 0-1), rationale "
        "(string), contributing_signals (array of strings drawn from the driving signals above), "
        "suggested_next_step (string, one of: monitor / visual check / flag for a vet visit if this "
        "persists). Never suggest a diagnosis or treatment. No other text."
    )
    return "\n".join(lines)
