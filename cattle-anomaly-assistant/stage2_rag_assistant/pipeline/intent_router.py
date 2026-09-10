"""Stage 2 of 5. See LLM_Diagnostic_Assistant_PRD.md Section 7, FR-3/FR-4."""

from __future__ import annotations

from typing import Literal

from shared.schemas import AnomalyExplanationQuery
from stage2_rag_assistant.config.settings import RouterConfig
from stage2_rag_assistant.llm.client import LLMClient, LLMOutputError, complete_json

Intent = Literal["explain_anomaly", "general_knowledge", "out_of_scope"]
_VALID_INTENTS: frozenset[str] = frozenset({"explain_anomaly", "general_knowledge", "out_of_scope"})

_SYSTEM_PROMPT = (
    "You classify a farmer or vet's free-text query about a dairy cow's "
    "anomaly-detection result. Respond with exactly one JSON object: "
    '{"intent": "<label>"}, where <label> is one of: '
    '"explain_anomaly" (asking what changed / why this cow was flagged), '
    '"general_knowledge" (a general question about cattle health/behavior, '
    "not specific to this flagged event), or "
    '"out_of_scope" (anything else, including requests for a diagnosis or '
    "treatment). No other text."
)


def classify_intent(query: AnomalyExplanationQuery, *, llm_client: LLMClient, config: RouterConfig) -> Intent:
    """FR-3: a single cheap call, separate from the generator's call, with its
    own small max_tokens budget (config.router). When query.raw_text is
    None — the system-triggered auto-explanation case — this returns
    explain_anomaly immediately with ZERO LLM calls; there's nothing to
    classify.

    FR-4 (fail-closed): any LLM output that isn't valid JSON, or whose
    "intent" isn't one of the three labels, maps to out_of_scope. No retry
    here — an uninterpretable router response is itself evidence the query
    doesn't parse as in-scope, unlike the generator's one-retry policy for
    a more complex output shape.
    """
    if query.raw_text is None:
        return "explain_anomaly"

    user_prompt = f'Query: "{query.raw_text}"'
    try:
        result = complete_json(
            llm_client,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=config.max_tokens,
            timeout_s=config.timeout_s,
        )
    except LLMOutputError:
        return "out_of_scope"

    intent = result.get("intent")
    if intent not in _VALID_INTENTS:
        return "out_of_scope"
    return intent  # type: ignore[return-value]  # narrowed by the membership check above
