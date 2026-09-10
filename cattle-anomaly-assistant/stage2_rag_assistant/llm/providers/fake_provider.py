"""Deterministic, network-free stand-in for a real LLM provider. Used both as
the test double (scripted responses/responder) and as config.llm.provider's
actual default (PRD Open Question 2 deferred a real provider) — so the whole
Stage 2 pipeline is runnable with zero API keys.
"""

from __future__ import annotations

import json
from collections.abc import Callable

Responder = Callable[[str, str], str]


def _default_responder(system_prompt: str, user_prompt: str) -> str:
    """A generic, schema-plausible reply for whichever stage is calling —
    good enough for ad-hoc runs; tests script exact responses instead of
    relying on this.

    Distinguishes the caller by system_prompt content rather than returning
    one blob with both shapes' fields: GeneratedContent is a StrictModel
    (extra="forbid"), so a response carrying an unrelated "intent" key would
    make every real generation call fail validation and silently exhaust its
    retries — found by running the manual end-to-end sanity check, not by
    the unit tests (which always script an exact response per stage).
    """
    if "classify" in system_prompt.lower():
        return json.dumps({"intent": "explain_anomaly"})
    return json.dumps(
        {
            "anomaly_summary": "This cow's behavior shows a mock-generated deviation from her baseline.",
            "llm_asserts_anomaly": True,
            "confidence": 0.75,
            "rationale": "Fake LLM client default response — no real reasoning performed.",
            "contributing_signals": [],
            "suggested_next_step": "monitor",
        }
    )


class FakeLLMClient:
    def __init__(
        self,
        responses: list[str] | None = None,
        responder: Responder | None = None,
    ) -> None:
        if responses is not None and responder is not None:
            raise ValueError("pass at most one of responses, responder")
        self._responses = list(responses) if responses is not None else None
        self._responder = responder or (None if responses is not None else _default_responder)
        self.call_log: list[dict] = []

    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int, timeout_s: float) -> str:
        self.call_log.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
            }
        )
        if self._responder is not None:
            return self._responder(system_prompt, user_prompt)
        if not self._responses:
            raise RuntimeError("FakeLLMClient has no more scripted responses")
        return self._responses.pop(0)
