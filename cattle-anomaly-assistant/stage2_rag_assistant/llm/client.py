"""LLMClient interface + JSON-completion helper + provider factory.

Mirrors src/herd_simulator/services/thingspeak.py's thin-wrapper idiom
(DECISION.md ADR-019: "_http_post/_http_get provide clean mockability"),
adapted to a dependency-injected Protocol instead of a patched private
function, since this code is designed for DI from the start.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from stage2_rag_assistant.config.settings import LLMConfig


class LLMClient(Protocol):
    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int, timeout_s: float) -> str: ...


class LLMOutputError(Exception):
    """The LLM's response could not be parsed as the expected JSON object."""


def complete_json(
    client: LLMClient,
    *,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    timeout_s: float,
) -> dict:
    """Calls client.complete(), strips a markdown code fence if present, and
    json.loads()s the result. Raises LLMOutputError on any failure — callers
    (intent_router, generator) own their own retry/fail-closed policy; this
    helper never retries.
    """
    raw = client.complete(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens, timeout_s=timeout_s)
    text = _strip_code_fence(raw)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputError(f"LLM response was not valid JSON: {raw!r}") from exc
    if not isinstance(parsed, dict):
        raise LLMOutputError(f"LLM response JSON was not an object: {raw!r}")
    return parsed


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def build_llm_client(config: "LLMConfig") -> LLMClient:
    """The one place that branches on the provider name. Adding a real
    provider later means adding one module under providers/ and one branch
    here — nothing else in the pipeline changes.
    """
    if config.provider == "fake":
        from stage2_rag_assistant.llm.providers.fake_provider import FakeLLMClient

        return FakeLLMClient()
    if config.provider == "openrouter":
        from stage2_rag_assistant.llm.providers.openrouter_provider import OpenRouterLLMClient

        return OpenRouterLLMClient(model=config.model)
    raise NotImplementedError(
        f"LLM provider {config.provider!r} is not implemented yet — see LLM_ASSISTANT_STATUS.md Open "
        "Question 2. Set llm.provider: fake in config until a real provider is chosen and built."
    )
