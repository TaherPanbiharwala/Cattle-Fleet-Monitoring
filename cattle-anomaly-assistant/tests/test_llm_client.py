from __future__ import annotations

import pytest

from stage2_rag_assistant.llm.client import LLMOutputError, complete_json


class _StubClient:
    def __init__(self, response) -> None:
        self._response = response

    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int, timeout_s: float):
        return self._response


def _call(response) -> dict:
    return complete_json(_StubClient(response), system_prompt="s", user_prompt="u", max_tokens=8, timeout_s=1.0)


def test_valid_json_round_trips():
    assert _call('{"a": 1}') == {"a": 1}


def test_strips_markdown_code_fence():
    assert _call('```json\n{"a": 1}\n```') == {"a": 1}


def test_invalid_json_raises_llm_output_error():
    with pytest.raises(LLMOutputError):
        _call("not json")


def test_non_object_json_raises_llm_output_error():
    with pytest.raises(LLMOutputError):
        _call("[1, 2, 3]")


@pytest.mark.parametrize("response", [None, ""])
def test_empty_or_none_response_raises_llm_output_error_not_attribute_error(response):
    """Regression test: a real provider (unlike FakeLLMClient) can legitimately
    return no content at all — e.g. a reasoning model exhausting its token
    budget on hidden thinking output before ever answering, found running
    OpenRouter/minimax-m3 live. This must surface as the same LLMOutputError
    a malformed-JSON response would, so generator.py's FR-9 retry-then-
    fallback logic (which only catches LLMOutputError/ValidationError) covers
    it too, not crash as an uncaught AttributeError.
    """
    with pytest.raises(LLMOutputError):
        _call(response)
