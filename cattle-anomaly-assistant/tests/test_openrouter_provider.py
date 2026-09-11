from __future__ import annotations

import json

import pytest

from stage2_rag_assistant.config.settings import LLMConfig
from stage2_rag_assistant.llm.client import build_llm_client
from stage2_rag_assistant.llm.providers.openrouter_provider import (
    OpenRouterError,
    OpenRouterLLMClient,
)


def _scripted_post(status: int, body: bytes):
    calls: list[dict] = []

    def _post(url: str, body_bytes: bytes, headers: dict[str, str], timeout_s: float):
        calls.append({"url": url, "body": json.loads(body_bytes), "headers": headers, "timeout_s": timeout_s})
        return status, body

    return _post, calls


def _success_body(content: str = "hello from minimax") -> bytes:
    return json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()


def test_missing_api_key_raises_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        OpenRouterLLMClient(model="minimax/minimax-m3")


def test_complete_sends_expected_request_and_parses_response(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")
    post_fn, calls = _scripted_post(200, _success_body("this cow is fine"))
    client = OpenRouterLLMClient(model="minimax/minimax-m3", post_fn=post_fn)

    result = client.complete(system_prompt="be helpful", user_prompt="Cow 7, is she ok?", max_tokens=64, timeout_s=5.0)

    assert result == "this cow is fine"
    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert call["headers"]["Authorization"] == "Bearer sk-test-123"
    assert call["timeout_s"] == 5.0
    assert call["body"]["model"] == "minimax/minimax-m3"
    assert call["body"]["max_tokens"] == 64
    assert call["body"]["messages"] == [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "Cow 7, is she ok?"},
    ]


def test_never_logs_or_leaks_the_api_key_into_the_request_body(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    post_fn, calls = _scripted_post(200, _success_body())
    client = OpenRouterLLMClient(model="minimax/minimax-m3", post_fn=post_fn)

    client.complete(system_prompt="s", user_prompt="u", max_tokens=8, timeout_s=1.0)

    assert "sk-super-secret" not in json.dumps(calls[0]["body"])


def test_non_200_status_raises_openrouter_error_with_status_code(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    post_fn, _ = _scripted_post(401, b'{"error": {"message": "invalid api key"}}')
    client = OpenRouterLLMClient(model="minimax/minimax-m3", post_fn=post_fn)

    with pytest.raises(OpenRouterError, match="401"):
        client.complete(system_prompt="s", user_prompt="u", max_tokens=8, timeout_s=1.0)


@pytest.mark.parametrize(
    "bad_body",
    [
        b"not json at all",
        b"{}",
        json.dumps({"choices": []}).encode(),
        json.dumps({"choices": [{"message": {}}]}).encode(),
    ],
)
def test_malformed_response_shape_raises_openrouter_error(monkeypatch, bad_body):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    post_fn, _ = _scripted_post(200, bad_body)
    client = OpenRouterLLMClient(model="minimax/minimax-m3", post_fn=post_fn)

    with pytest.raises(OpenRouterError):
        client.complete(system_prompt="s", user_prompt="u", max_tokens=8, timeout_s=1.0)


def test_build_llm_client_wires_up_openrouter_from_config(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    client = build_llm_client(LLMConfig(provider="openrouter", model="minimax/minimax-m3"))
    assert isinstance(client, OpenRouterLLMClient)


def test_build_llm_client_openrouter_without_key_raises_loudly(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(OpenRouterError, match="OPENROUTER_API_KEY"):
        build_llm_client(LLMConfig(provider="openrouter", model="minimax/minimax-m3"))
