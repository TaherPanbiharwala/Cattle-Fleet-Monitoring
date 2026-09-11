"""OpenRouter provider — resolves LLM_ASSISTANT_STATUS.md Open Question 2
for the main Stage 2 pipeline (not the Ragas eval judge, which is a
separate, still-deferred decision — see eval/ragas_llm.py).

OpenRouter (https://openrouter.ai) is an OpenAI-compatible chat-completions
proxy in front of many hosted models. This uses stdlib urllib rather than
adding an SDK dependency, mirroring src/herd_simulator/services/thingspeak.py's
_http_post/_http_get idiom (DECISION.md ADR-019: a bare, swappable function
is trivially mockable in tests without a mocking library) — that idiom is
already the one client.py's own docstring points to for this exact
extension point.

The API key is read from the OPENROUTER_API_KEY environment variable only
— never hardcoded, never logged (AGENTS.md golden rule 2) — and construction
fails loudly if it's missing rather than deferring the failure to the first
real call.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable

_API_URL = "https://openrouter.ai/api/v1/chat/completions"
_API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

PostFn = Callable[[str, bytes, dict[str, str], float], tuple[int, bytes]]


class OpenRouterError(Exception):
    """OpenRouter responded, but with a non-2xx status or an unparseable
    body. Deliberately NOT caught by pipeline/generator.py's retry loop
    (which only retries LLMOutputError/ValidationError, i.e. malformed
    *content*) — a transport/auth failure is an operational bug, not an
    expected LLM failure mode, and should surface rather than be silently
    swallowed into a fallback response. A genuine network failure (DNS,
    connection refused, timeout) raises the stdlib urllib.error.URLError /
    OSError / TimeoutError directly instead of being wrapped here.
    """


def _http_post_json(url: str, body: bytes, headers: dict[str, str], timeout_s: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, method="POST", headers=headers)
    try:
        resp = urllib.request.urlopen(req, timeout=timeout_s)
        return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


class OpenRouterLLMClient:
    """Implements llm/client.py's LLMClient protocol against OpenRouter's
    OpenAI-compatible /chat/completions endpoint.

    post_fn is a dependency-injection point purely for tests — production
    code always uses the module-level default.
    """

    def __init__(self, *, model: str, post_fn: PostFn = _http_post_json) -> None:
        api_key = os.environ.get(_API_KEY_ENV_VAR)
        if not api_key:
            raise OpenRouterError(
                f"{_API_KEY_ENV_VAR} is not set. Export it yourself (or add it to a gitignored .env you "
                "source before running) — llm.provider: openrouter never reads a key from config."
            )
        self._model = model
        self._api_key = api_key
        self._post = post_fn

    def complete(self, *, system_prompt: str, user_prompt: str, max_tokens: int, timeout_s: float) -> str:
        body = json.dumps(
            {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        status, raw_body = self._post(_API_URL, body, headers, timeout_s)
        if status != 200:
            raise OpenRouterError(f"OpenRouter returned HTTP {status}: {raw_body[:500]!r}")
        try:
            parsed = json.loads(raw_body)
            return parsed["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise OpenRouterError(f"Unexpected OpenRouter response shape: {raw_body[:500]!r}") from exc
