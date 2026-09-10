"""Import this module (for its side effect) before importing anything from
`ragas`, anywhere in this package.

ragas==0.4.3's legacy `ragas/llms/base.py` unconditionally does
`from langchain_community.chat_models.vertexai import ChatVertexAI` at
module load time, purely to support its deprecated `LangchainLLMWrapper`
class. That submodule no longer exists in `langchain-community==0.4.2`
(langchain-community is being sunset and split into standalone integration
packages; the Vertex AI shim was one of the casualties) — installing
`langchain-google-vertexai` does not fix it, since ragas imports the old
`langchain_community` path specifically, not the new standalone package.
Downgrading `langchain-community` cascades into an incompatible
`langchain-core`, which cascades into more broken imports across
`langchain-openai`/`langgraph` — verified by actually trying it. This is a
genuine upstream packaging bug in ragas 0.4.3, not a version-pinning
problem we can solve by choosing different pins.

We only ever use the modern `InstructorBaseRagasLLM`/`BaseRagasEmbedding`
API (never the legacy `LangchainLLMWrapper`/`VertexAI` classes this broken
import exists for), so the fix is to stub the missing submodule before
`ragas` is imported — this satisfies the import statement without ever
exercising the code that would actually need a real Vertex AI SDK.

Re-check this on any ragas upgrade: if a future release fixes the import
(e.g. makes it lazy/optional), this stub becomes a no-op and can be
deleted; if ragas changes the broken import's exact path, update the stub
below to match.
"""

from __future__ import annotations

import sys
import types

if "langchain_community.chat_models.vertexai" not in sys.modules:
    _fake_vertexai_chat_models = types.ModuleType("langchain_community.chat_models.vertexai")
    _fake_vertexai_chat_models.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[attr-defined]
    sys.modules["langchain_community.chat_models.vertexai"] = _fake_vertexai_chat_models
