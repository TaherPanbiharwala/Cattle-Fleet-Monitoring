from __future__ import annotations

from pathlib import Path

import pytest

from stage2_rag_assistant.kb.build_kb import DEFAULT_SCHEMA_PATH, DEFAULT_SEED_PATH, build_kb
from stage2_rag_assistant.llm.providers.fake_provider import FakeLLMClient


@pytest.fixture
def tmp_kb_db(tmp_path: Path) -> Path:
    """A throwaway KB built from the real, checked-in schema.sql + seed YAML
    (never the packaged kb.sqlite3, which is a gitignored build artifact)."""
    db_path = tmp_path / "kb.sqlite3"
    build_kb(schema_path=DEFAULT_SCHEMA_PATH, seed_path=DEFAULT_SEED_PATH, db_path=db_path)
    return db_path


@pytest.fixture
def fake_llm_client():
    def _make(responses: list[str] | None = None, responder=None) -> FakeLLMClient:
        return FakeLLMClient(responses=responses, responder=responder)

    return _make
