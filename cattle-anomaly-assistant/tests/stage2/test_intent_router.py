from __future__ import annotations

from datetime import datetime, timezone

from shared.schemas import AnomalyExplanationQuery
from stage2_rag_assistant.config.settings import RouterConfig
from stage2_rag_assistant.pipeline.intent_router import classify_intent

CONFIG = RouterConfig(max_tokens=32, timeout_s=10.0)


def _query(raw_text: str | None) -> AnomalyExplanationQuery:
    return AnomalyExplanationQuery(
        query_id="q-1",
        cow_id="cow-001",
        raw_text=raw_text,
        submitted_by="farmer",
        timestamp=datetime.now(timezone.utc),
    )


def test_no_raw_text_routes_to_explain_anomaly_with_zero_llm_calls(fake_llm_client):
    client = fake_llm_client(responses=[])
    intent = classify_intent(_query(None), llm_client=client, config=CONFIG)
    assert intent == "explain_anomaly"
    assert client.call_log == []


def test_valid_label_is_returned(fake_llm_client):
    client = fake_llm_client(responses=['{"intent": "general_knowledge"}'])
    intent = classify_intent(_query("what causes mastitis in general?"), llm_client=client, config=CONFIG)
    assert intent == "general_knowledge"
    assert len(client.call_log) == 1


def test_garbage_output_maps_to_out_of_scope(fake_llm_client):
    client = fake_llm_client(responses=["not json at all"])
    intent = classify_intent(_query("some query"), llm_client=client, config=CONFIG)
    assert intent == "out_of_scope"


def test_off_list_label_maps_to_out_of_scope(fake_llm_client):
    client = fake_llm_client(responses=['{"intent": "diagnose_disease"}'])
    intent = classify_intent(_query("what disease does she have?"), llm_client=client, config=CONFIG)
    assert intent == "out_of_scope"


def test_router_call_uses_its_own_config_budget(fake_llm_client):
    client = fake_llm_client(responses=['{"intent": "explain_anomaly"}'])
    classify_intent(_query("why was she flagged?"), llm_client=client, config=CONFIG)
    assert client.call_log[0]["max_tokens"] == CONFIG.max_tokens
