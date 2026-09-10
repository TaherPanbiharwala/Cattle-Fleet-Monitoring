from __future__ import annotations

from stage2_rag_assistant.pipeline.retriever import retrieve_facts


def test_single_signal_lookup_matches_its_category(tmp_kb_db):
    result = retrieve_facts(["lying_time"], None, db_path=tmp_kb_db)
    assert not result.is_empty
    names = [c.name for c in result.matched_categories]
    assert names == ["sustained_lying_increase"]


def test_multi_signal_lookup_matches_multiple_categories(tmp_kb_db):
    result = retrieve_facts(["lying_time", "herd_isolation"], None, db_path=tmp_kb_db)
    names = {c.name for c in result.matched_categories}
    assert names == {"sustained_lying_increase", "herd_isolation_increase"}


def test_unmapped_signal_returns_empty(tmp_kb_db):
    result = retrieve_facts(["totally_unknown_signal"], None, db_path=tmp_kb_db)
    assert result.is_empty
    assert result.matched_categories == []


def test_cbt_lookup_pulls_back_thi_context_via_shared_category(tmp_kb_db):
    """The FR-10 mechanism: matching on 'cbt' alone must still return the
    'thi_context' row, since thi is never itself a driving signal."""
    result = retrieve_facts(["cbt"], None, db_path=tmp_kb_db)
    assert len(result.matched_categories) == 1
    category = result.matched_categories[0]
    signal_names = {t.signal_name for t in category.thresholds}
    assert signal_names == {"cbt", "thi_context"}


def test_every_cited_fact_has_a_source(tmp_kb_db):
    result = retrieve_facts(["cbt", "lying_time", "activity_magnitude", "herd_isolation"], None, db_path=tmp_kb_db)
    all_facts = [fact for category in result.matched_categories for fact in category.cited_facts]
    assert all_facts
    assert all(fact.source.startswith("literature_links:") for fact in all_facts)


def test_free_text_keyword_matching_is_deterministic_substring_not_embedding(tmp_kb_db):
    result = retrieve_facts([], "she seems isolated from the rest of the herd", db_path=tmp_kb_db)
    names = [c.name for c in result.matched_categories]
    assert names == ["herd_isolation_increase"]


def test_short_keyword_does_not_match_inside_unrelated_words(tmp_kb_db):
    """'thi' is a real keyword (temperature_deviation_with_thi_context) short
    enough to appear as a raw substring inside common words like 'this' and
    'think' — word-boundary matching must not treat those as a match."""
    result = retrieve_facts([], "I think this cow looks fine to me", db_path=tmp_kb_db)
    assert result.is_empty


def test_retrieval_is_deterministic_across_calls(tmp_kb_db):
    first = retrieve_facts(["cbt", "lying_time"], None, db_path=tmp_kb_db)
    second = retrieve_facts(["cbt", "lying_time"], None, db_path=tmp_kb_db)
    assert first.model_dump() == second.model_dump()
