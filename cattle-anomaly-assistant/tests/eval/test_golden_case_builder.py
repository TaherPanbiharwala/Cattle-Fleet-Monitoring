from __future__ import annotations

import yaml

from shared.schemas import GoldenCase
from stage2_rag_assistant.eval.build_mock_golden_cases import _kb_seed, build_cases
from stage2_rag_assistant.kb.build_kb import DEFAULT_SEED_PATH


def test_builds_the_expected_case_count():
    assert len(build_cases()) == 17


def test_every_case_round_trips_through_golden_case():
    for case in build_cases():
        GoldenCase.model_validate(case.model_dump())


def test_build_cases_is_deterministic():
    """No randomness at all by design (unlike the general-purpose mock
    generator) — a curated golden set should be exactly reproducible, not
    just seed-reproducible."""
    first = [c.model_dump_json() for c in build_cases()]
    second = [c.model_dump_json() for c in build_cases()]
    assert first == second


def test_gold_key_facts_match_the_kb_seed_yaml_verbatim():
    kb_seed = _kb_seed()
    by_category = {c["name"]: c for c in kb_seed["categories"]}

    for case in build_cases():
        if len(case.gold_driving_signals) != 1:
            continue  # multi-signal cases combine 2 categories, checked separately below
        category_name = case.case_id.rsplit("-", 1)[0].removeprefix("mock-")
        expected = [" ".join(link["summary_text"].split()) for link in by_category[category_name]["literature_links"]]
        assert case.gold_key_facts == expected


def test_multi_signal_cases_combine_facts_from_both_categories():
    cases_by_id = {c.case_id: c for c in build_cases()}
    combo_case = cases_by_id["mock-multi-signal-00"]
    assert len(combo_case.gold_driving_signals) == 2
    assert len(combo_case.gold_key_facts) == 2


def test_no_case_has_empty_driving_signals():
    """Cases with no driving signals would hit empty retrieval and never
    reach llm_grounded — useless for Layer 2 scoring (see module docstring)."""
    for case in build_cases():
        assert case.gold_driving_signals


def test_kb_seed_path_is_the_real_checked_in_file():
    assert _kb_seed(DEFAULT_SEED_PATH) == yaml.safe_load(DEFAULT_SEED_PATH.read_text())
