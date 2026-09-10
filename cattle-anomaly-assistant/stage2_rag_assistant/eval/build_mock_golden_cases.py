"""Builds ~17 mock-derived GoldenCases for M2b's Layer 2 eval set.

Unlike stage2_rag_assistant/mock/generate_mock_records.py (randomly samples
driving signals for general Stage-2 development), these cases construct
each AnomalyRecord's fields directly and deliberately, per PRD Section 11's
own framing for injected cases: "the injection is deliberate, the ground
truth is exact." gold_key_facts is cross-referenced straight from
kb/seed_data/shift_categories.yaml's literature_links, guaranteeing it
matches what retriever._load_category() will actually return, since both
read the same source file.

GoldenCase.source has no "mock" value (Literal["mmcows_real",
"db_cow_walking_real", "synthetic_injection", "synthetic_adversarial"]).
These map to category="injected"/source="synthetic_injection" — the
closest fit, since a deliberate known-magnitude deviation is the same idea
as a real-sequence injection, just without a real healthy sequence
underneath — with injection_type prefixed "mock_" so they're visually
distinguishable from a real-sequence injection once the PRD Section 14
Integration Point rebuilds this file from real Stage 1 output.

Cases with anomaly_flag=False / no driving signals are deliberately not
included: retriever.retrieve_facts() would return is_empty=True for them,
which routes to fallback_insufficient_data before generation ever runs —
useless for Layer 2 scoring, and Layer 1 false-positive-rate testing is
out of scope for M2b.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import yaml

from shared.schemas import AnomalyRecord, BehaviorStateDistribution24h, GoldenCase
from stage2_rag_assistant.kb.build_kb import DEFAULT_SEED_PATH

DEFAULT_OUTPUT_PATH = Path(__file__).parent / "golden" / "mock_derived_cases.jsonl"
_BASE_TIMESTAMP = datetime(2026, 6, 1, tzinfo=timezone.utc)


def _kb_seed(seed_path: Path = DEFAULT_SEED_PATH) -> dict:
    return yaml.safe_load(seed_path.read_text())


def _gold_key_facts_for(category_names: list[str], kb_seed: dict) -> list[str]:
    facts: list[str] = []
    for category in kb_seed["categories"]:
        if category["name"] in category_names:
            for link in category["literature_links"]:
                facts.append(" ".join(link["summary_text"].split()))
    return facts


def _base_record(cow_id: str, window_suffix: str) -> AnomalyRecord:
    """Baseline/normal values everywhere; each case overrides only the
    field(s) its category deliberately deviates."""
    return AnomalyRecord(
        cow_id=cow_id,
        timestamp=_BASE_TIMESTAMP,
        window_id=f"{cow_id}-{window_suffix}",
        behavior_state="grazing",
        behavior_state_confidence=0.9,
        behavior_state_distribution_24h=BehaviorStateDistribution24h(
            walking=0.15, grazing=0.55, resting=0.25, miscellaneous=0.05
        ),
        cbt_c=38.6,
        cbt_deviation_sigma=0.1,
        cbt_cusum_value=0.0,
        lying_time_pct_24h=40.0,
        lying_time_deviation_sigma=0.1,
        thi=65.0,
        activity_magnitude_deviation_sigma=0.1,
        herd_isolation_score=0.1,
        anomaly_flag=True,
        anomaly_score=0.8,
        driving_signals=[],
        recent_anomaly_history=[],
    )


def _make_case(
    case_id: str,
    record: AnomalyRecord,
    *,
    driving_signals: list[str],
    query_text: str | None,
    category_names: list[str],
    kb_seed: dict,
    injection_type: str,
) -> GoldenCase:
    record = record.model_copy(update={"driving_signals": driving_signals})
    return GoldenCase(
        case_id=case_id,
        category="injected",
        source="synthetic_injection",
        injection_type=injection_type,
        input_record=record,
        query_text=query_text,
        gold_anomaly_flag=True,
        gold_driving_signals=driving_signals,
        gold_key_facts=_gold_key_facts_for(category_names, kb_seed),
        reviewed_by="mechanical",
    )


def build_cases() -> list[GoldenCase]:
    kb_seed = _kb_seed()
    cases: list[GoldenCase] = []

    # sustained_grazing_reduction — 3 magnitudes of grazing-share drop from the
    # 0.55 baseline; the deficit shifts to resting (a sick cow grazing less
    # tends to rest more), keeping walking/miscellaneous at baseline.
    for i, grazing_share in enumerate([0.30, 0.20, 0.12]):
        record = _base_record(f"cow-grazing-{i}", f"w{i}").model_copy(
            update={
                "behavior_state_distribution_24h": BehaviorStateDistribution24h(
                    walking=0.15, grazing=grazing_share, resting=0.80 - grazing_share, miscellaneous=0.05
                )
            }
        )
        cases.append(
            _make_case(
                f"mock-sustained_grazing_reduction-{i:02d}",
                record,
                driving_signals=["behavior_state"],
                query_text=[None, "why was this cow flagged?", "what changed for this cow?"][i],
                category_names=["sustained_grazing_reduction"],
                kb_seed=kb_seed,
                injection_type="mock_behavior_state_deviation",
            )
        )

    # sustained_lying_increase — 3 sigma magnitudes above the +2.0 threshold.
    for i, sigma in enumerate([2.1, 2.8, 3.5]):
        record = _base_record(f"cow-lying-{i}", f"w{i}").model_copy(
            update={"lying_time_pct_24h": 40.0 + sigma * 5, "lying_time_deviation_sigma": sigma}
        )
        cases.append(
            _make_case(
                f"mock-sustained_lying_increase-{i:02d}",
                record,
                driving_signals=["lying_time"],
                query_text=[None, "why was this cow flagged?", "what changed for this cow?"][i],
                category_names=["sustained_lying_increase"],
                kb_seed=kb_seed,
                injection_type="mock_lying_time_deviation",
            )
        )

    # activity_magnitude_reduction — 3 sigma magnitudes below the -2.0 threshold.
    for i, sigma in enumerate([-2.1, -2.8, -3.5]):
        record = _base_record(f"cow-activity-{i}", f"w{i}").model_copy(
            update={"activity_magnitude_deviation_sigma": sigma}
        )
        cases.append(
            _make_case(
                f"mock-activity_magnitude_reduction-{i:02d}",
                record,
                driving_signals=["activity_magnitude"],
                query_text=[None, "why was this cow flagged?", "what changed for this cow?"][i],
                category_names=["activity_magnitude_reduction"],
                kb_seed=kb_seed,
                injection_type="mock_activity_magnitude_deviation",
            )
        )

    # herd_isolation_increase — 3 scores above the 0.7 threshold.
    for i, score in enumerate([0.75, 0.85, 0.95]):
        record = _base_record(f"cow-isolation-{i}", f"w{i}").model_copy(update={"herd_isolation_score": score})
        cases.append(
            _make_case(
                f"mock-herd_isolation_increase-{i:02d}",
                record,
                driving_signals=["herd_isolation"],
                query_text=[None, "why was this cow flagged?", "what changed for this cow?"][i],
                category_names=["herd_isolation_increase"],
                kb_seed=kb_seed,
                injection_type="mock_herd_isolation_deviation",
            )
        )

    # temperature_deviation_with_thi_context — the FR-10 worked example itself:
    # elevated cbt with elevated THI (environmental explanation available),
    # elevated cbt with normal THI (points toward the animal), and a second
    # elevated-THI case at a larger magnitude.
    for i, (cbt_sigma, thi) in enumerate([(2.5, 78.0), (2.5, 60.0), (3.2, 80.0)]):
        record = _base_record(f"cow-temp-{i}", f"w{i}").model_copy(
            update={"cbt_c": 38.6 + cbt_sigma * 0.3, "cbt_deviation_sigma": cbt_sigma, "thi": thi}
        )
        cases.append(
            _make_case(
                f"mock-temperature_deviation_with_thi_context-{i:02d}",
                record,
                driving_signals=["cbt"],
                query_text=[None, "why was this cow flagged?", "what changed for this cow?"][i],
                category_names=["temperature_deviation_with_thi_context"],
                kb_seed=kb_seed,
                injection_type="mock_cbt_deviation",
            )
        )

    # Multi-signal combinations — exercises the generator's FR-10 multi-signal
    # reasoning instruction (len(driving_signals) > 1).
    record = _base_record("cow-multi-0", "w0").model_copy(
        update={"cbt_c": 39.4, "cbt_deviation_sigma": 2.5, "thi": 78.0, "lying_time_pct_24h": 54, "lying_time_deviation_sigma": 2.8}
    )
    cases.append(
        _make_case(
            "mock-multi-signal-00",
            record,
            driving_signals=["cbt", "lying_time"],
            query_text=None,
            category_names=["temperature_deviation_with_thi_context", "sustained_lying_increase"],
            kb_seed=kb_seed,
            injection_type="mock_multi_signal",
        )
    )
    record = _base_record("cow-multi-1", "w0").model_copy(
        update={"lying_time_pct_24h": 54, "lying_time_deviation_sigma": 2.8, "herd_isolation_score": 0.85}
    )
    cases.append(
        _make_case(
            "mock-multi-signal-01",
            record,
            driving_signals=["lying_time", "herd_isolation"],
            query_text="what changed for this cow?",
            category_names=["sustained_lying_increase", "herd_isolation_increase"],
            kb_seed=kb_seed,
            injection_type="mock_multi_signal",
        )
    )

    return cases


def main() -> None:
    cases = build_cases()
    DEFAULT_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DEFAULT_OUTPUT_PATH.open("w") as f:
        for case in cases:
            f.write(case.model_dump_json() + "\n")
    print(f"Wrote {len(cases)} cases to {DEFAULT_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
