"""Build the 55-case historical application evaluation corpus from derived data.

The command consumes only Stage 1 JSON outputs and the already-derived WASP
summary.  It never reads raw sensor rows and never presents the real MmCows
workflow cases as clinical ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml

from shared.schemas import GoldenCase
from stage1_anomaly_detection.behavior_classifier.historical_demo import historical_demo_anomaly_records
from stage1_anomaly_detection.behavior_classifier.errors import BehaviorError
from stage1_anomaly_detection.behavior_classifier.io import atomic_output_dir, read_jsonl, sha256_file, write_jsonl
from stage2_rag_assistant.historical_application import (
    DEFAULT_DEPLOYMENT_ID,
    DEFAULT_TIMEZONE,
    RECORD_COUNT,
    select_historical_cusum_windows,
)
from stage2_rag_assistant.kb.build_kb import DEFAULT_SEED_PATH

ADVERSARIAL_PATH = Path(__file__).parent / "golden" / "adversarial_cases.jsonl"


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _load_json(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{code}: invalid JSON at {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{code}: expected object at {path}")
    return value


def _facts_for(signals: list[str]) -> list[str]:
    seed = yaml.safe_load(DEFAULT_SEED_PATH.read_text(encoding="utf-8"))
    facts: list[str] = []
    for category in seed["categories"]:
        threshold_signals = {row["signal_name"] for row in category["thresholds"]}
        if threshold_signals.intersection(signals):
            facts.extend(" ".join(link["summary_text"].split()) for link in category["literature_links"])
    return facts


def _scenario_days(injection_config: Path) -> dict[tuple[str, str], str]:
    payload = _load_json(injection_config, code="INJECTION_PROVENANCE_INVALID")
    scenarios = payload.get("scenarios")
    if payload.get("schema_version") != 1 or not isinstance(scenarios, list):
        raise ValueError("INJECTION_PROVENANCE_INVALID: expected schema_version 1 scenarios")
    mapping: dict[tuple[str, str], str] = {}
    for scenario in scenarios:
        if not isinstance(scenario, dict):
            raise ValueError("INJECTION_PROVENANCE_INVALID: scenario is not an object")
        scenario_id = scenario.get("scenario_id")
        cow_id = scenario.get("cow_id")
        start_date = scenario.get("start_date")
        duration = scenario.get("duration_days")
        if not isinstance(scenario_id, str) or not isinstance(cow_id, str) or not isinstance(start_date, str) or not isinstance(duration, int):
            raise ValueError("INJECTION_PROVENANCE_INVALID: scenario has invalid identity/date/duration")
        start = date.fromisoformat(start_date)
        for day_offset in range(duration):
            key = (cow_id, (start + timedelta(days=day_offset)).isoformat())
            if key in mapping:
                raise ValueError("INJECTION_PROVENANCE_INVALID: overlapping injection scenario days")
            mapping[key] = scenario_id
    return mapping


def _date_for_row(row: dict[str, Any]) -> str:
    value = row.get("date")
    if isinstance(value, str):
        return value
    timestamp = row.get("timestamp")
    if isinstance(timestamp, str):
        return timestamp[:10]
    raise ValueError("A CUSUM row has no date or timestamp")


def _make_cases(
    records: list[dict[str, Any]],
    *,
    category: str,
    source: str,
    detector_run_sha256: str,
    model_sha256: str,
    source_config_sha256: str,
    scenario_for_row: dict[tuple[str, str], str] | None = None,
) -> list[GoldenCase]:
    output: list[GoldenCase] = []
    for index, record in enumerate(records, start=1):
        signals = list(record["driving_signals"])
        scenario_id = scenario_for_row.get((record["cow_id"], _date_for_row(record))) if scenario_for_row else None
        if scenario_id is None:
            scenario_id = f"real-workflow:{record['window_id']}"
        output.append(
            GoldenCase(
                case_id=f"{category}-{index:02d}-{record['window_id'].replace(':', '-')}",
                category=category,
                source=source,
                scenario_id=scenario_id,
                detector_run_sha256=detector_run_sha256,
                behavior_model_sha256=model_sha256,
                source_config_sha256=source_config_sha256,
                injection_type=scenario_id if category == "injected" else None,
                input_record=record,
                query_text=None,
                gold_anomaly_flag=record["anomaly_flag"],
                gold_driving_signals=signals,
                gold_key_facts=_facts_for(signals),
                reviewed_by="mechanical_public_data",
            )
        )
    return output


def build_cases(
    *,
    real_cusum_rows: list[dict[str, Any]],
    injected_cusum_rows: list[dict[str, Any]],
    behavior_summary: dict[str, Any],
    deployment_id: str,
    timezone_name: str,
    injection_config: Path,
    real_detector_hash: str,
    injected_detector_hash: str,
) -> tuple[list[GoldenCase], list[GoldenCase], list[GoldenCase]]:
    real_selected = select_historical_cusum_windows(real_cusum_rows)
    injected_scenarios = _scenario_days(injection_config)
    injected_selected = sorted(
        (
            row
            for row in injected_cusum_rows
            if (str(row.get("cow_id")), _date_for_row(row)) in injected_scenarios
            and row.get("detector_state") == "monitoring"
        ),
        key=lambda row: (str(row["cow_id"]), _date_for_row(row)),
    )
    if len(injected_selected) != RECORD_COUNT:
        raise ValueError(f"INJECTION_SELECTION_INVALID: expected 24 configured injection/control windows, found {len(injected_selected)}")
    real_records = historical_demo_anomaly_records(real_selected, deployment_id=deployment_id, timezone=timezone_name, behavior_summary=behavior_summary)
    injected_records = historical_demo_anomaly_records(injected_selected, deployment_id=deployment_id, timezone=timezone_name, behavior_summary=behavior_summary)
    model_hash = behavior_summary["model_sha256"]
    config_hash = _stable_hash({"timezone": timezone_name, "deployment_id": deployment_id, "injection_config_sha256": sha256_file(injection_config)})
    real_cases = _make_cases(
        real_records,
        category="core",
        source="mmcows_real",
        detector_run_sha256=real_detector_hash,
        model_sha256=model_hash,
        source_config_sha256=config_hash,
    )
    injected_cases = _make_cases(
        injected_records,
        category="injected",
        source="synthetic_injection",
        detector_run_sha256=injected_detector_hash,
        model_sha256=model_hash,
        source_config_sha256=config_hash,
        scenario_for_row=injected_scenarios,
    )
    # Adversarial fixtures have no detector run by design. Their immutable
    # provenance instead identifies the checked-in synthetic fixture exactly;
    # it must never be mistaken for a CUSUM run hash.
    adversarial_fixture_hash = sha256_file(ADVERSARIAL_PATH)
    adversarial_cases = []
    for raw_case in read_jsonl(ADVERSARIAL_PATH):
        base_case = GoldenCase.model_validate(raw_case)
        adversarial_cases.append(
            base_case.model_copy(
                update={
                    "scenario_id": f"adversarial:{base_case.case_id}",
                    "detector_run_sha256": _stable_hash(
                        {"kind": "synthetic_adversarial_fixture", "fixture_sha256": adversarial_fixture_hash}
                    ),
                    "behavior_model_sha256": model_hash,
                    "source_config_sha256": _stable_hash(
                        {
                            "kind": "synthetic_adversarial_fixture",
                            "fixture_sha256": adversarial_fixture_hash,
                            "historical_config_sha256": config_hash,
                        }
                    ),
                }
            )
        )
    if len(real_cases) != 24 or len(injected_cases) != 24 or len(adversarial_cases) != 7:
        raise AssertionError("Historical golden-case corpus count invariant failed")
    return real_cases, injected_cases, adversarial_cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build derived 24 real + 24 injected + 7 adversarial historical evaluation cases.")
    parser.add_argument("--real-cusum-windows-jsonl", required=True, type=Path)
    parser.add_argument("--injected-cusum-windows-jsonl", required=True, type=Path)
    parser.add_argument("--historical-behavior-summary", required=True, type=Path)
    parser.add_argument("--injection-config", required=True, type=Path)
    parser.add_argument("--deployment-id", default=DEFAULT_DEPLOYMENT_ID)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        real_rows = read_jsonl(args.real_cusum_windows_jsonl)
        injected_rows = read_jsonl(args.injected_cusum_windows_jsonl)
        summary = _load_json(args.historical_behavior_summary, code="HISTORICAL_CONTEXT_INVALID")
        real_cases, injected_cases, adversarial_cases = build_cases(
            real_cusum_rows=real_rows,
            injected_cusum_rows=injected_rows,
            behavior_summary=summary,
            deployment_id=args.deployment_id,
            timezone_name=args.timezone,
            injection_config=args.injection_config,
            real_detector_hash=sha256_file(args.real_cusum_windows_jsonl),
            injected_detector_hash=sha256_file(args.injected_cusum_windows_jsonl),
        )

        def writer(temp_dir: Path) -> None:
            write_jsonl(temp_dir / "real_cases.jsonl", [case.model_dump(mode="json") for case in real_cases])
            write_jsonl(temp_dir / "injected_cases.jsonl", [case.model_dump(mode="json") for case in injected_cases])
            write_jsonl(temp_dir / "adversarial_cases.jsonl", [case.model_dump(mode="json") for case in adversarial_cases])

        published = atomic_output_dir(args.output_dir, writer)
        print(json.dumps({"status": "complete", "output_dir": str(published), "real_cases": 24, "injected_cases": 24, "adversarial_cases": 7}, sort_keys=True))
        return 0
    except (ValueError, KeyError, BehaviorError) as exc:
        code = "GOLDEN_BUILD_INVALID" if isinstance(exc, ValueError) else exc.code
        message = str(exc) if isinstance(exc, ValueError) else exc.message
        details: dict[str, Any] = {} if isinstance(exc, ValueError) else exc.details
        print(json.dumps({"code": code, "message": message, "details": details}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
