"""Scenario-grouped reporting for the fixed historical-application gate.

This runner records raw pre-gate Stage 2 responses, confidence buckets, and
an untouched scenario-held-out slice. The configured tau is the explicit
user-selected 0.70 operational demo policy; it is never reported as clinical
calibration or as CUSUM detection accuracy.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from stage2_rag_assistant.calibration.bucketing import (
    CalibrationCaseResult,
    aggregate_by_bucket,
    evaluate_case,
    split_calibration_test,
)
from stage2_rag_assistant.config.settings import Stage2Config, load_config
from stage2_rag_assistant.golden_loading import DEFAULT_GOLDEN_FILES, load_cases, query_for
from stage2_rag_assistant.llm.client import LLMClient, build_llm_client
from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

_HERE = Path(__file__).parent
DEFAULT_REPORTS_DIR = _HERE / "reports"

_HEADER = (
    "Stage 2 historical-application threshold report — scenario-grouped confidence and held-out workflow "
    "reporting for the fixed 0.70 operational demo gate. It is not a clinical calibration or detection-accuracy result."
)


def run(
    golden_files: list[Path] | None = None,
    *,
    pipeline_config: Stage2Config | None = None,
    llm_client: LLMClient | None = None,
    out_dir: Path = DEFAULT_REPORTS_DIR,
    write_report: bool = True,
) -> dict:
    """llm_client is normally built from pipeline_config.llm, but can be
    injected directly — this is what lets tests exercise multiple
    confidence buckets with a per-case-scripted FakeLLMClient, since
    build_llm_client()'s factory only ever returns one with the generic
    default responder.
    """
    golden_files = golden_files if golden_files is not None else DEFAULT_GOLDEN_FILES
    pipeline_config = pipeline_config or load_config()

    cases = load_cases(golden_files)
    calibration_cases, test_cases = split_calibration_test(cases)
    llm_client = llm_client or build_llm_client(pipeline_config.llm)

    def evaluate_split(cases_for_split: list, split_name: str) -> tuple[list[CalibrationCaseResult], dict[str, int], list[dict]]:
        results: list[CalibrationCaseResult] = []
        excluded_by_path: dict[str, int] = {}
        raw: list[dict] = []
        for case in cases_for_split:
            response = run_pipeline(case.input_record, query_for(case), config=pipeline_config, llm_client=llm_client)
            raw.append(
                {
                    "case_id": case.case_id,
                    "scenario_id": case.scenario_id or f"legacy:{case.case_id}",
                    "split": split_name,
                    "path_taken": response.path_taken,
                    "confidence": response.confidence,
                    "disagreement_flag": response.disagreement_flag,
                }
            )
            result = evaluate_case(case, response)
            if result is None:
                excluded_by_path[response.path_taken] = excluded_by_path.get(response.path_taken, 0) + 1
                continue
            results.append(result)
        return results, excluded_by_path, raw

    results, excluded_by_path, calibration_raw = evaluate_split(calibration_cases, "calibration")
    held_out_results, held_out_excluded, held_out_raw = evaluate_split(test_cases, "held_out")
    overall_correct = sum(1 for result in results if result.correct)
    held_out_correct = sum(1 for result in held_out_results if result.correct)
    is_demo_mode = pipeline_config.llm.provider == "fake"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": _HEADER,
        "scope_note": "The 0.70 gate is a fixed operational demonstration policy. Reports remain separate from Stage 1 detection correctness and clinical claims.",
        "demo_mode": is_demo_mode,
        "demo_mode_banner": "DEMO MODE: responses use the deterministic fake provider. This verifies split/report plumbing only." if is_demo_mode else None,
        "configured_tau": pipeline_config.fallback.tau,
        "tau_policy": "balanced operational demo gate; not clinical calibration",
        "total_golden_cases": len(cases),
        "calibration_split_size": len(calibration_cases),
        "held_out_split_size": len(test_cases),
        "calibration_scenario_ids": sorted({case.scenario_id or f"legacy:{case.case_id}" for case in calibration_cases}),
        "held_out_scenario_ids": sorted({case.scenario_id or f"legacy:{case.case_id}" for case in test_cases}),
        "excluded_by_path_taken": excluded_by_path,
        "held_out_excluded_by_path_taken": held_out_excluded,
        "bucket_stats": [bucket.model_dump() for bucket in aggregate_by_bucket(results)],
        "overall_llm_accuracy": {
            "count": len(results),
            "correct_count": overall_correct,
            "accuracy": (overall_correct / len(results)) if results else None,
        },
        "held_out_workflow_result": {
            "count": len(held_out_results),
            "correct_count": held_out_correct,
            "accuracy": (held_out_correct / len(held_out_results)) if held_out_results else None,
        },
        "raw_pre_gate": [*calibration_raw, *held_out_raw],
        "per_case": [result.model_dump() for result in results],
        "held_out_per_case": [result.model_dump() for result in held_out_results],
    }

    if write_report:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = out_dir / f"calibration_report_{stamp}.json"
        md_path = out_dir / f"calibration_report_{stamp}.md"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        md_path.write_text(_render_markdown(report))
        report["_json_path"] = str(json_path)
        report["_md_path"] = str(md_path)

    return report


def _render_markdown(report: dict) -> str:
    lines = [f"# {report['header']}", "", f"**{report['scope_note']}**", ""]
    if report["demo_mode_banner"]:
        lines += [f"> **{report['demo_mode_banner']}**", ""]
    lines += [
        f"Generated: {report['generated_at']}",
        f"Configured operational tau: {report['configured_tau']}",
        f"Total golden cases: {report['total_golden_cases']}",
        f"Calibration split: {report['calibration_split_size']} | Scenario-held-out split: "
        f"{report['held_out_split_size']}",
        f"Excluded from calibration (non-llm_grounded): {report['excluded_by_path_taken']}",
        "",
        "## Empirical workflow agreement per confidence bucket",
        "",
        "| Bucket | Count | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for bucket in report["bucket_stats"]:
        accuracy = f"{bucket['accuracy']:.3f}" if bucket["accuracy"] is not None else "n/a"
        lines.append(f"| {bucket['bucket']} | {bucket['count']} | {bucket['correct_count']} | {accuracy} |")
    overall = report["overall_llm_accuracy"]
    overall_acc = f"{overall['accuracy']:.3f}" if overall["accuracy"] is not None else "n/a"
    held = report["held_out_workflow_result"]
    held_acc = f"{held['accuracy']:.3f}" if held["accuracy"] is not None else "n/a"
    lines += ["", f"Calibration workflow agreement: {overall_acc} ({overall['correct_count']}/{overall['count']})"]
    lines += [f"Scenario-held-out workflow agreement: {held_acc} ({held['correct_count']}/{held['count']})"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report scenario-grouped workflow agreement for the configured historical-app tau.")
    parser.add_argument("--golden-file", action="append", dest="golden_files", type=Path, default=None)
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    pipeline_config = load_config(args.pipeline_config) if args.pipeline_config else load_config()

    report = run(args.golden_files, pipeline_config=pipeline_config, out_dir=args.out_dir)
    print(f"Wrote {report['_json_path']} and {report['_md_path']}")
    print(
        f"Calibration split: {report['calibration_split_size']} cases, "
        f"overall LLM accuracy: {report['overall_llm_accuracy']['accuracy']}"
    )


if __name__ == "__main__":
    main()
