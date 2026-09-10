"""CLI + importable run() for PRD Section 10's calibration steps 1-3 only.

Full calibration is 6 steps: (1) hold out a calibration split from the
golden set, separate from a test split; (2) run the pipeline, bucket
responses by stated confidence; (3) compute empirical accuracy per
bucket; (4) compute Stage 1's own accuracy on the identical slice; (5)
set tau to the lowest bucket where LLM accuracy >= Stage 1's; (6)
re-validate on the test split. Steps 4-6 need real Stage 1 output that
doesn't exist yet (Stage 1 is still with its collaborators) — this module
implements 1-3 only, prototyped on the mock golden set.

Deliberately does NOT compute step 4, even though "Stage 1's own
accuracy" is technically possible on the current mock golden set: every
mock-derived case was built with input_record.anomaly_flag already
matching gold_anomaly_flag, so that number would trivially compute to
~100% — a real-looking but meaningless figure. PRD's own words: "Stage
1's accuracy on mocked data is meaningless." This module never reports
or even computes anything shaped like a Stage-1-accuracy figure.

Never writes to config/default.yaml's real tau — reads it read-only, for
context in the report.
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
    "Stage 2 tau-Calibration — PRD Section 10 steps 1-3 ONLY (hold out a calibration split, bucket "
    "responses by stated confidence, compute the LLM's own empirical accuracy per bucket). Steps 4-6 "
    "(Stage 1's own accuracy on the identical slice, setting tau, re-validating on the test split) need "
    "real Stage 1 output and are separate, later scope (PRD Section 14 Integration Point). Nothing in "
    "this report is a calibrated tau — config/default.yaml's tau is untouched."
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

    results: list[CalibrationCaseResult] = []
    excluded_by_path: dict[str, int] = {}

    for case in calibration_cases:
        response = run_pipeline(case.input_record, query_for(case), config=pipeline_config, llm_client=llm_client)
        result = evaluate_case(case, response)
        if result is None:
            excluded_by_path[response.path_taken] = excluded_by_path.get(response.path_taken, 0) + 1
            continue
        results.append(result)

    overall_correct = sum(1 for r in results if r.correct)
    is_demo_mode = pipeline_config.llm.provider == "fake"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": _HEADER,
        "scope_note": "Steps 1-3 only. Steps 4-6 need real Stage 1 output — see this module's docstring.",
        "demo_mode": is_demo_mode,
        "demo_mode_banner": (
            "DEMO MODE: ran against a fake LLM (no real provider chosen yet, PRD Open Question 2). Its "
            "default responder returns a fixed 0.75 confidence for every case, so this run will show "
            "exactly one populated bucket — that's expected, not a bug in the bucketing mechanism. "
            "Re-run once a real LLM provider lands to see a real confidence distribution."
        )
        if is_demo_mode
        else None,
        "current_config_tau": pipeline_config.fallback.tau,
        "total_golden_cases": len(cases),
        "calibration_split_size": len(calibration_cases),
        "test_split_size": len(test_cases),
        "test_split_case_ids": [c.case_id for c in test_cases],
        "excluded_by_path_taken": excluded_by_path,
        "bucket_stats": [b.model_dump() for b in aggregate_by_bucket(results)],
        "overall_llm_accuracy": {
            "count": len(results),
            "correct_count": overall_correct,
            "accuracy": (overall_correct / len(results)) if results else None,
        },
        "per_case": [r.model_dump() for r in results],
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
        f"Current config tau (unchanged): {report['current_config_tau']}",
        f"Total golden cases: {report['total_golden_cases']}",
        f"Calibration split: {report['calibration_split_size']} | Test split (held out, not run): "
        f"{report['test_split_size']}",
        f"Excluded from calibration (non-llm_grounded): {report['excluded_by_path_taken']}",
        "",
        "## Empirical accuracy per confidence bucket (LLM's own judgment only — not Stage 1's)",
        "",
        "| Bucket | Count | Correct | Accuracy |",
        "|---|---|---|---|",
    ]
    for bucket in report["bucket_stats"]:
        accuracy = f"{bucket['accuracy']:.3f}" if bucket["accuracy"] is not None else "n/a"
        lines.append(f"| {bucket['bucket']} | {bucket['count']} | {bucket['correct_count']} | {accuracy} |")
    overall = report["overall_llm_accuracy"]
    overall_acc = f"{overall['accuracy']:.3f}" if overall["accuracy"] is not None else "n/a"
    lines += ["", f"Overall LLM accuracy (all buckets combined): {overall_acc} ({overall['correct_count']}/{overall['count']})"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prototype PRD Section 10's calibration steps 1-3.")
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
