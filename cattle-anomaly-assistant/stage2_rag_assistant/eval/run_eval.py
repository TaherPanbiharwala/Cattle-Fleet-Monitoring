"""CLI + importable run() for the Layer 2 (Ragas) eval harness.
See LLM_Diagnostic_Assistant_PRD.md Section 11.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from stage2_rag_assistant.eval import _ragas_compat  # noqa: F401  (import for its side effect, before ragas)
from ragas.metrics.collections import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness

from stage2_rag_assistant.config.settings import Stage2Config, load_config
from stage2_rag_assistant.eval.eval_config import EvalConfig, load_eval_config
from stage2_rag_assistant.eval.metrics import CaseScore, aggregate, score_case
from stage2_rag_assistant.eval.ragas_embeddings import build_ragas_embedding
from stage2_rag_assistant.eval.ragas_llm import build_ragas_llm
from stage2_rag_assistant.golden_loading import DEFAULT_GOLDEN_FILES, load_cases, query_for
from stage2_rag_assistant.llm.client import build_llm_client
from stage2_rag_assistant.pipeline.orchestrator import run_pipeline

_HERE = Path(__file__).parent
DEFAULT_REPORTS_DIR = _HERE / "reports"

_LAYER2_HEADER = (
    "Stage 2 RAG Evaluation — Layer 2 (Ragas generation quality) ONLY. "
    "Layer 1 (detection/explanation accuracy: flag/driving-signal correctness, false-positive rate, "
    "the G4 ablation, hallucination/calibration) is separate, later scope (PRD Section 11, Section 14 "
    "'Final phase'). Nothing in this report is a detection-accuracy result."
)


async def run(
    golden_files: list[Path] | None = None,
    *,
    pipeline_config: Stage2Config | None = None,
    eval_config: EvalConfig | None = None,
    out_dir: Path = DEFAULT_REPORTS_DIR,
    write_report: bool = True,
) -> dict:
    """In demo mode (the default, no real provider chosen — PRD Open
    Question 2), the un-scripted FakeLLMClient's default responder always
    answers "explain_anomaly" for any router call, since it has no real
    intent-classification ability. This means the two out-of-scope
    adversarial cases WILL still get scored here (verified: they do), even
    though tests/eval/test_adversarial_routing.py separately proves the
    routing logic itself is correct, by scripting each case's router
    response explicitly. Not a bug in this function — a real limitation of
    running Layer 2 without a real judge, exactly what the demo-mode
    banner below exists to make visible.
    """
    golden_files = golden_files if golden_files is not None else DEFAULT_GOLDEN_FILES
    pipeline_config = pipeline_config or load_config()
    eval_config = eval_config or load_eval_config()

    cases = load_cases(golden_files)
    llm_client = build_llm_client(pipeline_config.llm)

    judge_llm = build_ragas_llm(eval_config.judge_llm)
    judge_embedding = build_ragas_embedding(eval_config.judge_embedding)
    faithfulness = Faithfulness(llm=judge_llm)
    answer_relevancy = AnswerRelevancy(llm=judge_llm, embeddings=judge_embedding)
    context_precision = ContextPrecision(llm=judge_llm)
    context_recall = ContextRecall(llm=judge_llm)

    per_case: list[dict] = []
    scored: list[CaseScore] = []
    excluded_by_path: dict[str, int] = {}

    for case in cases:
        response = run_pipeline(case.input_record, query_for(case), config=pipeline_config, llm_client=llm_client)

        if response.path_taken != "llm_grounded":
            # Fallback responses' rationale is a hardcoded template, never LLM
            # output — scoring faithfulness against a template that can't
            # hallucinate would silently pollute the aggregate with a
            # meaningless "perfect" score. Counted, never dropped silently.
            excluded_by_path[response.path_taken] = excluded_by_path.get(response.path_taken, 0) + 1
            per_case.append({"case_id": case.case_id, "path_taken": response.path_taken, "scored": False})
            continue

        case_score = await score_case(
            case,
            response,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
        )
        scored.append(case_score)
        per_case.append(
            {
                "case_id": case.case_id,
                "path_taken": response.path_taken,
                "scored": True,
                "scores": case_score.model_dump(),
            }
        )

    is_demo_mode = pipeline_config.llm.provider == "fake"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "header": _LAYER2_HEADER,
        "demo_mode": is_demo_mode,
        "demo_mode_banner": (
            "DEMO MODE: ran against a fake LLM (no real provider chosen yet, PRD Open Question 2). These "
            "scores are plumbing-verification only, not a real quality measurement. Re-run once a real "
            "LLM provider lands."
        )
        if is_demo_mode
        else None,
        "total_cases": len(cases),
        "excluded_by_path_taken": excluded_by_path,
        "aggregates": aggregate(scored),
        "per_case": per_case,
    }

    if write_report:
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = out_dir / f"layer2_report_{stamp}.json"
        md_path = out_dir / f"layer2_report_{stamp}.md"
        json_path.write_text(json.dumps(report, indent=2, default=str))
        md_path.write_text(_render_markdown(report))
        report["_json_path"] = str(json_path)
        report["_md_path"] = str(md_path)

    return report


def _render_markdown(report: dict) -> str:
    lines = [f"# {report['header']}", ""]
    if report["demo_mode_banner"]:
        lines += [f"> **{report['demo_mode_banner']}**", ""]
    lines += [
        f"Generated: {report['generated_at']}",
        f"Total cases: {report['total_cases']}",
        f"Excluded (non-llm_grounded): {report['excluded_by_path_taken']}",
        "",
        "## Aggregate scores (Layer 2)",
        "",
        "| Metric | Mean | Median | Count | Errors |",
        "|---|---|---|---|---|",
    ]
    for metric in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        agg = report["aggregates"][metric]
        errors = report["aggregates"]["errors_by_metric"].get(metric, 0)
        mean = f"{agg['mean']:.3f}" if agg["mean"] is not None else "n/a"
        median = f"{agg['median']:.3f}" if agg["median"] is not None else "n/a"
        lines.append(f"| {metric} | {mean} | {median} | {agg['count']} | {errors} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Stage 2 Layer 2 (Ragas) evaluation.")
    parser.add_argument("--golden-file", action="append", dest="golden_files", type=Path, default=None)
    parser.add_argument("--pipeline-config", type=Path, default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    args = parser.parse_args()

    pipeline_config = load_config(args.pipeline_config) if args.pipeline_config else load_config()
    eval_config = load_eval_config(args.eval_config) if args.eval_config else load_eval_config()

    report = asyncio.run(
        run(args.golden_files, pipeline_config=pipeline_config, eval_config=eval_config, out_dir=args.out_dir)
    )
    print(f"Wrote {report['_json_path']} and {report['_md_path']}")
    print(f"Scored {report['aggregates']['total_cases_scored']} of {report['total_cases']} cases")


if __name__ == "__main__":
    main()
