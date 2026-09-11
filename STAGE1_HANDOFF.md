# Stage 1 Behavior Classifier and Safe Fusion — Handoff

**For:** whoever's picking up the WASP-lab (`db-cow-walking`) behavior classification work.
**Your job, in one sentence:** maintain a public-data XGBoost behavior benchmark and the fail-closed contract that can later fuse it with same-cow runtime CUSUM context.
**Last updated:** 2026-09-11

---

## 1. The 60-second version of what this project is

This repo is building a system that detects when a dairy cow's behavior/physiology has shifted from her own normal, and explains that shift in plain language. It's split into two independent halves:

- **Stage 1 (this is you)** — classical ML, no LLM anywhere. Two parts: (a) a behavior classifier trained on public IMU data — **your task** — and (b) a per-cow statistical baseline (SPC/CUSUM) over a different dataset, MmCows — someone else's task, not yours, don't worry about it; see [`STAGE1_ANOMALY_BASELINE_HANDOFF.md`](STAGE1_ANOMALY_BASELINE_HANDOFF.md) if you're curious what that half involves. Together they produce a structured record saying "this cow's behavior/physiology looks different, and here's specifically what changed."
- **Stage 2 (already built)** — an LLM layer that takes Stage 1's output and explains it in cited, farmer-readable language, with a hard-grounded fallback so it never invents facts. This is done and tested (against fake/mocked Stage 1 data) — you don't need to touch it, and you don't need to understand it deeply. It's just the reason your output needs to match a specific shape (Section 4 below).

The full spec is [`LLM_Diagnostic_Assistant_PRD.md`](LLM_Diagnostic_Assistant_PRD.md) and the living project tracker is [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md). You don't need to read either cover to cover — pointers below tell you exactly which sections matter for your part.

## 2. What now exists

The maintained implementation lives entirely below `cattle-anomaly-assistant/stage1_anomaly_detection/behavior_classifier/`:

- `wasp.py` — strict source reader for `Time` plus six MPU9250 columns, four safe labels, and hash-only provenance.
- `features.py` — 50-sample / 10 Hz windows with 25-sample stride and exactly 112 features.
- `benchmark.py` — CPU XGBoost and nested LOCO evaluation.
- `artifact.py` — native JSON-only save/load and manifest verification; no Python pickle support.
- `context.py` and `runtime.py` — derived daily context and same-cow fusion contract.
- `historical_demo.py` — public-dataset application bridge using the active Kaggle 25-channel XGBoost JSON model and independently-derived MmCows CUSUM indicators.

The older `src/dataset_adapters/wasp_lab.py` and `src/ml/*` remain historical references only. Do not import them, modify them, or copy their unsafe pickle serialization into this subproject.

## 3. The actual target

- **Dataset:** `db-cow-walking` (Morales-Vargas et al. 2025) — 441 labeled IMU events from 10 cows. Get it from [github.com/WASP-lab/db-cow-walking](https://github.com/WASP-lab/db-cow-walking), or follow [`PHASE3_BENCHMARK.md`](PHASE3_BENCHMARK.md)'s Kaggle setup (a private Kaggle dataset + a notebook, `notebooks/phase3_wasp_benchmark.ipynb` — built for the old `src/ml/` location, you'll want your own copy/adaptation once your code lives in the new folder).
- **The evidence:** nested leave-one-cow-out is the primary and only eligibility protocol. The published **0.9625** result used a non-comparable random split and is a reference only, not a target or gate. The provisional benchmark-only gate is mean LOCO macro F1 ≥0.85 plus pooled Walking/Miscellaneous recall ≥0.75. A pass is never field or health validation.
- **Classes:** walking, grazing, resting, miscellaneous/other. **Never map miscellaneous to "restless"** — a hard rule everywhere in this codebase (`AGENTS.md` golden rule 6), because "restless" means something specific and safety-relevant elsewhere in this system.
- **Validation:** nested leave-one-cow-out, always. Each outer cow is excluded from both fitting and tuning. Never split windows from the same cow across a train/test boundary.
- The PRD (`LLM_Diagnostic_Assistant_PRD.md` Section 2) references a file called `behavior-classification-beat-baseline-plan.md` for more detail on this target — that file isn't in this git repo (it's referenced as living in the project owner's separate Claude Projects space per the PRD's own reference list, Section 17). If you need more detail than what's here and in `PHASE3_BENCHMARK.md`, ask for it rather than assuming it doesn't exist.

## 4. Output and fusion boundary

The model emits derived per-window state/confidence and daily contexts with the following fields required by the shared schema:

- `behavior_state`: one of `walking | grazing | resting | miscellaneous`
- `behavior_state_confidence`: float in [0,1]
- `behavior_state_distribution_24h`: a rolling 24-hour distribution across the 4 classes — **must sum to ~1.0** (this is schema-enforced downstream, not just a suggestion)

`behavior_state_confidence` is maximum softprob and therefore uncalibrated. The behavior state is contextual in M1d: it cannot change the existing CUSUM anomaly score or become a driver yet.

WASP and MmCows cannot be joined, even though both are public cattle datasets. The fusion contract requires an exact `(cow_id, local_date, timezone, deployment_id)` match and `same_cow_runtime` provenance from both sources. It rejects public `wasp_public` / `mmcows_public` inputs before writing output. Test fixtures exercise the schema path; they are not a field-ready data source.

For the public-data-only application mode, use `historical-demo-records` instead of `fuse-daily`. It runs the active Kaggle 25-channel XGBoost model over the staged WASP data and attaches its **dataset-level historical behavior summary** to flagged MmCows CUSUM records. The resulting `AnomalyRecord`s explicitly declare `behavior_context_source="wasp_public_historical_dataset"` and `behavior_context_relation="cross_dataset_historical_demo"`; they do not claim the behavior belongs to the MmCows cow/date. CUSUM remains the only source of the anomaly flag, score, and driving signals. See the assistant README for the command.

## 5. Commands and verification

```bash
cd cattle-anomaly-assistant
python3.12 -m venv .venv
.venv/bin/python -m pip install -e '.[behavior,dev]'
.venv/bin/behavior-classifier preflight --dataset-dir /absolute/path/to/db-cow-walking
.venv/bin/behavior-classifier benchmark --dataset-dir /absolute/path/to/db-cow-walking --output-dir /absolute/path/to/new-output
```

Every output directory must be new and is atomically published. Outputs are JSON/JSONL summaries, hashes, metrics, manifests, and derived predictions/context/records only—never raw source rows, feature matrices, or CSV reports. See the assistant [`README`](cattle-anomaly-assistant/README.md) for the full workflow and command semantics.

## 6. What's explicitly not your problem right now

This repo has a lot else going on that's unrelated to your task:
- `src/herd_simulator/` and `src/collar_gateway/` — a separate cattle-fleet digital-twin simulator and physical-collar firmware. Ignore completely.
- `cattle-anomaly-assistant/stage2_rag_assistant/` — the already-built LLM/RAG explanation layer (KB, pipeline, fallback logic). Already done and tested against fake data; not your concern.
- The `MmCows` dataset / SPC-CUSUM baseline work (a different Stage 1 sub-task, "M1b"/"M1c" in the tracker — see [`STAGE1_ANOMALY_BASELINE_HANDOFF.md`](STAGE1_ANOMALY_BASELINE_HANDOFF.md)) — someone else's job, not yours.

Stay inside `cattle-anomaly-assistant/stage1_anomaly_detection/`.

## 7. Reading order

1. This document.
2. [`LLM_Diagnostic_Assistant_PRD.md`](LLM_Diagnostic_Assistant_PRD.md) Section 2 (background/datasets) — skim the rest, you don't need Stage 2's details.
3. [`cattle-anomaly-assistant/README.md`](cattle-anomaly-assistant/README.md) — the supported M1a/M1d workflow.
4. [`AGENTS.md`](AGENTS.md) §3 — the golden rules; the two that bind your work specifically are called out in Section 3 above.
5. [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) §5's "M1a" section — the exact task checklist this handoff is based on, kept current as work progresses.

## 8. Reporting progress

[`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) is the shared tracker across everyone and every tool working on this project — when you finish a chunk of M1a's checklist, tick the box in §5 and add a dated line to the progress log in §8, the same way every other milestone here has been tracked. That's how the project owner (and anyone else who picks this up) knows what happened without having to ask you directly or dig through commit history.
