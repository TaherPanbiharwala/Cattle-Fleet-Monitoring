# Stage 1 Behavior Classifier — Handoff

**For:** whoever's picking up the WASP-lab (`db-cow-walking`) behavior classification work.
**Your job, in one sentence:** build a baseline ML model that classifies a cow's 5-second IMU window as walking/grazing/resting/other, reproducing (then beating) a published benchmark under proper leave-one-cow-out validation.
**Last updated:** 2026-09-10

---

## 1. The 60-second version of what this project is

This repo is building a system that detects when a dairy cow's behavior/physiology has shifted from her own normal, and explains that shift in plain language. It's split into two independent halves:

- **Stage 1 (this is you)** — classical ML, no LLM anywhere. Two parts: (a) a behavior classifier trained on public IMU data — **your task** — and (b) a per-cow statistical baseline over a different dataset (someone else's task, not yours, don't worry about it). Together they produce a structured record saying "this cow's behavior/physiology looks different, and here's specifically what changed."
- **Stage 2 (already built)** — an LLM layer that takes Stage 1's output and explains it in cited, farmer-readable language, with a hard-grounded fallback so it never invents facts. This is done and tested (against fake/mocked Stage 1 data) — you don't need to touch it, and you don't need to understand it deeply. It's just the reason your output needs to match a specific shape (Section 4 below).

The full spec is [`LLM_Diagnostic_Assistant_PRD.md`](LLM_Diagnostic_Assistant_PRD.md) and the living project tracker is [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md). You don't need to read either cover to cover — pointers below tell you exactly which sections matter for your part.

## 2. What already exists for you to build on

Someone already wrote a first pass at this — **before** this exact plan existed, as part of an earlier phase of the broader project (this repo also runs a cattle-fleet IoT simulator; ignore that entirely, it's unrelated to your work). It's at:

- [`src/dataset_adapters/wasp_lab.py`](src/dataset_adapters/wasp_lab.py) — parses the WASP-lab dataset's filename/CSV format.
- [`src/ml/features.py`](src/ml/features.py) — feature engineering: 5-second windows at 10 Hz with 50% overlap, 14 statistical/spectral measures across 8 signals (6 IMU axes + 2 magnitudes) = 112 features per window.
- [`src/ml/benchmark.py`](src/ml/benchmark.py), [`src/ml/train.py`](src/ml/train.py) — 5 model tiers (Logistic Regression, Random Forest, RBF SVM, Gradient-Boosted Trees, an experimental 1D CNN) and a leave-one-cow-out cross-validation harness.

**This code has never been run against real data.** No committed artifacts, no confirmed accuracy number — treat "does it actually work" as open, not assumed. That's the first thing to find out.

**Your first real step:** copy those four files into a new folder, `cattle-anomaly-assistant/stage1_anomaly_detection/behavior_classifier/`, adapting imports/paths as needed. Copy, don't import from `src/` — `cattle-anomaly-assistant/` is a self-contained subproject with its own `.venv`, own `pyproject.toml`, and deliberately no dependency on the old `src/` tree. Leave the `src/` originals alone.

## 3. The actual target

- **Dataset:** `db-cow-walking` (Morales-Vargas et al. 2025) — 441 labeled IMU events from 10 cows. Get it from [github.com/WASP-lab/db-cow-walking](https://github.com/WASP-lab/db-cow-walking), or follow [`PHASE3_BENCHMARK.md`](PHASE3_BENCHMARK.md)'s Kaggle setup (a private Kaggle dataset + a notebook, `notebooks/phase3_wasp_benchmark.ipynb` — built for the old `src/ml/` location, you'll want your own copy/adaptation once your code lives in the new folder).
- **The bar:** beat the *published* SVM baseline — **96.29% accuracy / 0.9625 macro-F1**, under leave-one-cow-out cross-validation. This is the number that governs (per [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) §2) — `PHASE3_BENCHMARK.md` states an older, lower gate (macro-F1 ≥ 0.85) from before this project's current direction; that one's superseded, don't target it.
- **Classes:** walking, grazing, resting, miscellaneous/other. **Never map miscellaneous to "restless"** — a hard rule everywhere in this codebase (`AGENTS.md` golden rule 6), because "restless" means something specific and safety-relevant elsewhere in this system.
- **Validation:** leave-one-cow-out, always. Never split windows from the same cow across train and test — that's data leakage and it's treated as a hard rule here (`AGENTS.md`, `DECISION.md` ADR-015), not a style preference.
- The PRD (`LLM_Diagnostic_Assistant_PRD.md` Section 2) references a file called `behavior-classification-beat-baseline-plan.md` for more detail on this target — that file isn't in this git repo (it's referenced as living in the project owner's separate Claude Projects space per the PRD's own reference list, Section 17). If you need more detail than what's here and in `PHASE3_BENCHMARK.md`, ask for it rather than assuming it doesn't exist.

## 4. The output shape you're eventually building toward

Not needed for your first baseline pass, but worth knowing where this is headed: once you have a working classifier, it needs to eventually emit, per window:

- `behavior_state`: one of `walking | grazing | resting | miscellaneous`
- `behavior_state_confidence`: float in [0,1]
- `behavior_state_distribution_24h`: a rolling 24-hour distribution across the 4 classes — **must sum to ~1.0** (this is schema-enforced downstream, not just a suggestion)

These three fields are part of a frozen, already-built pydantic schema — see [`cattle-anomaly-assistant/shared/schemas/anomaly_record.py`](cattle-anomaly-assistant/shared/schemas/anomaly_record.py) if you're curious, but you don't need to touch it. Get a working, validated baseline classifier first; wiring your output into that schema (a file called `to_anomaly_record.py`) is a separate, later task.

## 5. What's explicitly not your problem right now

This repo has a lot else going on that's unrelated to your task:
- `src/herd_simulator/` and `src/collar_gateway/` — a separate cattle-fleet digital-twin simulator and physical-collar firmware. Ignore completely.
- `cattle-anomaly-assistant/stage2_rag_assistant/` — the already-built LLM/RAG explanation layer (KB, pipeline, fallback logic). Already done and tested against fake data; not your concern.
- The `MmCows` dataset / SPC-CUSUM baseline work (a different Stage 1 sub-task, "M1b" in the tracker) — someone else's job, not yours.

Stay inside `cattle-anomaly-assistant/stage1_anomaly_detection/`.

## 6. Reading order

1. This document.
2. [`LLM_Diagnostic_Assistant_PRD.md`](LLM_Diagnostic_Assistant_PRD.md) Section 2 (background/datasets) — skim the rest, you don't need Stage 2's details.
3. [`PHASE3_BENCHMARK.md`](PHASE3_BENCHMARK.md) — the existing setup instructions for this exact dataset/benchmark; reuse this workflow rather than reinventing it.
4. [`AGENTS.md`](AGENTS.md) §3 — the golden rules; the two that bind your work specifically are called out in Section 3 above.
5. [`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) §5's "M1a" section — the exact task checklist this handoff is based on, kept current as work progresses.

## 7. Reporting progress

[`LLM_ASSISTANT_STATUS.md`](LLM_ASSISTANT_STATUS.md) is the shared tracker across everyone and every tool working on this project — when you finish a chunk of M1a's checklist, tick the box in §5 and add a dated line to the progress log in §8, the same way every other milestone here has been tracked. That's how the project owner (and anyone else who picks this up) knows what happened without having to ask you directly or dig through commit history.
