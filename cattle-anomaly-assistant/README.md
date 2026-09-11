# Cattle Anomaly Assistant

Stage 1 detects per-cow behavioural or physiological shifts; Stage 2 turns the resulting structured record into a grounded explanation. This is a self-contained subproject: it never imports from `../src/`, and Stage 1 never passes raw IMU or temperature rows to an LLM.

Read [`../LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) and [`../LLM_ASSISTANT_STATUS.md`](../LLM_ASSISTANT_STATUS.md) before changing either stage.

## Setup and fixture verification

```bash
cd cattle-anomaly-assistant
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[behavior,dev]'   # everything: Stage 1 ML deps + ragas + pytest-asyncio
.venv/bin/python -m pytest tests/ -v
```

This is the supported local setup for Python 3.11–3.13. Python 3.14 is intentionally unsupported until compatible XGBoost wheels are verified. The test fixtures are checked in; the public datasets are not. Stage 2-only work (no Stage 1 ML code) may install `-e '.[dev]'` instead — `pip install -r requirements.txt` is the plain-pip equivalent of the full `[behavior,dev]` install, kept in sync with `pyproject.toml`.

If anything importing `ragas` fails on a fresh install, read [`stage2_rag_assistant/eval/_ragas_compat.py`](stage2_rag_assistant/eval/_ragas_compat.py)'s docstring first — it's a documented upstream packaging bug, not a setup mistake.

```bash
.venv/bin/python -m stage2_rag_assistant.kb.build_kb --overwrite   # build the local KB (gitignored, not needed for tests)
.venv/bin/python -m stage2_rag_assistant.eval.run_eval             # run the Layer 2 eval, writes a report under eval/reports/
.venv/bin/python -m stage2_rag_assistant.calibration.tune_threshold   # prototype tau bucketing, writes a report under calibration/reports/
```

### Running against a real LLM (OpenRouter)

`llm.provider: fake` stays the checked-in default in [`config/default.yaml`](stage2_rag_assistant/config/default.yaml) — tests and a bare `run_eval`/`tune_threshold` invocation never need an API key. To run the pipeline for real against [OpenRouter](https://openrouter.ai) (`minimax/minimax-m3` by default):

```bash
cp .env.example .env   # then fill in OPENROUTER_API_KEY yourself — never commit this file
set -a && source .env && set +a
.venv/bin/python -m stage2_rag_assistant.eval.run_eval --pipeline-config stage2_rag_assistant/config/openrouter.yaml
.venv/bin/python -m stage2_rag_assistant.calibration.tune_threshold --pipeline-config stage2_rag_assistant/config/openrouter.yaml
```

See [`stage2_rag_assistant/config/openrouter.yaml`](stage2_rag_assistant/config/openrouter.yaml) and [`stage2_rag_assistant/llm/providers/openrouter_provider.py`](stage2_rag_assistant/llm/providers/openrouter_provider.py). This only wires the main Stage 2 pipeline's generation/routing calls — the Ragas eval judge (`eval/ragas_llm.py`) is a separate, still-deferred decision (real judging would mean a real paid API call per metric per golden case).

## Layout

- `shared/schemas/` — the frozen `AnomalyRecord` / `AnomalyExplanationQuery` / `AnomalyExplanationResponse` / `GoldenCase` pydantic models (PRD Section 6), all inheriting a `StrictModel` base (`extra="forbid"`, `AwareDatetime` timestamps). Both stages import from here; never duplicate a model.
- `stage1_anomaly_detection/` — see the M1a–M1d sections below for `behavior_classifier/` (WASP six-axis benchmark + the accepted external reference model) and `baseline_spc_cusum/` (MmCows CUSUM baseline + injection harness); `to_anomaly_record.py` is the fusion entry point.
- `stage2_rag_assistant/mock/generate_mock_records.py` — produces schema-valid mock `AnomalyRecord`s so Stage 2 could be built before Stage 1 exists (PRD Section 6.2). The real Stage 1 output to swap in at the Integration Point is `historical_demo_anomaly_records()`'s cross-dataset output, not a same-cow fused record — see M1d below for why.
- `stage2_rag_assistant/config/` — Stage 2 pipeline config (`default.yaml` + pydantic loader). `llm.provider: fake` by default; `openrouter.yaml` is the real-provider config (see above).
- `stage2_rag_assistant/llm/` — the `LLMClient` protocol, a factory (`fake` and `openrouter` work today; `anthropic`/`openai`/`google` still raise `NotImplementedError`), `providers/fake_provider.py`, and `providers/openrouter_provider.py` (stdlib `urllib`, no SDK dependency — reads `OPENROUTER_API_KEY` from the environment only, never from config).
- `stage2_rag_assistant/kb/` — the SQLite knowledge base: `schema.sql`, an editable `seed_data/shift_categories.yaml` (the actual source of truth — edit this, not the built DB), and `build_kb.py`. The built `kb.sqlite3` is gitignored.
- `stage2_rag_assistant/pipeline/` — the 5 pipeline stages (`record_assembler`, `intent_router`, `retriever`, `generator`, `fallback_gate`) plus `response_builder`/`audit_log`, wired end to end by `orchestrator.run_pipeline()`.
- `stage2_rag_assistant/eval/` — the Layer 2 (Ragas) eval harness: a fake judge LLM/embedding (`ragas_llm.py`/`ragas_embeddings.py`, same deferred-provider policy as `llm/`), field mapping + scoring (`metrics.py`), a 24-case golden set under `golden/` (17 mock-derived + 7 hand-written adversarial — `real_cases.jsonl`/`injected_cases.jsonl` need real Stage 1 output and are deliberately not created yet), and `run_eval.py`. Read `_ragas_compat.py` before touching anything that imports `ragas` directly.
- `stage2_rag_assistant/golden_loading.py` — shared golden-set loading (`load_cases`/`query_for`), used by both `eval/` and `calibration/` — lives here rather than nested under either since both depend on it equally.
- `stage2_rag_assistant/calibration/` — the τ-calibration prototype (PRD Section 10 steps 1-3 only): a deterministic calibration/test split of the golden set, confidence bucketing, and the LLM's own per-bucket empirical accuracy (`bucketing.py`), run via `tune_threshold.py`. Never writes to `config/default.yaml`'s real `tau` — steps 4-6 need real Stage 1 output (Integration Point), not this milestone.
- `stage2_rag_assistant/api/` — not created yet; see `LLM_ASSISTANT_STATUS.md` for what's next.

## M1a — WASP six-axis behaviour benchmark

`.venv/bin/behavior-classifier` rebuilds a four-state XGBoost classifier from public WASP source CSVs. It reads only the six MPU9250 acceleration/gyroscope columns, makes 50-sample (5-second) windows at 10 Hz with a 25-sample stride, and produces 112 deterministic statistical/spectral features. It has no import from the older repository `src/ml` code.

The internal class order is contiguous for XGBoost: resting, grazing, walking, miscellaneous. Output is decoded to the platform-safe behaviour codes `0`, `1`, `3`, and `5`; miscellaneous is never treated as Restless (`4`). The reported maximum class probability is explicitly **uncalibrated**.

**This repo's own from-scratch benchmark is real, but its real result fails its own eligibility gate** (mean LOCO macro F1 0.63 against a required ≥0.85 — see `benchmark_report.json` from an actual run). The accepted M1a deliverable going forward is instead a pre-trained external reference model — see "Historical dataset application mode" below — since this project will never have labeled collar-cow data to train and properly LOCO-validate a same-cow model. The from-scratch pipeline below is kept as real, working, honestly-negative evidence, not deleted.

Start with a no-write check of the local public data. The attached friend-supplied `.pkl` may be mentioned only as an inspection reference: it is hashed but never loaded or deserialized.

```bash
.venv/bin/behavior-classifier preflight \
  --dataset-dir /Users/taherpanbiharwala/Desktop/IoT/db-cow-walking \
  --legacy-pickle /Users/taherpanbiharwala/Downloads/cow_behavior_xgboost.pkl
```

The preflight reports only derived counts, class/cow support, timing-gap exclusions, and a combined source hash. It must pass before training. The expected public layout has `Resting`, `Grazing`, `Walking`, and `Miscellaneous behaviors` label directories and CSVs with `Time` plus the six `MPU9250_*` channels.

Run the grouped public benchmark into a new directory:

```bash
.venv/bin/behavior-classifier benchmark \
  --dataset-dir /Users/taherpanbiharwala/Desktop/IoT/db-cow-walking \
  --output-dir /Users/taherpanbiharwala/Desktop/wasp-behavior-benchmark-001 \
  --legacy-pickle /Users/taherpanbiharwala/Downloads/cow_behavior_xgboost.pkl
```

This uses nested leave-one-cow-out (LOCO): each cow is held out once, tuning uses only the remaining cows, and the primary score is the unweighted mean of outer-fold macro F1 over classes actually present in each held-out cow. It also reports pooled out-of-fold F1, class support/recall (`null` when a held-out cow has no support), confusion matrix, and fixed-seed cow-level uncertainty.

The published `0.9625` random-split result is stored as a reference only; it is not a LOCO target. A native JSON artifact is created only if this provisional **public-benchmark** gate passes: mean LOCO macro F1 ≥0.85 plus pooled Walking and Miscellaneous recall ≥0.75. Passing this gate is not collar, farm, health, or clinical validation. **The real run against the real dataset did not pass this gate** (0.63 macro F1, 0.15 miscellaneous recall) — no artifact was produced, which is the expected, correct behavior of a failed gate, not a bug.

Benchmark output never contains raw IMU rows or feature matrices:

- `benchmark_report.json`, `fold_metrics.jsonl`, `class_metrics.json`, `confusion_matrix.json`, and `uncertainty.json` — grouped evaluation evidence.
- `dataset_provenance.json` — aggregate hashes and counts only.
- `behavior_model.json` and `behavior_model.manifest.json` — native XGBoost JSON plus feature/class/window/provenance/hash contract, only if the benchmark gate passes.

Verify a qualifying artifact before any inference:

```bash
.venv/bin/behavior-classifier verify-artifact \
  --model-path /Users/taherpanbiharwala/Desktop/wasp-behavior-benchmark-001/behavior_model.json
```

`.pkl`/`.pickle` paths are rejected with `LEGACY_PICKLE_REJECTED`. Model tampering, changed features, unsupported manifest versions, non-finite values, and invalid probability shapes all fail closed with an actionable JSON error.

## M1d — future same-cow daily fusion

The public WASP behavior data and public MmCows physiology data are different cows, dates, sensors, and deployments. **Do not join them.** `fuse-daily` deliberately rejects these sources with `CONTEXT_SOURCE_NOT_RUNTIME`; it creates no `AnomalyRecord` and never silently reports a non-anomaly. **This will never change for this project** — there is no plan to acquire aligned same-cow runtime behavior + physiology data, so the workflow below documents the mechanism, not an expected near-term path; the accepted real Stage 1 → Stage 2 handoff is "Historical dataset application mode" below instead.

For a future compatible deployment, the workflow would be:

```bash
# 1. Predict from input-only 50x6, 10 Hz runtime IMU JSONL.
.venv/bin/behavior-classifier predict --help

# 2. Aggregate only the derived predictions, with an explicit expected count.
.venv/bin/behavior-classifier aggregate-context --help

# 3. Fuse only exactly aligned same-cow runtime behavior and CUSUM context.
.venv/bin/behavior-classifier fuse-daily --help
```

`predict` accepts only `same_cow_runtime` records with 50 timestamped 10 Hz six-axis samples, declared `m_s2`/`deg_s` units, and a calibration ID; it writes `behavior_predictions.jsonl` without copying the raw samples. `aggregate-context` creates `BehaviorDailyContext` records when ≥75% of expected windows are observed, using a deterministic daily state/distribution and uncalibrated confidence. `fuse-daily` requires matching `(cow_id, local_date, timezone, deployment_id)` keys, unique rows, valid coverage, and matching runtime provenance; it writes `anomaly_records.jsonl` only after every requested join validates atomically.

Fusion copies the CUSUM flag, score, physiology values, and literal drivers unchanged. Behaviour is context only in this release: it does not alter the score or add `behavior_state` as a driver. Records use the end of the local day converted to UTC and retain the prior three successfully fused records for the same cow. The Stage 2 record assembler already consumes the resulting shared `AnomalyRecord` schema without a code change.

### Historical dataset application mode — the accepted M1a/Integration Point mechanism

The application runs from the public datasets already staged locally. `historical-demo-records` uses the accepted reference XGBoost model (`stage1_anomaly_detection/behavior_classifier/reference_model/`, see that directory's own README for its full provenance and caveats) to classify the public WASP recordings, then attaches the resulting **dataset-level historical behavior summary** to each independently-derived MmCows CUSUM `AnomalyRecord`.

```bash
.venv/bin/behavior-classifier historical-demo-records \
  --wasp-dataset-dir /Users/taherpanbiharwala/Desktop/IoT/db-cow-walking \
  --model-path stage1_anomaly_detection/behavior_classifier/reference_model/cow_behavior_xgboost_reference.json \
  --manifest-path stage1_anomaly_detection/behavior_classifier/reference_model/cow_behavior_xgboost_reference.manifest.json \
  --cusum-windows-jsonl /Users/taherpanbiharwala/Desktop/MmCows-anomaly-results/run-002-with-immu/anomaly_windows.jsonl \
  --deployment-id mmcows-public-2023 \
  --timezone America/Chicago \
  --output-dir /Users/taherpanbiharwala/Desktop/MmCows-anomaly-results/historical-xgboost-demo-001
```

The output has `anomaly_records.jsonl`, `historical_behavior_summary.json`, and `summary.json`. It emits the flagged MmCows monitoring-day records (a real run produced 11), because the seven baseline days deliberately have no CUSUM deviations and must not be imputed. Every record explicitly carries `behavior_context_source="wasp_public_historical_dataset"` and `behavior_context_relation="cross_dataset_historical_demo"`. This makes the behavior result visible to the application without claiming that a WASP behavior prediction belongs to a MmCows cow or date. CUSUM is still the only source of the anomaly flag, score, and drivers.

## M1b — MmCows daily personal-baseline detector

`stage1_anomaly_detection.baseline_spc_cusum` reads only these MmCows wearable streams:

- Required: `main_data/cbt`, `main_data/ankle`, `main_data/thi`.
- Optional: either `main_data/immu/acceleration` or the public per-tag layout `main_data/immu/T01/T01_0721.csv`; add `--require-immu` to make its absence fail preflight.
- It normalizes the public `T01` and ankle-export `C01` directory conventions to `T01`…`T10`, and always excludes stationary T13/T14.

The upstream dataset identifies Unix timestamps as CDT and provides its selected sensors under `main_data`; stage only the needed extracted streams outside Git. See the [MmCows repository](https://github.com/neis-lab/mmcows) for the source layout and citation.

First run a no-write preflight:

```bash
.venv/bin/python -m stage1_anomaly_detection.baseline_spc_cusum \
  --data-root /Volumes/datasets/mmcows-sensor-data \
  --validate-only --require-immu
```

Then write derived artifacts to a separate location. The command refuses an output path that is the input directory, its parent, or its child.

```bash
.venv/bin/python -m stage1_anomaly_detection.baseline_spc_cusum \
  --data-root /Volumes/datasets/mmcows-sensor-data \
  --output-dir /Volumes/results/mmcows-baseline-run-001 \
  --config stage1_anomaly_detection/baseline_spc_cusum/config/default.json \
  --require-immu --verbose
```

The adapter accepts one-file-per-cow CSVs or wide CSVs with `T01`…`T10`/`C01`…`C10` value columns. It recognizes the public IMMU acceleration columns `accel_x_mps2`, `accel_y_mps2`, and `accel_z_mps2`; a row with a non-finite IMMU axis is excluded as an unobserved sample, never imputed. It fails with a JSON error and `UNSUPPORTED_HEADERS` rather than guessing a timestamp, value, axis, or cow-ID field. Use a JSON config's `columns` object for an explicit local-export override, for example:

```json
{
  "columns": {
    "ankle": {"timestamp": "timestamp", "value": "lying"},
    "immu": {"timestamp": "timestamp", "x": "ax", "y": "ay", "z": "az"}
  }
}
```

Numeric ankle values use `1=lying` and `0=standing` by default. Confirm that encoding during the staged-data preflight; an export with different explicit values must provide `ankle_lying_values` and `ankle_standing_values` in its config. The detector never invents an orientation threshold.

### Detector contract

All daily windows use `America/Chicago`, measured per-stream cadence, and at least 75% observed coverage. Missing values are never imputed. It learns each signal's immutable mean and sample standard deviation from the first seven consecutive valid days. A zero-variance or unavailable signal is marked unavailable; it is never made usable by dividing by an epsilon.

CBT, daily lying-time percentage, and optional daily IMMU activity magnitude each receive an independent two-sided CUSUM with `k=0.5σ` and `h=5σ`. IMMU activity is gravity-centred three-axis acceleration RMS, aggregated online per day so raw 100ms samples are not retained in detector output. A gap or invalid monitoring day resets that signal's CUSUM. THI is copied into every daily record as environmental context and is never an anomaly driver.

The separate output directory contains only derived values:

- `daily_features.jsonl` — daily values and coverage/provenance.
- `baselines.jsonl` — fixed reference distributions and availability reasons.
- `cusum_traces.jsonl` — both CUSUM directions, deviation, reset reason, and alarm.
- `anomaly_windows.jsonl` — interim Stage 1 fields, score, and literal drivers: `cbt`, `lying_time`, `activity_magnitude`.
- `manifest.json` — version, configuration, inputs, and safety limitations.

`anomaly_score = min(max_active_cusum / (2h), 1)`. A flag fires when any monitored CUSUM reaches `h`. These are anomaly indicators, not clinical or veterinary diagnoses; MmCows has no verified healthy reference period, so the baseline metadata explicitly records that limitation.

## M1c — deterministic feature injection

Injections operate only on derived monitoring-day features after the immutable baseline, never on raw sensors or baseline days. This is both the detector's correctness harness and a future source of mechanically-labelled evaluation cases.

```bash
.venv/bin/python -m stage1_anomaly_detection.baseline_spc_cusum \
  --data-root /Volumes/datasets/mmcows-sensor-data \
  --output-dir /Volumes/results/mmcows-cbt-injection \
  --injection-config stage1_anomaly_detection/baseline_spc_cusum/config/injection_template.json
```

Each scenario has a `cow_id`, `start_date`, positive `duration_days`, and `changes`. A change names one literal detector signal and its signed baseline-relative target deviation. Use `+2.5` for a CBT increase, `-2.5` for a lying-time or activity decrease, and an empty `changes` list for an explicit clean control. A sustained ±2.5σ target for three days contributes 2 CUSUM units/day and crosses the conservative `h=5` threshold on day three. The bundled template exercises CBT rise, both lying-time directions, activity drop, a combined change, and a clean control.

Real MmCows smoke results should report dataset coverage and mechanical detection behaviour only—never disease prevalence, clinical accuracy, or false-positive rates without verified health labels.

### Verified real-data smoke test

The detector was run on a staged 15-day MmCows subset containing CBT, ankle, THI, and the 6.8 GB public per-tag IMMU export. All ten wearable cows had usable baselines for all three monitored signals (30 total), and all 150 daily feature windows were processed. The normal run emitted 11 mechanical anomaly windows: CBT drove 9 and lying time drove 3; activity did not independently cross the conservative threshold in this dataset.

The deterministic real-data injection run then verified third-day CUSUM triggering for CBT rise, lying-time rise, lying-time drop, activity drop, and a combined three-signal change (independently re-derived by hand during a later review, not just read from the report). Its clean control remained unflagged. Some post-injection windows remained flagged because CUSUM retains accumulated evidence until it decays or a monitoring gap resets it. These checks validate detector mechanics, not disease-detection accuracy.
