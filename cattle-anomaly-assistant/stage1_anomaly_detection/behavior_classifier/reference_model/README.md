# Reference XGBoost model — M1a's accepted deliverable

MTZ's explicit decision (2026-09-11): this externally-trained model, not
`benchmark.py`'s own from-scratch pipeline, is M1a's accepted output going
forward. The repo's own leave-one-cow-out benchmark against the real WASP
`db-cow-walking` dataset produced a real result — macro F1 0.63, "miscellaneous
behaviors" recall 0.15 — that fails its own eligibility gate (`primary >= 0.85`,
`walking_recall >= 0.75`, `miscellaneous_recall >= 0.75`, see `benchmark.py`),
so no qualifying artifact exists from that path. Since the project will never
have its own labeled collar-cow behavior data to train and properly validate a
same-cow model against, this pre-trained reference is what M1a runs on instead
— not a stopgap, the accepted answer for the amount of real data this project
actually has.

## What this is, honestly

Per `cow_behavior_xgboost_reference.metrics.json`/`.manifest.json` (both
included here verbatim, never edit their contents — the code validates
`model_sha256`/`feature_sha256` against them before loading):

- **`scope: "reference_only"`, `not_valid_for_collar_or_clinical_use: true`,
  `eligible_for_project_collar_deployment: false`,
  `eligible_for_public_WASP_plus_MmCows_fusion: false`** — the manifest itself
  says so; nothing here overrides that.
- **`split_protocol: "stratified random window split; not grouped by cow or
  recording"`** — the reported 0.982 macro F1 / 98.9% accuracy is *not* a
  leave-one-cow-out number. Windows from the same cow/recording can appear on
  both sides of the split, which inflates apparent accuracy relative to a
  true held-out-cow evaluation. Treat it as an optimistic upper bound, not a
  real-world estimate.
- **25 raw sensor channels (16 BNO055 + 9 MPU9250) × 9 stats = 225 features**
  (`manifest.json`'s `feature_names`/`sensor_columns`) — a richer sensor set
  than this project's own WASP adapter extracts (6 raw MPU9250 channels,
  112 derived features, `features.py`). This model was trained on a dataset
  with more sensor channels than the project's own WASP ingestion path reads,
  which is *why* it's used only via `historical_demo.py`'s independent code
  path (reading its own 25-channel CSV columns directly), never through
  `wasp.py`/`features.py`/`benchmark.py`.
- **Cross-dataset only**: `historical_demo_anomaly_records()` never claims
  WASP and MmCows rows share a cow, date, or deployment — see that module's
  docstring. This is the accepted Integration Point mechanism (MTZ, 2026-09-11):
  a labelled historical behavior summary attached to independently-derived
  MmCows CUSUM indicators, not a same-cow runtime inference.

## Usage

```bash
python -m stage1_anomaly_detection.behavior_classifier historical-demo-records \
  --wasp-dataset-dir <db-cow-walking dir> \
  --model-path stage1_anomaly_detection/behavior_classifier/reference_model/cow_behavior_xgboost_reference.json \
  --manifest-path stage1_anomaly_detection/behavior_classifier/reference_model/cow_behavior_xgboost_reference.manifest.json \
  --cusum-windows-jsonl <MmCows anomaly_windows.jsonl> \
  --deployment-id <id> \
  --timezone <IANA tz> \
  --output-dir <new dir>
```

Verified end-to-end (2026-09-11) against the real `db-cow-walking` WASP data
and a real MmCows CUSUM run: 11 records produced, every one validated cleanly
against the frozen `AnomalyRecord` schema via
`AnomalyRecord.model_validate()`.

## Files

- `cow_behavior_xgboost_reference.json` — native XGBoost JSON model (never
  pickle — `historical_demo.py` refuses `.pkl`/`.pickle` outright).
- `cow_behavior_xgboost_reference.manifest.json` — the contract
  `historical_demo._read_manifest()` checks the model against
  (`model_sha256`, `feature_names`, `feature_sha256`, `sensor_columns`,
  `class_labels`, `window_settings`) before it's ever loaded.
- `cow_behavior_xgboost_reference.metrics.json` — the training-time metrics
  above, kept for the record; not read by any code.
