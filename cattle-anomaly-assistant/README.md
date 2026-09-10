# Cattle Anomaly Assistant

Stage 1 detects per-cow behavioural or physiological shifts; Stage 2 turns the resulting structured record into a grounded explanation. This is a self-contained subproject: it never imports from `../src/`, and Stage 1 never passes raw IMU or temperature rows to an LLM.

Read [`../LLM_Diagnostic_Assistant_PRD.md`](../LLM_Diagnostic_Assistant_PRD.md) and [`../LLM_ASSISTANT_STATUS.md`](../LLM_ASSISTANT_STATUS.md) before changing either stage.

## Setup and fixture verification

```bash
cd cattle-anomaly-assistant
python3 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -v
```

The Stage 1 test fixture is checked in and does not need MmCows. It covers CSV discovery, CDT conversion, coverage, immutable baselines, zero variance, CUSUM resets, output publication, and deterministic injections.

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

The adapter accepts one-file-per-cow CSVs or wide CSVs with `T01`…`T10`/`C01`…`C10` value columns. It recognizes the public IMMU acceleration columns `accel_x_mps2`, `accel_y_mps2`, and `accel_z_mps2`; a row with a non-finite IMMU axis is excluded as an unobserved sample, never imputed. It fails with a JSON error and `UNSUPPORTED_HEADERS` rather than guessing a timestamp, value, axis, or cow-ID field. Use a JSON config’s `columns` object for an explicit local-export override, for example:

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

All daily windows use `America/Chicago`, measured per-stream cadence, and at least 75% observed coverage. Missing values are never imputed. It learns each signal’s immutable mean and sample standard deviation from the first seven consecutive valid days. A zero-variance or unavailable signal is marked unavailable; it is never made usable by dividing by an epsilon.

CBT, daily lying-time percentage, and optional daily IMMU activity magnitude each receive an independent two-sided CUSUM with `k=0.5σ` and `h=5σ`. IMMU activity is gravity-centred three-axis acceleration RMS, aggregated online per day so raw 100ms samples are not retained in detector output. A gap or invalid monitoring day resets that signal’s CUSUM. THI is copied into every daily record as environmental context and is never an anomaly driver.

The separate output directory contains only derived values:

- `daily_features.jsonl` — daily values and coverage/provenance.
- `baselines.jsonl` — fixed reference distributions and availability reasons.
- `cusum_traces.jsonl` — both CUSUM directions, deviation, reset reason, and alarm.
- `anomaly_windows.jsonl` — interim Stage 1 fields, score, and literal drivers: `cbt`, `lying_time`, `activity_magnitude`.
- `manifest.json` — version, configuration, inputs, and safety limitations.

`anomaly_score = min(max_active_cusum / (2h), 1)`. A flag fires when any monitored CUSUM reaches `h`. These are anomaly indicators, not clinical or veterinary diagnoses; MmCows has no verified healthy reference period, so the baseline metadata explicitly records that limitation.

## M1c — deterministic feature injection

Injections operate only on derived monitoring-day features after the immutable baseline, never on raw sensors or baseline days. This is both the detector’s correctness harness and a future source of mechanically-labelled evaluation cases.

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

The deterministic real-data injection run then verified third-day CUSUM triggering for CBT rise, lying-time rise, lying-time drop, activity drop, and a combined three-signal change. Its clean control remained unflagged. Some post-injection windows remained flagged because CUSUM retains accumulated evidence until it decays or a monitoring gap resets it. These checks validate detector mechanics, not disease-detection accuracy.
