# Phase 3: Public-Cow IMU Behaviour Benchmark

## Scope

This deliverable measures behaviour classification on the labelled WASP-lab
dataset. It is not a validation study on this project's physical collar, cow,
farm, breed, mounting position, or environment. It is not a fever, lameness,
diagnosis, or treatment model.

The only output classes are:

| Dataset label | Platform code | Model output |
|---|---:|---|
| Resting | 0 | Resting |
| Grazing | 1 | Grazing |
| Walking | 3 | Walking |
| Miscellaneous behaviours | 5 | Other/Unknown |

Ruminating (`2`) and Restless (`4`) are intentionally not inferred. In
particular, Miscellaneous never maps to Restless.

## Kaggle setup

1. Obtain the WASP files from the [source repository](https://github.com/WASP-lab/db-cow-walking).
2. Create a **private** Kaggle Dataset preserving the source attribution and
   licence, then attach it to `notebooks/phase3_wasp_benchmark.ipynb`.
3. In Kaggle, add a `WANDB_API_KEY` secret and enable Internet. The notebook
   reads it only into the current process; it is never printed or saved.
4. Update the notebook's `DATASET_DIR` to the attached input path and set
   `REPOSITORY_COMMIT` to the full 40-character repository commit SHA before
   running all cells. The notebook verifies the checked-out SHA before it
   installs the project.

Kaggle owns the raw-data input. The repository and W&B receive only aggregate
metrics, model artifacts, reports, and a hash-only manifest.

## Local equivalent

Install optional dependencies:

```bash
python -m pip install -e '.[ml]'
```

Run the strict cow-grouped benchmark:

```bash
python -m ml.train \
  --dataset-dir /absolute/path/to/db-cow-walking \
  --output-dir artifacts/wasp_seed42 \
  --wandb-mode online \
  --wandb-project cattle-fleet-phase3 \
  --source-revision "$(git rev-parse HEAD)"
```

Use `--wandb-mode disabled` for tests or `offline` to generate local W&B run
files without syncing. Never pass W&B keys on the command line.

## Evaluation contract

- Only the six MPU9250 acceleration/gyroscope axes are read, matching the
  future MPU6050 capability; two magnitudes yield eight feature signals.
- Data is segmented into five-second windows at 10 Hz with 50% overlap.
  Windows never cross a source event or timing gap.
- Fourteen statistics/spectral measures per signal create 112 features.
- The outer split is leave-one-cow-out. Inner selection is cow-grouped. A
  cow, event, or overlapping window can never appear in both train and test.
- Models are Logistic Regression, Random Forest, RBF SVM, Gradient-Boosted
  Trees, and an experimental 1D CNN.

The release gate is cow-grouped macro F1 ≥ 0.85 with Walking and
Other/Unknown recall each ≥ 0.75. Missing a gate is reported as a benchmark
failure, never corrected with synthetic data.
