"""Leakage-safe, cow-grouped Phase 3 behaviour benchmark runner.

The generated reports describe public-dataset performance only.  This module
contains no fever, lameness, treatment, or target-farm prediction logic.
"""

from __future__ import annotations

import copy
import csv
import json
import pickle
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from dataset_adapters.wasp_lab import create_dataset_manifest, load_wasp_dataset

from .features import FEATURE_BASE_NAMES, FEATURE_NAMES, SAMPLE_RATE_HZ, WINDOW_SAMPLES, build_feature_dataset
from .wandb_tracking import WandbSettings, WandbTracker

BEHAVIOUR_CODES: tuple[int, ...] = (0, 1, 3, 5)
BEHAVIOUR_NAMES: dict[int, str] = {
    0: "Resting",
    1: "Grazing",
    3: "Walking",
    5: "Other/Unknown",
}
ModelName = Literal["logistic_regression", "random_forest", "rbf_svm", "gradient_boosted_trees", "cnn_1d"]


@dataclass(frozen=True)
class BenchmarkConfig:
    """All public benchmark choices, persisted to ``run_config.json``."""

    seed: int = 42
    wandb: WandbSettings = field(default_factory=WandbSettings)
    data_source_ref: str = "WASP-lab/db-cow-walking (private Kaggle input)"
    include_cnn: bool = True
    inner_splits: int = 5
    cnn_max_epochs: int = 30
    cnn_patience_epochs: int = 5
    source_revision: str | None = None


@dataclass(frozen=True)
class FoldMetric:
    model_name: str
    held_out_cow_id: str
    macro_f1: float
    accuracy: float
    walking_recall: float
    unknown_recall: float
    latency_ms_per_window: float
    train_windows: int
    test_windows: int
    best_params: str


@dataclass(frozen=True)
class PerClassMetric:
    """Per-cow, per-model class metrics safe to store in aggregate reports."""

    model_name: str
    held_out_cow_id: str
    behaviour_code: int
    behaviour_name: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class BenchmarkResult:
    output_dir: Path
    selected_model_name: str
    release_gate_passed: bool
    mean_macro_f1: float
    walking_recall: float
    unknown_recall: float


def _ml_imports() -> dict[str, Any]:
    """Load heavy optional dependencies only when a benchmark is executed."""

    try:
        import numpy as np
        from sklearn.base import clone
        from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
        from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except ImportError as exc:  # pragma: no cover - environment-dependent boundary
        raise RuntimeError(
            "Phase 3 training requires the 'ml' optional dependencies. "
            "Install them with: python -m pip install -e '.[ml]'"
        ) from exc
    return locals()


def _set_seed(seed: int, *, include_torch: bool) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - caught later by the trainer
        pass
    if include_torch:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment-dependent boundary
            raise RuntimeError(
                "The experimental 1D CNN requires PyTorch. Install the 'ml' optional dependencies."
            ) from exc
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)


def _macro_scorer(imports: dict[str, Any]) -> Callable[[Any, Any, Any], float]:
    f1_score = imports["f1_score"]

    def score(estimator: Any, features: Any, labels: Any) -> float:
        return float(
            f1_score(
                labels,
                estimator.predict(features),
                labels=list(BEHAVIOUR_CODES),
                average="macro",
                zero_division=0,
            )
        )

    return score


def _classical_specs(imports: dict[str, Any], seed: int) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    Pipeline = imports["Pipeline"]
    StandardScaler = imports["StandardScaler"]
    LogisticRegression = imports["LogisticRegression"]
    RandomForestClassifier = imports["RandomForestClassifier"]
    SVC = imports["SVC"]
    GradientBoostingClassifier = imports["GradientBoostingClassifier"]
    return {
        "logistic_regression": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            class_weight="balanced",
                            max_iter=2_000,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.1, 1.0]},
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=seed,
            ),
            {"max_depth": [None, 12], "min_samples_leaf": [1]},
        ),
        "rbf_svm": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            {"model__C": [0.5, 1.0], "model__gamma": ["scale"]},
        ),
        "gradient_boosted_trees": (
            GradientBoostingClassifier(random_state=seed),
            {"n_estimators": [100], "learning_rate": [0.1], "max_depth": [2, 3]},
        ),
    }


def _model_search_manifest(config: BenchmarkConfig) -> dict[str, dict[str, Any]]:
    """Return every result-affecting model setting in JSON-safe form."""

    return {
        "logistic_regression": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "fixed_parameters": {
                "class_weight": "balanced",
                "max_iter": 2_000,
                "random_state": config.seed,
            },
            "search_grid": {"model__C": [0.1, 1.0]},
        },
        "random_forest": {
            "estimator": "sklearn.ensemble.RandomForestClassifier",
            "fixed_parameters": {
                "n_estimators": 200,
                "class_weight": "balanced_subsample",
                "n_jobs": -1,
                "random_state": config.seed,
            },
            "search_grid": {"max_depth": [None, 12], "min_samples_leaf": [1]},
        },
        "rbf_svm": {
            "estimator": "sklearn.svm.SVC",
            "fixed_parameters": {"class_weight": "balanced", "random_state": config.seed},
            "search_grid": {"model__C": [0.5, 1.0], "model__gamma": ["scale"]},
        },
        "gradient_boosted_trees": {
            "estimator": "sklearn.ensemble.GradientBoostingClassifier",
            "fixed_parameters": {"random_state": config.seed},
            "search_grid": {
                "n_estimators": [100],
                "learning_rate": [0.1],
                "max_depth": [2, 3],
            },
        },
        "cnn_1d": {
            "estimator": "PyTorch TinyCnn",
            "fixed_parameters": {
                "input_channels": 6,
                "conv_channels": [32, 64],
                "kernel_sizes": [5, 3],
                "dropout": 0.20,
                "optimizer": "Adam",
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "batch_size": 64,
                "device": "cpu",
            },
            "early_stopping": {
                "max_epochs": config.cnn_max_epochs,
                "patience_epochs": config.cnn_patience_epochs,
                "validation": "one cow held out from each outer training partition",
            },
        },
    }


def _build_run_config(config: BenchmarkConfig, model_names: list[str]) -> dict[str, Any]:
    """Build the complete non-secret configuration persisted with a benchmark."""

    return {
        "schema_version": 1,
        "benchmark": {
            "seed": config.seed,
            "data_source_ref": config.data_source_ref,
            "source_revision": config.source_revision,
            "include_cnn": config.include_cnn,
            "inner_splits": config.inner_splits,
            "cnn_max_epochs": config.cnn_max_epochs,
            "cnn_patience_epochs": config.cnn_patience_epochs,
        },
        "models": model_names,
        "model_search": {
            model_name: _model_search_manifest(config)[model_name] for model_name in model_names
        },
        "validation": "leave-one-cow-out outer validation; grouped inner model selection",
        "wandb": asdict(config.wandb),
    }


def _fit_classical_model(
    *,
    model_name: str,
    features: Any,
    labels: Any,
    groups: Any,
    imports: dict[str, Any],
    seed: int,
    inner_splits: int,
) -> tuple[Any, dict[str, Any]]:
    specs = _classical_specs(imports, seed)
    estimator, param_grid = specs[model_name]
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("Inner grouped validation requires at least two cows.")
    GroupKFold = imports["GroupKFold"]
    GridSearchCV = imports["GridSearchCV"]
    cv = GroupKFold(n_splits=min(inner_splits, len(unique_groups)))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=_macro_scorer(imports),
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(features, labels, groups=groups)
    return search.best_estimator_, dict(search.best_params_)


@dataclass
class _CnnBundle:
    model: Any
    mean: Any
    std: Any


def _cnn_model(torch: Any) -> Any:
    nn = torch.nn

    class TinyCnn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv1d(6, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(0.20),
                nn.Linear(64, len(BEHAVIOUR_CODES)),
            )

        def forward(self, inputs: Any) -> Any:
            return self.network(inputs)

    return TinyCnn()


def _encode_labels(np: Any, labels: Any) -> Any:
    mapping = {code: index for index, code in enumerate(BEHAVIOUR_CODES)}
    try:
        return np.asarray([mapping[int(value)] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unexpected behaviour code in CNN labels: {exc.args[0]}") from exc


def _cnn_class_weights(np: Any, encoded_labels: Any) -> Any:
    """Compute stable inverse-frequency weights without dividing by zero."""

    class_counts = np.bincount(encoded_labels, minlength=len(BEHAVIOUR_CODES))
    weights = np.zeros(len(BEHAVIOUR_CODES), dtype=np.float64)
    present = class_counts > 0
    weights[present] = class_counts.sum() / class_counts[present]
    return weights


def _fit_cnn_model(
    *,
    raw_windows: Any,
    labels: Any,
    groups: Any,
    seed: int,
    max_epochs: int,
    patience_epochs: int,
) -> tuple[_CnnBundle, dict[str, Any]]:
    """Train the experimental CNN with one grouped validation cow for early stop."""

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed, include_torch=True)
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        raise ValueError("CNN early stopping requires at least two cows.")
    validation_cow = unique_groups[seed % len(unique_groups)]
    validation_mask = groups == validation_cow
    train_mask = ~validation_mask
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("Grouped CNN split produced an empty partition.")

    train_windows = raw_windows[train_mask]
    mean = train_windows.mean(axis=(0, 1), keepdims=True)
    std = train_windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    normalised = (raw_windows - mean) / std
    encoded = _encode_labels(np, labels)

    class_weights = _cnn_class_weights(np, encoded[train_mask])
    device = torch.device("cpu")
    model = _cnn_model(torch).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    train_dataset = TensorDataset(
        torch.tensor(normalised[train_mask].transpose(0, 2, 1), dtype=torch.float32),
        torch.tensor(encoded[train_mask], dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=generator)
    validation_inputs = torch.tensor(
        normalised[validation_mask].transpose(0, 2, 1), dtype=torch.float32, device=device
    )
    validation_labels = torch.tensor(encoded[validation_mask], dtype=torch.long, device=device)

    best_state: dict[str, Any] | None = None
    best_loss = float("inf")
    stale_epochs = 0
    epochs_completed = 0
    for epoch in range(max_epochs):
        model.train()
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_inputs), validation_labels).item())
        epochs_completed = epoch + 1
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience_epochs:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return _CnnBundle(model=model, mean=mean, std=std), {
        "validation_cow": validation_cow,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_loss,
    }


def _fit_cnn_final_model(
    *,
    raw_windows: Any,
    labels: Any,
    seed: int,
    epochs: int,
) -> _CnnBundle:
    """Refit the selected CNN on every valid window for the selected epoch count."""

    if epochs < 1:
        raise ValueError("Final CNN training requires at least one epoch.")

    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    _set_seed(seed, include_torch=True)
    mean = raw_windows.mean(axis=(0, 1), keepdims=True)
    std = raw_windows.std(axis=(0, 1), keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    normalised = (raw_windows - mean) / std
    encoded = _encode_labels(np, labels)
    class_weights = _cnn_class_weights(np, encoded)
    device = torch.device("cpu")
    model = _cnn_model(torch).to(device)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    dataset = TensorDataset(
        torch.tensor(normalised.transpose(0, 2, 1), dtype=torch.float32),
        torch.tensor(encoded, dtype=torch.long),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, generator=generator)
    model.train()
    for _ in range(epochs):
        for batch_inputs, batch_labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_inputs.to(device)), batch_labels.to(device))
            loss.backward()
            optimizer.step()
    model.eval()
    return _CnnBundle(model=model, mean=mean, std=std)


def _predict_cnn(bundle: _CnnBundle, raw_windows: Any) -> Any:
    import numpy as np
    import torch

    normalised = (raw_windows - bundle.mean) / bundle.std
    inputs = torch.tensor(normalised.transpose(0, 2, 1), dtype=torch.float32)
    with torch.no_grad():
        class_indexes = bundle.model(inputs).argmax(dim=1).cpu().numpy()
    return np.asarray([BEHAVIOUR_CODES[int(index)] for index in class_indexes], dtype=np.int64)


def _calculate_metrics(
    *,
    imports: dict[str, Any],
    expected: Any,
    predicted: Any,
    model: Any,
    inference_input: Any,
    predict: Callable[[Any, Any], Any],
) -> tuple[dict[str, Any], Any]:
    f1_score = imports["f1_score"]
    accuracy_score = imports["accuracy_score"]
    precision_recall_fscore_support = imports["precision_recall_fscore_support"]
    confusion_matrix = imports["confusion_matrix"]
    started = time.perf_counter()
    for _ in range(5):
        predict(model, inference_input)
    elapsed = time.perf_counter() - started
    precision, recall, f1, support = precision_recall_fscore_support(
        expected,
        predicted,
        labels=list(BEHAVIOUR_CODES),
        zero_division=0,
    )
    per_class = {
        BEHAVIOUR_NAMES[code]: {
            "code": code,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, code in enumerate(BEHAVIOUR_CODES)
    }
    metrics = {
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=list(BEHAVIOUR_CODES),
                average="macro",
                zero_division=0,
            )
        ),
        "accuracy": float(accuracy_score(expected, predicted)),
        "latency_ms_per_window": float((elapsed / 5 / len(expected)) * 1_000) if len(expected) else 0.0,
        "per_class": per_class,
    }
    return metrics, confusion_matrix(expected, predicted, labels=list(BEHAVIOUR_CODES))


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as destination:
        json.dump(value, destination, indent=2, sort_keys=True)
        destination.write("\n")


def _write_fold_metrics(path: Path, metrics: list[FoldMetric]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)


def _write_per_class_metrics(path: Path, metrics: list[PerClassMetric]) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=list(asdict(metrics[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(metric) for metric in metrics)


def _write_confusion_matrix(path: Path, matrix: Any) -> None:
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["actual\\predicted", *(str(code) for code in BEHAVIOUR_CODES)])
        for code, values in zip(BEHAVIOUR_CODES, matrix, strict=True):
            writer.writerow([code, *(int(value) for value in values)])


def _aggregate_per_class_metrics(matrix: Any) -> dict[str, dict[str, float | int]]:
    """Calculate complete per-class metrics from one model's aggregate confusion matrix."""

    result: dict[str, dict[str, float | int]] = {}
    for index, code in enumerate(BEHAVIOUR_CODES):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        result[BEHAVIOUR_NAMES[code]] = {
            "code": code,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    return result


def _write_model_card(
    path: Path,
    *,
    selected_model_name: str,
    summary: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    path.write_text(
        "# Phase 3 WASP Behaviour Benchmark Model Card\n\n"
        "## Intended use\n\n"
        "This artifact classifies 5-second public-dataset IMU windows as Resting, Grazing, "
        "Walking, or Other/Unknown. It is an academic benchmark only.\n\n"
        "## Critical limitation\n\n"
        "No on-animal validation was performed with this project's ESP32 collar, target farm, "
        "breed, mounting position, or environment. This is not a fever, lameness, diagnostic, "
        "or treatment model.\n\n"
        f"## Selected procedure\n\n- Model: `{selected_model_name}`\n"
        f"- Cow-grouped mean macro F1: `{summary['mean_macro_f1']:.4f}`\n"
        f"- Walking recall: `{summary['mean_walking_recall']:.4f}`\n"
        f"- Other/Unknown recall: `{summary['mean_unknown_recall']:.4f}`\n"
        f"- Final artifact size: `{summary['model_size_bytes']}` bytes\n"
        f"- Release gate passed: `{summary['release_gate_passed']}`\n\n"
        "## Data provenance\n\n"
        f"- Source: {manifest['source_url']}\n"
        f"- Combined input SHA-256: `{manifest['combined_sha256']}`\n"
        "- Raw samples and feature rows were not uploaded to experiment tracking.\n",
        encoding="utf-8",
    )


def _assert_group_integrity(cow_ids: Any, event_ids: Any, train_indices: Any, test_indices: Any) -> None:
    train_cows = set(cow_ids[train_indices].tolist())
    test_cows = set(cow_ids[test_indices].tolist())
    if train_cows.intersection(test_cows):
        raise RuntimeError("Cow-grouped split leakage detected.")
    train_events = {f"{cow}:{event}" for cow, event in zip(cow_ids[train_indices], event_ids[train_indices])}
    test_events = {f"{cow}:{event}" for cow, event in zip(cow_ids[test_indices], event_ids[test_indices])}
    if train_events.intersection(test_events):
        raise RuntimeError("Event-level split leakage detected.")


def _train_final_classical(
    *,
    model_name: str,
    dataset: Any,
    imports: dict[str, Any],
    config: BenchmarkConfig,
) -> tuple[Any, dict[str, Any]]:
    return _fit_classical_model(
        model_name=model_name,
        features=dataset.features,
        labels=dataset.labels,
        groups=dataset.cow_ids,
        imports=imports,
        seed=config.seed,
        inner_splits=config.inner_splits,
    )


def run_benchmark(dataset_dir: Path, output_dir: Path, config: BenchmarkConfig | None = None) -> BenchmarkResult:
    """Execute LOSO evaluation, write aggregate artifacts, and train one final bundle."""

    config = config or BenchmarkConfig()
    imports = _ml_imports()
    np = imports["np"]
    LeaveOneGroupOut = imports["LeaveOneGroupOut"]
    _set_seed(config.seed, include_torch=config.include_cnn)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = load_wasp_dataset(dataset_dir)
    dataset = build_feature_dataset(events)
    manifest = create_dataset_manifest(dataset_dir)
    if len(set(dataset.cow_ids.tolist())) < 3:
        raise ValueError(
            "Leave-one-cow-out validation with grouped inner tuning requires data from at least three cows."
        )

    _write_json(output_dir / "dataset_manifest.json", manifest)
    _write_json(
        output_dir / "feature_manifest.json",
        {
            "schema_version": 1,
            "sample_rate_hz": SAMPLE_RATE_HZ,
            "window_samples": WINDOW_SAMPLES,
            "window_overlap_samples": WINDOW_SAMPLES // 2,
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_definition": list(FEATURE_BASE_NAMES),
            "raw_samples_embedded": False,
        },
    )
    model_names: list[str] = list(_classical_specs(imports, config.seed))
    if config.include_cnn:
        model_names.append("cnn_1d")
    run_config = _build_run_config(config, model_names)
    _write_json(output_dir / "run_config.json", run_config)

    tracker = WandbTracker(config.wandb)
    outer_splitter = LeaveOneGroupOut()
    all_metrics: list[FoldMetric] = []
    all_per_class_metrics: list[PerClassMetric] = []
    cnn_epoch_counts: list[int] = []
    confusion_by_model: dict[str, Any] = defaultdict(lambda: np.zeros((4, 4), dtype=np.int64))

    for model_name in model_names:
        for fold_index, (train_indices, test_indices) in enumerate(
            outer_splitter.split(dataset.features, dataset.labels, dataset.cow_ids), start=1
        ):
            _assert_group_integrity(dataset.cow_ids, dataset.event_ids, train_indices, test_indices)
            held_out_cow = str(dataset.cow_ids[test_indices][0])
            tracking_config = {
                "benchmark": run_config["benchmark"],
                "model_search": run_config["model_search"][model_name],
                "model_name": model_name,
                "held_out_cow_id": held_out_cow,
                "fold_index": fold_index,
                "dataset_combined_sha256": manifest["combined_sha256"],
                "validation": "leave-one-cow-out",
            }
            tracked_run = tracker.start_fold(
                model_name=model_name,
                held_out_cow_id=held_out_cow,
                config=tracking_config,
            )
            try:
                if model_name == "cnn_1d":
                    fitted_model, best_params = _fit_cnn_model(
                        raw_windows=dataset.raw_windows[train_indices],
                        labels=dataset.labels[train_indices],
                        groups=dataset.cow_ids[train_indices],
                        seed=config.seed + fold_index,
                        max_epochs=config.cnn_max_epochs,
                        patience_epochs=config.cnn_patience_epochs,
                    )
                    cnn_epoch_counts.append(int(best_params["epochs_completed"]))
                    predictions = _predict_cnn(fitted_model, dataset.raw_windows[test_indices])
                    predict = lambda model, values: _predict_cnn(model, values)
                    inference_input = dataset.raw_windows[test_indices]
                else:
                    fitted_model, best_params = _fit_classical_model(
                        model_name=model_name,
                        features=dataset.features[train_indices],
                        labels=dataset.labels[train_indices],
                        groups=dataset.cow_ids[train_indices],
                        imports=imports,
                        seed=config.seed + fold_index,
                        inner_splits=config.inner_splits,
                    )
                    predictions = fitted_model.predict(dataset.features[test_indices])
                    predict = lambda model, values: model.predict(values)
                    inference_input = dataset.features[test_indices]
                metrics, matrix = _calculate_metrics(
                    imports=imports,
                    expected=dataset.labels[test_indices],
                    predicted=predictions,
                    model=fitted_model,
                    inference_input=inference_input,
                    predict=predict,
                )
                confusion_by_model[model_name] += matrix
                fold_metric = FoldMetric(
                    model_name=model_name,
                    held_out_cow_id=held_out_cow,
                    macro_f1=metrics["macro_f1"],
                    accuracy=metrics["accuracy"],
                    walking_recall=metrics["per_class"]["Walking"]["recall"],
                    unknown_recall=metrics["per_class"]["Other/Unknown"]["recall"],
                    latency_ms_per_window=metrics["latency_ms_per_window"],
                    train_windows=int(len(train_indices)),
                    test_windows=int(len(test_indices)),
                    best_params=json.dumps(best_params, sort_keys=True),
                )
                all_metrics.append(fold_metric)
                all_per_class_metrics.extend(
                    PerClassMetric(
                        model_name=model_name,
                        held_out_cow_id=held_out_cow,
                        behaviour_code=int(class_metric["code"]),
                        behaviour_name=class_name,
                        precision=float(class_metric["precision"]),
                        recall=float(class_metric["recall"]),
                        f1=float(class_metric["f1"]),
                        support=int(class_metric["support"]),
                    )
                    for class_name, class_metric in metrics["per_class"].items()
                )
                tracked_run.log({"fold": asdict(fold_metric), "per_class": metrics["per_class"]})
            finally:
                tracked_run.finish()

    _write_fold_metrics(output_dir / "fold_metrics.csv", all_metrics)
    _write_per_class_metrics(output_dir / "per_class_metrics.csv", all_per_class_metrics)
    for model_name, matrix in confusion_by_model.items():
        _write_confusion_matrix(output_dir / f"confusion_matrix_{model_name}.csv", matrix)

    per_model: dict[str, dict[str, float]] = {}
    for model_name in model_names:
        rows = [metric for metric in all_metrics if metric.model_name == model_name]
        per_model[model_name] = {
            "mean_macro_f1": float(np.mean([metric.macro_f1 for metric in rows])),
            "mean_accuracy": float(np.mean([metric.accuracy for metric in rows])),
            "mean_walking_recall": float(np.mean([metric.walking_recall for metric in rows])),
            "mean_unknown_recall": float(np.mean([metric.unknown_recall for metric in rows])),
            "mean_latency_ms_per_window": float(np.mean([metric.latency_ms_per_window for metric in rows])),
        }
    per_model_per_class = {
        model_name: _aggregate_per_class_metrics(confusion_by_model[model_name]) for model_name in model_names
    }
    selected_model_name = min(
        per_model,
        key=lambda name: (-per_model[name]["mean_macro_f1"], per_model[name]["mean_latency_ms_per_window"], name),
    )
    selected_summary = dict(per_model[selected_model_name])
    selected_summary["selected_model_name"] = selected_model_name
    selected_summary["release_gate_passed"] = bool(
        selected_summary["mean_macro_f1"] >= 0.85
        and selected_summary["mean_walking_recall"] >= 0.75
        and selected_summary["mean_unknown_recall"] >= 0.75
    )
    report = {
        "schema_version": 1,
        "benchmark_scope": "Public labelled-cow dataset benchmark; not target-collar or field validation.",
        "models": per_model,
        "per_class_metrics": per_model_per_class,
        "selected_model": selected_summary,
        "window_count": int(len(dataset.labels)),
        "cow_count": int(len(set(dataset.cow_ids.tolist()))),
        "raw_samples_embedded": False,
    }
    if selected_model_name == "cnn_1d":
        if not cnn_epoch_counts:
            raise RuntimeError("CNN was selected but no grouped inner-training epoch counts were recorded.")
        final_epochs = max(1, int(round(float(np.median(cnn_epoch_counts)))))
        final_model = _fit_cnn_final_model(
            raw_windows=dataset.raw_windows,
            labels=dataset.labels,
            seed=config.seed,
            epochs=final_epochs,
        )
        final_params = {
            "training_data": "all valid public windows",
            "epochs": final_epochs,
            "epoch_selection": "median grouped-inner early-stopping epochs across outer folds",
            "outer_fold_epoch_counts": cnn_epoch_counts,
        }
        import torch

        model_path = output_dir / "selected_model.pt"
        torch.save(
            {
                "state_dict": final_model.model.state_dict(),
                "mean": final_model.mean,
                "std": final_model.std,
                "behaviour_codes": BEHAVIOUR_CODES,
                "params": final_params,
            },
            model_path,
        )
    else:
        final_model, final_params = _train_final_classical(
            model_name=selected_model_name,
            dataset=dataset,
            imports=imports,
            config=config,
        )
        model_path = output_dir / "selected_model.pkl"
        with model_path.open("wb") as destination:
            pickle.dump(final_model, destination, protocol=pickle.HIGHEST_PROTOCOL)
    selected_summary["model_size_bytes"] = model_path.stat().st_size
    report["selected_model"] = selected_summary
    _write_json(output_dir / "benchmark_report.json", report)
    _write_json(
        output_dir / "selected_model_metadata.json",
        {
            "schema_version": 1,
            "model_name": selected_model_name,
            "model_file": model_path.name,
            "final_training_params": final_params,
            "behaviour_codes": list(BEHAVIOUR_CODES),
            "benchmark_scope": report["benchmark_scope"],
        },
    )
    model_card_path = output_dir / "model_card.md"
    _write_model_card(
        model_card_path,
        selected_model_name=selected_model_name,
        summary=selected_summary,
        manifest=manifest,
    )

    summary_run = tracker.start_summary(
        {
            "benchmark": run_config["benchmark"],
            "model_search": run_config["model_search"],
            "selected_model": selected_model_name,
            "dataset_combined_sha256": manifest["combined_sha256"],
            "benchmark_scope": report["benchmark_scope"],
        }
    )
    try:
        summary_run.log(
            {
                "models": per_model,
                "per_class_metrics": per_model_per_class,
                "selected_model": selected_summary,
            }
        )
        report_files = (
            output_dir / "dataset_manifest.json",
            output_dir / "feature_manifest.json",
            output_dir / "run_config.json",
            output_dir / "fold_metrics.csv",
            output_dir / "per_class_metrics.csv",
            output_dir / "benchmark_report.json",
            model_card_path,
        ) + tuple(output_dir / f"confusion_matrix_{model_name}.csv" for model_name in model_names)
        tracker.log_aggregate_artifact(
            summary_run,
            name="phase3-benchmark-report",
            artifact_type="benchmark-report",
            files=report_files,
            metadata={"raw_samples_embedded": False, "selected_model": selected_model_name},
            aliases=("latest",),
        )
        tracker.log_aggregate_artifact(
            summary_run,
            name="phase3-behaviour-model",
            artifact_type="model",
            files=(model_path, output_dir / "selected_model_metadata.json", model_card_path),
            metadata={"raw_samples_embedded": False, "selected_model": selected_model_name},
            aliases=("latest", "public-benchmark"),
        )
    finally:
        summary_run.finish()

    return BenchmarkResult(
        output_dir=output_dir,
        selected_model_name=selected_model_name,
        release_gate_passed=bool(selected_summary["release_gate_passed"]),
        mean_macro_f1=float(selected_summary["mean_macro_f1"]),
        walking_recall=float(selected_summary["mean_walking_recall"]),
        unknown_recall=float(selected_summary["mean_unknown_recall"]),
    )
