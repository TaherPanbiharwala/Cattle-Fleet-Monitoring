"""Nested leave-one-cow-out XGBoost benchmark for public WASP behavior data."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from .artifact import legacy_pickle_provenance, save_model_artifact
from .errors import fail
from .features import FEATURE_NAMES, build_feature_dataset
from .io import assert_new_output_dir, atomic_output_dir, write_json, write_jsonl
from .wasp import CLASS_NAMES, PLATFORM_CODES, dataset_provenance, load_dataset

SEED: Final[int] = 42
INNER_SPLITS: Final[int] = 5
BOOTSTRAP_RESAMPLES: Final[int] = 1000
XGBOOST_GRID: Final[tuple[dict[str, object], ...]] = tuple(
    {
        "n_estimators": estimators,
        "max_depth": depth,
        "learning_rate": rate,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    for estimators in (200, 300)
    for depth in (4, 6)
    for rate in (0.03, 0.05)
)


@dataclass(frozen=True)
class BenchmarkResult:
    output_dir: Path
    artifact_eligible: bool
    primary_mean_macro_f1: float
    pooled_oof_macro_f1: float


def _imports() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        import xgboost as xgb
        from sklearn.metrics import confusion_matrix, f1_score
        from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
    except ImportError as exc:  # pragma: no cover - optional-install boundary
        fail("DEPENDENCY_MISSING", "Benchmarking requires the behavior extra.", install="python -m pip install -e '.[behavior,dev]'")
        raise AssertionError from exc
    return np, xgb, f1_score, confusion_matrix, (GroupKFold, LeaveOneGroupOut)


def _model(xgb: Any, parameters: dict[str, object], seed: int) -> Any:
    return xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        eval_metric="mlogloss",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
        **parameters,
    )


def _assert_training_classes(labels: Any, *, stage: str, held_out_cow: str) -> None:
    missing = sorted(set(range(4)) - set(int(value) for value in labels.tolist()))
    if missing:
        fail(
            "INSUFFICIENT_CLASS_SUPPORT",
            "A grouped training partition does not contain all four behavior classes.",
            stage=stage,
            held_out_cow=held_out_cow,
            missing_internal_classes=missing,
        )


def _macro_f1_present(f1_score: Any, expected: Any, predicted: Any) -> float:
    labels = sorted(set(int(value) for value in expected.tolist()))
    return float(f1_score(expected, predicted, labels=labels, average="macro", zero_division=0))


def _candidate_sort_key(score: float, parameters: dict[str, object]) -> tuple[float, float, float, float]:
    # Higher score first; then favor the simpler model deterministically.
    return (score, -float(parameters["max_depth"]), -float(parameters["n_estimators"]), -float(parameters["learning_rate"]))


def _choose_parameters(x: Any, y: Any, groups: Any, *, held_out_cow: str, seed: int) -> tuple[dict[str, object], float]:
    np, xgb, f1_score, _, splitters = _imports()
    GroupKFold, _ = splitters
    unique_groups = sorted(set(groups.tolist()))
    if len(unique_groups) < 2:
        fail("SPLIT_INTEGRITY", "Nested LOCO needs at least two cows in every outer training partition.", held_out_cow=held_out_cow)
    splits = min(INNER_SPLITS, len(unique_groups))
    splitter = GroupKFold(n_splits=splits)
    best: tuple[dict[str, object], float] | None = None
    for parameters in XGBOOST_GRID:
        scores: list[float] = []
        for inner_train, inner_test in splitter.split(x, y, groups):
            train_groups = set(groups[inner_train].tolist())
            test_groups = set(groups[inner_test].tolist())
            if train_groups.intersection(test_groups):
                fail("SPLIT_INTEGRITY", "Cow leakage detected in an inner grouped split.", held_out_cow=held_out_cow)
            _assert_training_classes(y[inner_train], stage="inner", held_out_cow=held_out_cow)
            fitted = _model(xgb, parameters, seed)
            fitted.fit(x[inner_train], y[inner_train])
            scores.append(_macro_f1_present(f1_score, y[inner_test], fitted.predict(x[inner_test])))
        score = statistics.fmean(scores)
        if best is None or _candidate_sort_key(score, parameters) > _candidate_sort_key(best[1], best[0]):
            best = (dict(parameters), score)
    assert best is not None
    return best


def _class_metrics(expected: Any, predicted: Any) -> dict[str, dict[str, float | int | None]]:
    result: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(CLASS_NAMES):
        support = int((expected == index).sum())
        predicted_count = int((predicted == index).sum())
        true_positive = int(((expected == index) & (predicted == index)).sum())
        if support == 0:
            result[name] = {"support": 0, "precision": None, "recall": None, "f1": None}
            continue
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {"support": support, "precision": precision, "recall": recall, "f1": f1}
    return result


def _bootstrap_interval(np: Any, fold_scores: list[float]) -> dict[str, float]:
    generator = np.random.default_rng(SEED)
    values = np.asarray(fold_scores, dtype=np.float64)
    samples = np.asarray([generator.choice(values, size=len(values), replace=True).mean() for _ in range(BOOTSTRAP_RESAMPLES)])
    return {"resamples": BOOTSTRAP_RESAMPLES, "lower_95": float(np.quantile(samples, 0.025)), "upper_95": float(np.quantile(samples, 0.975))}


def preflight(dataset_dir: Path) -> dict[str, object]:
    """Validate data and calculate readiness without outputting IMU data or features."""

    events = load_dataset(dataset_dir)
    data = build_feature_dataset(events)
    cow_count = len(set(data.cow_ids.tolist()))
    event_counts = {name: sum(event.label == name for event in events) for name in CLASS_NAMES}
    window_counts = {name: int((data.labels == index).sum()) for index, name in enumerate(CLASS_NAMES)}
    if cow_count < 3:
        fail("INSUFFICIENT_CLASS_SUPPORT", "Nested LOCO requires at least three cows.", cow_count=cow_count)
    for index, name in enumerate(CLASS_NAMES):
        cows = set(data.cow_ids[data.labels == index].tolist())
        if len(cows) < 2:
            fail("INSUFFICIENT_CLASS_SUPPORT", "Each class must occur in at least two cows for grouped evaluation.", behavior_state=name, cow_count=len(cows))
    return {
        "schema_version": 1,
        "status": "validated",
        "dataset": dataset_provenance(dataset_dir),
        "cow_count": cow_count,
        "event_counts": event_counts,
        "valid_window_counts": window_counts,
        "valid_window_total": int(len(data.labels)),
        "timing_gap_or_short_segments_discarded": data.gap_segments_discarded,
        "validation": "nested leave-one-cow-out; public benchmark only",
        "raw_sensor_rows_persisted": False,
    }


def run_benchmark(dataset_dir: Path, output_dir: Path, *, legacy_pickle: Path | None = None) -> BenchmarkResult:
    """Evaluate nested LOCO and conditionally publish a native benchmark artifact."""

    assert_new_output_dir(output_dir)
    readiness = preflight(dataset_dir)
    events = load_dataset(dataset_dir)
    dataset = build_feature_dataset(events)
    np, xgb, f1_score, confusion_matrix, splitters = _imports()
    _, LeaveOneGroupOut = splitters
    folds: list[dict[str, object]] = []
    all_expected: list[Any] = []
    all_predicted: list[Any] = []
    outer = LeaveOneGroupOut()
    for ordinal, (train, test) in enumerate(outer.split(dataset.features, dataset.labels, dataset.cow_ids), start=1):
        train_cows = set(dataset.cow_ids[train].tolist())
        test_cows = set(dataset.cow_ids[test].tolist())
        if train_cows.intersection(test_cows) or len(test_cows) != 1:
            fail("SPLIT_INTEGRITY", "Outer LOCO split leaked a cow or held out an invalid group.")
        cow_id = str(dataset.cow_ids[test][0])
        _assert_training_classes(dataset.labels[train], stage="outer", held_out_cow=cow_id)
        parameters, inner_score = _choose_parameters(dataset.features[train], dataset.labels[train], dataset.cow_ids[train], held_out_cow=cow_id, seed=SEED + ordinal)
        fitted = _model(xgb, parameters, SEED + ordinal)
        fitted.fit(dataset.features[train], dataset.labels[train])
        predicted = fitted.predict(dataset.features[test]).astype(np.int64)
        scores = _class_metrics(dataset.labels[test], predicted)
        folds.append({
            "schema_version": 1,
            "held_out_cow_id": cow_id,
            "train_cow_count": len(train_cows),
            "test_window_count": int(len(test)),
            "macro_f1_present_classes": _macro_f1_present(f1_score, dataset.labels[test], predicted),
            "inner_mean_macro_f1": inner_score,
            "selected_params": parameters,
            "class_metrics": scores,
        })
        all_expected.append(dataset.labels[test])
        all_predicted.append(predicted)
    expected = np.concatenate(all_expected)
    predicted = np.concatenate(all_predicted)
    primary = statistics.fmean(float(fold["macro_f1_present_classes"]) for fold in folds)
    pooled = float(f1_score(expected, predicted, labels=list(range(4)), average="macro", zero_division=0))
    aggregate_classes = _class_metrics(expected, predicted)
    walking_recall = aggregate_classes["walking"]["recall"]
    miscellaneous_recall = aggregate_classes["miscellaneous"]["recall"]
    assert isinstance(walking_recall, float) and isinstance(miscellaneous_recall, float)
    eligible = primary >= 0.85 and walking_recall >= 0.75 and miscellaneous_recall >= 0.75
    final_parameters, final_inner_score = _choose_parameters(dataset.features, dataset.labels, dataset.cow_ids, held_out_cow="none-final-refit", seed=SEED)
    matrix = confusion_matrix(expected, predicted, labels=list(range(4))).tolist()
    uncertainty = _bootstrap_interval(np, [float(fold["macro_f1_present_classes"]) for fold in folds])
    report = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_protocol": "nested leave-one-cow-out; each fold macro F1 averages only classes supported by its held-out cow",
        "primary_mean_outer_fold_macro_f1": primary,
        "pooled_out_of_fold_macro_f1": pooled,
        "aggregate_class_metrics": aggregate_classes,
        "artifact_eligibility": {
            "passed": eligible,
            "criteria": {"mean_outer_fold_macro_f1_min": 0.85, "walking_recall_min": 0.75, "miscellaneous_recall_min": 0.75},
            "benchmark_only": True,
            "field_deployment_validated": False,
        },
        "random_split_reference": {"value": 0.9625, "meaning": "published random-split reference only; not comparable to or used by the LOCO gate"},
        "final_refit_inner_selection": {"selected_params": final_parameters, "inner_mean_macro_f1": final_inner_score},
        "confidence": "uncalibrated_max_class_probability",
        "legacy_pickle_provenance": legacy_pickle_provenance(legacy_pickle),
        "preflight": {key: readiness[key] for key in ("cow_count", "event_counts", "valid_window_counts", "valid_window_total", "timing_gap_or_short_segments_discarded")},
        "raw_sensor_rows_persisted": False,
    }

    provenance = dataset_provenance(dataset_dir)

    def writer(temp_dir: Path) -> None:
        write_jsonl(temp_dir / "fold_metrics.jsonl", folds)
        write_json(temp_dir / "benchmark_report.json", report)
        write_json(temp_dir / "class_metrics.json", aggregate_classes)
        write_json(temp_dir / "confusion_matrix.json", {"internal_class_order": list(CLASS_NAMES), "matrix": matrix})
        write_json(temp_dir / "uncertainty.json", {"metric": "mean_outer_fold_macro_f1", **uncertainty})
        write_json(temp_dir / "dataset_provenance.json", provenance)
        if eligible:
            fitted = _model(xgb, final_parameters, SEED)
            fitted.fit(dataset.features, dataset.labels)
            save_model_artifact(fitted, temp_dir, dataset_provenance=provenance, selected_params=final_parameters, seed=SEED, legacy_pickle=legacy_pickle)

    published = atomic_output_dir(output_dir, writer)
    return BenchmarkResult(published, eligible, primary, pooled)
