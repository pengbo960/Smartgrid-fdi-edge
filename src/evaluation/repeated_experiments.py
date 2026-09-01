from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


def validate_seeds(raw_seeds: Iterable[Any]) -> tuple[int, ...]:
    """Return a non-empty sequence of unique integer random seeds."""
    seeds: list[int] = []
    for raw_seed in raw_seeds:
        if isinstance(raw_seed, bool) or not isinstance(raw_seed, int):
            raise TypeError("Every repeated-experiment seed must be an integer")
        seeds.append(raw_seed)
    if not seeds:
        raise ValueError("At least one repeated-experiment seed is required")
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated-experiment seeds must be unique")
    return tuple(seeds)


def validate_folds(
    raw_folds: Iterable[Any],
    fold_count: int,
) -> tuple[int, ...]:
    """Return unique one-based fold numbers within the configured range."""
    if isinstance(fold_count, bool) or not isinstance(fold_count, int):
        raise TypeError("fold_count must be an integer")
    if fold_count < 3:
        raise ValueError("fold_count must be at least three")

    folds: list[int] = []
    for raw_fold in raw_folds:
        if isinstance(raw_fold, bool) or not isinstance(raw_fold, int):
            raise TypeError("Every fold number must be an integer")
        if not 1 <= raw_fold <= fold_count:
            raise ValueError(
                f"Fold numbers must be between 1 and {fold_count}"
            )
        folds.append(raw_fold)
    if not folds:
        raise ValueError("At least one fold is required")
    if len(folds) != len(set(folds)):
        raise ValueError("Fold numbers must be unique")
    return tuple(folds)


def _configure_split(
    config: dict[str, Any],
    seed: int,
    fold_index: int | None,
    validation_offset: int,
) -> None:
    config["split"]["random_seed"] = seed
    if fold_index is not None:
        config["split"].update(
            {
                "strategy": "grouped_kfold",
                "fold_index": fold_index,
                "validation_offset": validation_offset,
            }
        )


def configure_ablation_run(
    base_config: dict[str, Any],
    seed: int,
    workspace: str | Path,
    fold_index: int | None = None,
    validation_offset: int = 1,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    root = Path(workspace)
    _configure_split(config, seed, fold_index, validation_offset)
    config["model"]["random_seed"] = seed
    config["output"] = {
        "metrics_directory": str(root / "metrics"),
        "predictions_directory": str(root / "predictions"),
        "figures_directory": str(root / "figures"),
        "summary_csv": str(root / "summary.csv"),
    }
    return config


def configure_model_comparison_run(
    base_config: dict[str, Any],
    seed: int,
    workspace: str | Path,
    fold_index: int | None = None,
    validation_offset: int = 1,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    root = Path(workspace)
    _configure_split(config, seed, fold_index, validation_offset)
    for model_config in config["models"].values():
        model_config["random_seed"] = seed
    config["output"] = {
        "directory": str(root),
        "summary_csv": str(root / "summary.csv"),
        "comparison_figure": str(root / "comparison.png"),
        "logistic_model": str(root / "logistic.joblib"),
        "random_forest_model": str(root / "random_forest.joblib"),
    }
    return config


def configure_open_set_run(
    base_config: dict[str, Any],
    seed: int,
    workspace: str | Path,
    fold_index: int | None = None,
    validation_offset: int = 1,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    root = Path(workspace)
    _configure_split(config, seed, fold_index, validation_offset)
    config["model"]["random_seed"] = seed
    config["output"] = {
        "classifier_path": str(root / "classifier.joblib"),
        "scaler_path": str(root / "scaler.joblib"),
        "anomaly_detector_path": str(root / "anomaly_detector.joblib"),
        "metadata_path": str(root / "metadata.json"),
        "metrics_path": str(root / "metrics.json"),
        "predictions_path": str(root / "predictions.csv"),
    }
    return config


def extract_open_set_row(
    report: dict[str, Any],
    seed: int,
    fold: int | None = None,
) -> dict[str, Any]:
    """Flatten the thesis-relevant scalar open-set metrics."""
    thresholds = report["thresholds"]
    closed = report["known_closed_set"]
    known = report["known_open_set"]
    unseen = report["unseen"]
    row: dict[str, Any] = {
        "seed": seed,
        "confidence_threshold": thresholds["confidence_threshold"],
        "anomaly_threshold": thresholds["anomaly_threshold"],
        "known_closed_set_accuracy": closed["accuracy"],
        "known_closed_set_macro_f1": closed["macro_f1"],
        "false_unknown_rate": known["false_unknown_rate"],
        "known_acceptance_rate": known["acceptance_rate"],
        "known_overall_correct_rate": known["overall_correct_rate"],
        "unknown_recall": unseen["unknown_recall"],
        "unknown_precision": unseen["unknown_precision"],
        "confidence_only_recall": unseen["confidence_only_recall"],
        "normal_anomaly_recall": unseen["normal_anomaly_recall"],
        "mean_first_unknown_step": unseen["mean_first_unknown_step"],
    }
    if fold is not None:
        row["fold"] = fold
    for metric in (
        "unknown_true_positive",
        "unknown_false_positive",
    ):
        if metric in unseen:
            row[metric] = unseen[metric]
    if "unseen_test_rows" in report.get("dataset", {}):
        row["unseen_test_rows"] = report["dataset"]["unseen_test_rows"]
    if "unseen_source_files" in report.get("dataset", {}):
        row["unseen_source_files"] = ";".join(
            str(source_file)
            for source_file in report["dataset"]["unseen_source_files"]
        )
    for label, recall in known.get("per_class_recall", {}).items():
        row[f"known_{label}_recall"] = recall
    return row


def aggregate_repeated_runs(
    runs: pd.DataFrame,
    group_columns: Iterable[str],
    metric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate repeated runs as mean, sample SD, minimum and maximum."""
    groups = list(group_columns)
    missing_groups = set(groups) - set(runs.columns)
    if missing_groups:
        raise ValueError(f"Missing grouping columns: {sorted(missing_groups)}")
    if runs.empty:
        raise ValueError("Repeated experiment runs must not be empty")

    if metric_columns is None:
        metrics = [
            column for column in runs.select_dtypes(include="number").columns
            if column not in {"seed", "fold", "model_seed"}
            and column not in groups
        ]
    else:
        metrics = list(metric_columns)
        missing_metrics = set(metrics) - set(runs.columns)
        if missing_metrics:
            raise ValueError(f"Missing metric columns: {sorted(missing_metrics)}")
    if not metrics:
        raise ValueError("At least one numeric metric column is required")

    grouped = runs.groupby(groups, dropna=False, sort=False)
    summary = grouped[metrics].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    summary.insert(len(groups), "runs", grouped.size().to_numpy())
    return summary


def aggregate_binary_confusion_matrix(
    runs: pd.DataFrame,
    experiment_name: str,
) -> np.ndarray:
    """Sum binary confusion counts for one experiment across grouped folds."""
    required_columns = {
        "experiment_name",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }
    missing_columns = required_columns - set(runs.columns)
    if missing_columns:
        raise ValueError(
            "Missing confusion-matrix columns: "
            f"{sorted(missing_columns)}"
        )

    selected = runs[
        runs["experiment_name"].astype(str).eq(experiment_name)
    ]
    if selected.empty:
        raise ValueError(f"No runs found for experiment: {experiment_name}")

    count_columns = [
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    ]
    counts = selected[count_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    if counts.isna().any().any() or (counts < 0).any().any():
        raise ValueError(
            "Confusion-matrix counts must be numeric and non-negative"
        )
    if not np.equal(counts.to_numpy(), np.floor(counts.to_numpy())).all():
        raise ValueError("Confusion-matrix values must be whole counts")

    totals = counts.sum().astype(int)
    return np.asarray(
        [
            [totals["true_negative"], totals["false_positive"]],
            [totals["false_negative"], totals["true_positive"]],
        ],
        dtype=int,
    )


def save_table(frame: pd.DataFrame, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
