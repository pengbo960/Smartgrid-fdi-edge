from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import (
    BinaryClassificationMetrics,
    calculate_attack_type_recall,
    calculate_binary_metrics,
)
from src.evaluation.thresholds import (
    predictions_from_probabilities,
    search_best_threshold,
)
from src.training.prepare_dataset import PreparedDataset
from src.training.split_data import DatasetSplit


@dataclass(frozen=True)
class ModelComparisonResult:
    model_name: str
    feature_count: int
    selected_threshold: float
    metrics: BinaryClassificationMetrics
    attack_type_recall: dict[str, float]
    training_seconds: float
    inference_mean_ms: float
    inference_p95_ms: float
    model_size_mb: float

    def to_dict(self) -> dict[str, Any]:
        row: dict[str, Any] = {
            "model_name": self.model_name,
            "feature_count": self.feature_count,
            "selected_threshold": self.selected_threshold,
            **self.metrics.to_dict(),
            "training_seconds": self.training_seconds,
            "inference_mean_ms": self.inference_mean_ms,
            "inference_p95_ms": self.inference_p95_ms,
            "model_size_mb": self.model_size_mb,
        }
        for attack_type, recall in self.attack_type_recall.items():
            row[f"{attack_type}_recall"] = recall
        return row


def extract_comparison_arrays(
    prepared: PreparedDataset,
    split: DatasetSplit,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    columns = list(prepared.feature_columns)

    def extract(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        labels = pd.to_numeric(
            frame[prepared.target_column], errors="coerce"
        ).to_numpy(dtype=int)
        if not np.isfinite(values).all():
            raise ValueError("Comparison features contain non-finite values")
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Comparison labels must be binary")
        return values, labels

    x_train, y_train = extract(split.train)
    x_validation, y_validation = extract(split.validation)
    x_test, y_test = extract(split.test)
    return x_train, y_train, x_validation, y_validation, x_test, y_test


def measure_single_row_latency(
    model: Any,
    features: np.ndarray,
    sample_size: int,
) -> tuple[float, float]:
    if sample_size <= 0:
        raise ValueError("latency_sample_size must be greater than zero")
    count = min(sample_size, len(features))
    if count == 0:
        raise ValueError("Latency features must not be empty")

    model.predict_proba(features[:1])
    latencies: list[float] = []
    for row in features[:count]:
        started = perf_counter()
        model.predict_proba(row.reshape(1, -1))
        latencies.append((perf_counter() - started) * 1000.0)
    return float(np.mean(latencies)), float(np.percentile(latencies, 95))


def run_model_comparison(
    prepared: PreparedDataset,
    split: DatasetSplit,
    model_name: str,
    model_config: dict[str, Any],
    default_threshold: float,
    threshold_metric: str,
    latency_sample_size: int,
    model_path: str | Path,
) -> ModelComparisonResult:
    (
        x_train_raw,
        y_train,
        x_validation_raw,
        y_validation,
        x_test_raw,
        y_test,
    ) = extract_comparison_arrays(prepared, split)

    scaler: StandardScaler | None = None
    if model_name == "logistic_regression":
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw)
        x_validation = scaler.transform(x_validation_raw)
        x_test = scaler.transform(x_test_raw)
        model: Any = LogisticRegression(
            class_weight=model_config.get("class_weight", "balanced"),
            max_iter=int(model_config.get("max_iter", 1000)),
            random_state=int(model_config.get("random_seed", 42)),
        )
    elif model_name == "random_forest":
        x_train = x_train_raw
        x_validation = x_validation_raw
        x_test = x_test_raw
        model = RandomForestClassifier(
            n_estimators=int(model_config.get("n_estimators", 300)),
            max_depth=model_config.get("max_depth"),
            min_samples_leaf=int(model_config.get("min_samples_leaf", 1)),
            max_features=model_config.get("max_features", "sqrt"),
            class_weight=model_config.get("class_weight", "balanced"),
            n_jobs=int(model_config.get("training_n_jobs", -1)),
            random_state=int(model_config.get("random_seed", 42)),
        )
    else:
        raise ValueError(f"Unsupported comparison model: {model_name}")

    training_started = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - training_started

    if model_name == "random_forest":
        model.n_jobs = int(model_config.get("inference_n_jobs", 1))

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]
    best_threshold, _ = search_best_threshold(
        y_true=y_validation,
        probabilities=validation_probabilities,
        metric=threshold_metric,
        reference_threshold=default_threshold,
    )
    predictions = predictions_from_probabilities(
        probabilities=test_probabilities,
        threshold=best_threshold.threshold,
    )
    metrics = calculate_binary_metrics(
        y_true=y_test,
        y_pred=predictions,
        probabilities=test_probabilities,
    )
    attack_recall = calculate_attack_type_recall(
        attack_types=split.test["attack_type"].to_numpy(),
        y_true=y_test,
        y_pred=predictions,
    )
    mean_ms, p95_ms = measure_single_row_latency(
        model=model,
        features=x_test,
        sample_size=latency_sample_size,
    )

    output_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "feature_columns": prepared.feature_columns,
            "threshold": best_threshold.threshold,
        },
        output_path,
    )

    return ModelComparisonResult(
        model_name=model_name,
        feature_count=len(prepared.feature_columns),
        selected_threshold=float(best_threshold.threshold),
        metrics=metrics,
        attack_type_recall=attack_recall,
        training_seconds=float(training_seconds),
        inference_mean_ms=mean_ms,
        inference_p95_ms=p95_ms,
        model_size_mb=float(output_path.stat().st_size / (1024 ** 2)),
    )


def save_model_comparison(
    results: list[ModelComparisonResult],
    output_path: str | Path,
) -> pd.DataFrame:
    if not results:
        raise ValueError("Model comparison results must not be empty")
    frame = pd.DataFrame([result.to_dict() for result in results])
    frame = frame.sort_values(
        ["macro_f1", "recall", "false_positive_rate"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def save_model_comparison_figure(
    summary: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save a paper-ready effectiveness and edge-cost comparison."""
    required = {
        "model_name",
        "macro_f1",
        "recall",
        "false_positive_rate",
        "inference_mean_ms",
        "model_size_mb",
    }
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(
            "Comparison summary is missing figure columns: "
            f"{sorted(missing)}"
        )

    labels = summary["model_name"].replace(
        {
            "logistic_regression": "Logistic Regression",
            "random_forest": "Random Forest",
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        ("macro_f1", "Macro-F1", (0.99, 1.0)),
        ("false_positive_rate", "False-positive rate", None),
        ("inference_mean_ms", "Mean inference latency (ms)", None),
        ("model_size_mb", "Serialized model size (MB)", None),
    )
    colors = ["#2563eb", "#f97316"]
    for axis, (column, title, limits) in zip(axes.flat, panels):
        bars = axis.bar(labels, summary[column], color=colors)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.tick_params(axis="x", rotation=10)
        axis.bar_label(bars, fmt="%.4g", padding=3)

    figure.suptitle(
        "Logistic Regression vs Random Forest: Effectiveness and Edge Cost"
    )
    figure.tight_layout()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)
