from pathlib import Path

import numpy as np
import pytest

from src.evaluation.metrics import BinaryClassificationMetrics
from src.training.model_comparison import (
    ModelComparisonResult,
    measure_single_row_latency,
    save_model_comparison,
    save_model_comparison_figure,
)


class FakeModel:
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return np.tile([0.8, 0.2], (len(features), 1))


def metrics(macro_f1: float) -> BinaryClassificationMetrics:
    return BinaryClassificationMetrics(
        accuracy=0.9, precision=0.9, recall=0.9, f1=0.9,
        macro_f1=macro_f1, false_positive_rate=0.01,
        specificity=0.99, roc_auc=0.95, pr_auc=0.94,
        true_negative=99, false_positive=1,
        false_negative=2, true_positive=18,
    )


def result(name: str, macro_f1: float) -> ModelComparisonResult:
    return ModelComparisonResult(
        model_name=name, feature_count=48, selected_threshold=0.5,
        metrics=metrics(macro_f1), attack_type_recall={"random": 0.9},
        training_seconds=1.0, inference_mean_ms=0.1,
        inference_p95_ms=0.2, model_size_mb=1.0,
    )


def test_measure_latency() -> None:
    mean_ms, p95_ms = measure_single_row_latency(
        FakeModel(), np.ones((5, 2)), sample_size=3
    )
    assert mean_ms >= 0
    assert p95_ms >= 0


def test_invalid_latency_sample_size() -> None:
    with pytest.raises(ValueError, match="sample_size"):
        measure_single_row_latency(FakeModel(), np.ones((2, 2)), 0)


def test_save_comparison_sorts_by_macro_f1(tmp_path: Path) -> None:
    path = tmp_path / "comparison.csv"
    frame = save_model_comparison(
        [result("lower", 0.8), result("higher", 0.9)], path
    )
    assert path.exists()
    assert frame["model_name"].tolist() == ["higher", "lower"]


def test_save_comparison_figure(tmp_path: Path) -> None:
    summary = save_model_comparison(
        [result("logistic_regression", 0.8), result("random_forest", 0.9)],
        tmp_path / "comparison.csv",
    )
    path = tmp_path / "comparison.png"
    save_model_comparison_figure(summary, path)
    assert path.exists()
    assert path.stat().st_size > 0
