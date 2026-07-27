from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class ThresholdEvaluation:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    false_positive_rate: float


def predictions_from_probabilities(
    probabilities: np.ndarray | list[float],
    threshold: float,
) -> np.ndarray:
    if not 0 <= threshold <= 1:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    )

    if probability_array.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional"
        )

    if probability_array.size == 0:
        raise ValueError(
            "probabilities must not be empty"
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "probabilities contain non-finite values"
        )

    if (
        (probability_array < 0)
        | (probability_array > 1)
    ).any():
        raise ValueError(
            "probabilities must be between 0 and 1"
        )

    return (
        probability_array >= threshold
    ).astype(int)


def evaluate_threshold(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    threshold: float,
) -> ThresholdEvaluation:
    true_labels = np.asarray(
        y_true,
        dtype=int,
    )

    predictions = (
        predictions_from_probabilities(
            probabilities=probabilities,
            threshold=threshold,
        )
    )

    if true_labels.ndim != 1:
        raise ValueError(
            "y_true must be one-dimensional"
        )

    if len(true_labels) != len(
        predictions
    ):
        raise ValueError(
            "y_true and probabilities must have the same length"
        )

    if not np.isin(
        true_labels,
        [0, 1],
    ).all():
        raise ValueError(
            "y_true must contain only 0 and 1"
        )

    true_negative = int(
        (
            (true_labels == 0)
            & (predictions == 0)
        ).sum()
    )

    false_positive = int(
        (
            (true_labels == 0)
            & (predictions == 1)
        ).sum()
    )

    negative_count = (
        true_negative
        + false_positive
    )

    false_positive_rate = (
        false_positive / negative_count
        if negative_count > 0
        else 0.0
    )

    return ThresholdEvaluation(
        threshold=float(
            threshold
        ),
        accuracy=float(
            accuracy_score(
                true_labels,
                predictions,
            )
        ),
        precision=float(
            precision_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                true_labels,
                predictions,
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                true_labels,
                predictions,
                average="macro",
                zero_division=0,
            )
        ),
        false_positive_rate=float(
            false_positive_rate
        ),
    )


def search_best_threshold(
    y_true: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
    metric: str = "macro_f1",
    thresholds: np.ndarray | None = None,
    reference_threshold: float = 0.5,
) -> tuple[
    ThresholdEvaluation,
    list[ThresholdEvaluation],
]:
    if not 0 <= reference_threshold <= 1:
        raise ValueError(
            "reference_threshold must be between 0 and 1"
        )
    supported_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
    }

    if metric not in supported_metrics:
        raise ValueError(
            f"Unsupported threshold metric: {metric}"
        )

    if thresholds is None:
        thresholds = np.linspace(
            0.01,
            0.99,
            99,
        )

    evaluations = [
        evaluate_threshold(
            y_true=y_true,
            probabilities=probabilities,
            threshold=float(
                threshold
            ),
        )
        for threshold in thresholds
    ]

    best = max(
        evaluations,
        key=lambda item: (
            getattr(
                item,
                metric,
            ),
            -item.false_positive_rate,
            -abs(
                item.threshold
                - reference_threshold
            ),
            -item.threshold,
        ),
    )

    return best, evaluations