from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class BinaryClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    false_positive_rate: float
    specificity: float
    roc_auc: float
    pr_auc: float

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_binary_array(
    values: np.ndarray | list[int],
    field_name: str,
) -> np.ndarray:
    array = np.asarray(values)

    if array.ndim != 1:
        raise ValueError(
            f"{field_name} must be one-dimensional"
        )

    if array.size == 0:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    try:
        numeric = array.astype(int)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must contain binary values"
        ) from exc

    if not np.isin(numeric, [0, 1]).all():
        raise ValueError(
            f"{field_name} must contain only 0 and 1"
        )

    return numeric


def _as_probability_array(
    values: np.ndarray | list[float],
) -> np.ndarray:
    probabilities = np.asarray(
        values,
        dtype=float,
    )

    if probabilities.ndim != 1:
        raise ValueError(
            "probabilities must be one-dimensional"
        )

    if probabilities.size == 0:
        raise ValueError(
            "probabilities must not be empty"
        )

    if not np.isfinite(
        probabilities
    ).all():
        raise ValueError(
            "probabilities contain non-finite values"
        )

    if (
        (probabilities < 0)
        | (probabilities > 1)
    ).any():
        raise ValueError(
            "probabilities must be between 0 and 1"
        )

    return probabilities


def calculate_binary_metrics(
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
    probabilities: np.ndarray | list[float],
) -> BinaryClassificationMetrics:
    """
    Calculate binary attack-detection metrics.

    Label 1 is treated as the positive attack class.
    """
    true_labels = _as_binary_array(
        y_true,
        "y_true",
    )

    predicted_labels = _as_binary_array(
        y_pred,
        "y_pred",
    )

    probability_values = (
        _as_probability_array(
            probabilities
        )
    )

    if not (
        len(true_labels)
        == len(predicted_labels)
        == len(probability_values)
    ):
        raise ValueError(
            "y_true, y_pred and probabilities "
            "must have the same length"
        )

    if len(np.unique(true_labels)) < 2:
        raise ValueError(
            "y_true must contain both classes"
        )

    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=[0, 1],
    )

    true_negative = int(
        matrix[0, 0]
    )

    false_positive = int(
        matrix[0, 1]
    )

    false_negative = int(
        matrix[1, 0]
    )

    true_positive = int(
        matrix[1, 1]
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

    specificity = (
        true_negative / negative_count
        if negative_count > 0
        else 0.0
    )

    return BinaryClassificationMetrics(
        accuracy=float(
            accuracy_score(
                true_labels,
                predicted_labels,
            )
        ),
        precision=float(
            precision_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        recall=float(
            recall_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                zero_division=0,
            )
        ),
        macro_f1=float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        ),
        false_positive_rate=float(
            false_positive_rate
        ),
        specificity=float(
            specificity
        ),
        roc_auc=float(
            roc_auc_score(
                true_labels,
                probability_values,
            )
        ),
        pr_auc=float(
            average_precision_score(
                true_labels,
                probability_values,
            )
        ),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
    )


def calculate_attack_type_recall(
    attack_types: np.ndarray | list[str],
    y_true: np.ndarray | list[int],
    y_pred: np.ndarray | list[int],
) -> dict[str, float]:
    """
    Calculate detection recall separately for each attack type.

    Rows labelled none are ignored.
    """
    attack_type_array = np.asarray(
        attack_types,
        dtype=str,
    )

    true_labels = _as_binary_array(
        y_true,
        "y_true",
    )

    predicted_labels = _as_binary_array(
        y_pred,
        "y_pred",
    )

    if not (
        len(attack_type_array)
        == len(true_labels)
        == len(predicted_labels)
    ):
        raise ValueError(
            "attack_types, y_true and y_pred "
            "must have the same length"
        )

    result: dict[str, float] = {}

    for attack_type in sorted(
        set(attack_type_array)
        - {"none"}
    ):
        mask = (
            attack_type_array
            == attack_type
        )

        attack_true = true_labels[
            mask
        ]

        attack_predicted = (
            predicted_labels[mask]
        )

        if len(attack_true) == 0:
            continue

        if not (
            attack_true == 1
        ).all():
            raise ValueError(
                f"Attack type {attack_type} "
                "contains non-attack labels"
            )

        result[attack_type] = float(
            recall_score(
                attack_true,
                attack_predicted,
                zero_division=0,
            )
        )

    return result

def save_metrics_report(
    report: dict[str, Any],
    output_path: str | Path,
) -> None:
    """
    Save a metrics dictionary as formatted JSON.
    """
    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )