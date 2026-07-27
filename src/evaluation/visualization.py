from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.evaluation.thresholds import (
    ThresholdEvaluation,
)


def _prepare_output_path(
    output_path: str | Path,
) -> Path:
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    path = _prepare_output_path(
        output_path
    )

    display = (
        ConfusionMatrixDisplay
        .from_predictions(
            y_true,
            y_pred,
            labels=[
                0,
                1,
            ],
            display_labels=[
                "Normal",
                "Attack",
            ],
            values_format="d",
        )
    )

    display.ax_.set_title(
        title
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        display.figure_
    )


def save_roc_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
    title: str = "ROC Curve",
) -> None:
    path = _prepare_output_path(
        output_path
    )

    display = RocCurveDisplay.from_predictions(
        y_true,
        probabilities,
    )

    display.ax_.set_title(
        title
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        display.figure_
    )


def save_precision_recall_curve(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path,
    title: str = "Precision-Recall Curve",
) -> None:
    path = _prepare_output_path(
        output_path
    )

    display = (
        PrecisionRecallDisplay
        .from_predictions(
            y_true,
            probabilities,
        )
    )

    display.ax_.set_title(
        title
    )

    display.figure_.tight_layout()

    display.figure_.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        display.figure_
    )


def save_threshold_curve(
    evaluations: Iterable[
        ThresholdEvaluation
    ],
    selected_threshold: float,
    output_path: str | Path,
) -> None:
    path = _prepare_output_path(
        output_path
    )

    evaluations = list(
        evaluations
    )

    if not evaluations:
        raise ValueError(
            "evaluations must not be empty"
        )

    thresholds = [
        item.threshold
        for item in evaluations
    ]

    macro_f1 = [
        item.macro_f1
        for item in evaluations
    ]

    recall = [
        item.recall
        for item in evaluations
    ]

    false_positive_rate = [
        item.false_positive_rate
        for item in evaluations
    ]

    figure, axis = plt.subplots()

    axis.plot(
        thresholds,
        macro_f1,
        label="Macro F1",
    )

    axis.plot(
        thresholds,
        recall,
        label="Attack Recall",
    )

    axis.plot(
        thresholds,
        false_positive_rate,
        label="False Positive Rate",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            "Selected threshold "
            f"{selected_threshold:.2f}"
        ),
    )

    axis.set_xlabel(
        "Decision Threshold"
    )

    axis.set_ylabel(
        "Metric Value"
    )

    axis.set_title(
        "Validation Threshold Evaluation"
    )

    axis.legend()
    figure.tight_layout()

    figure.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )