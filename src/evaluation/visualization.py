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


def save_confusion_matrix_from_counts(
    confusion_matrix: np.ndarray,
    output_path: str | Path,
    title: str = "Confusion Matrix",
) -> None:
    """Save a binary confusion matrix that has already been aggregated."""
    matrix = np.asarray(confusion_matrix)
    if matrix.shape != (2, 2):
        raise ValueError("confusion_matrix must have shape (2, 2)")
    if not np.issubdtype(matrix.dtype, np.number):
        raise TypeError("confusion_matrix values must be numeric")
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError(
            "confusion_matrix values must be finite and non-negative"
        )
    if not np.equal(matrix, np.floor(matrix)).all():
        raise ValueError("confusion_matrix values must be whole counts")

    matrix = matrix.astype(int)
    row_totals = matrix.sum(axis=1, keepdims=True)
    proportions = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )

    path = _prepare_output_path(output_path)
    figure, axis = plt.subplots(figsize=(7.2, 5.8))
    image = axis.imshow(
        proportions,
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    labels = ["Normal", "Attack"]
    axis.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    for row in range(2):
        for column in range(2):
            proportion = proportions[row, column]
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}\n({proportion:.2%})",
                ha="center",
                va="center",
                color="white" if proportion > 0.5 else "#1F2937",
                fontsize=12,
                fontweight="bold",
            )

    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label("Proportion within true class")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)


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
