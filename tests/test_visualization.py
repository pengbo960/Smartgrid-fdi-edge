import numpy as np

from src.evaluation.thresholds import (
    search_best_threshold,
)
from src.evaluation.visualization import (
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_curve,
)


def test_visualizations_are_saved(
    tmp_path,
) -> None:
    y_true = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    y_pred = np.array(
        [
            0,
            0,
            1,
            1,
        ]
    )

    probabilities = np.array(
        [
            0.1,
            0.2,
            0.8,
            0.9,
        ]
    )

    confusion_path = (
        tmp_path
        / "confusion.png"
    )

    roc_path = (
        tmp_path
        / "roc.png"
    )

    pr_path = (
        tmp_path
        / "pr.png"
    )

    threshold_path = (
        tmp_path
        / "threshold.png"
    )

    best, evaluations = (
        search_best_threshold(
            y_true=y_true,
            probabilities=probabilities,
        )
    )

    save_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
        output_path=confusion_path,
    )

    save_roc_curve(
        y_true=y_true,
        probabilities=probabilities,
        output_path=roc_path,
    )

    save_precision_recall_curve(
        y_true=y_true,
        probabilities=probabilities,
        output_path=pr_path,
    )

    save_threshold_curve(
        evaluations=evaluations,
        selected_threshold=(
            best.threshold
        ),
        output_path=threshold_path,
    )

    for path in [
        confusion_path,
        roc_path,
        pr_path,
        threshold_path,
    ]:
        assert path.exists()
        assert path.stat().st_size > 0