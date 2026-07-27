import numpy as np
import pytest

from src.evaluation.thresholds import (
    evaluate_threshold,
    predictions_from_probabilities,
    search_best_threshold,
)


def test_predictions_use_threshold() -> None:
    probabilities = np.array(
        [
            0.1,
            0.49,
            0.5,
            0.9,
        ]
    )

    predictions = (
        predictions_from_probabilities(
            probabilities=probabilities,
            threshold=0.5,
        )
    )

    assert predictions.tolist() == [
        0,
        0,
        1,
        1,
    ]


def test_threshold_evaluation() -> None:
    evaluation = evaluate_threshold(
        y_true=[
            0,
            0,
            1,
            1,
        ],
        probabilities=[
            0.1,
            0.8,
            0.9,
            0.4,
        ],
        threshold=0.5,
    )

    assert evaluation.accuracy == 0.5
    assert evaluation.precision == 0.5
    assert evaluation.recall == 0.5
    assert (
        evaluation.false_positive_rate
        == 0.5
    )


def test_search_best_threshold_finds_perfect_split() -> None:
    best, evaluations = (
        search_best_threshold(
            y_true=[
                0,
                0,
                1,
                1,
            ],
            probabilities=[
                0.1,
                0.2,
                0.8,
                0.9,
            ],
            metric="macro_f1",
            thresholds=np.array(
                [
                    0.2,
                    0.5,
                    0.8,
                ]
            ),
        )
    )

    assert best.threshold == 0.5
    assert best.macro_f1 == 1.0
    assert len(evaluations) == 3


def test_invalid_threshold_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="threshold",
    ):
        predictions_from_probabilities(
            probabilities=[
                0.1,
                0.9,
            ],
            threshold=1.1,
        )


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported threshold metric",
    ):
        search_best_threshold(
            y_true=[
                0,
                1,
            ],
            probabilities=[
                0.1,
                0.9,
            ],
            metric="unknown",
        )


def test_threshold_tie_prefers_reference_threshold() -> None:
    best, _ = search_best_threshold(
        y_true=[
            0,
            0,
            1,
            1,
        ],
        probabilities=[
            0.1,
            0.2,
            0.8,
            0.9,
        ],
        metric="macro_f1",
        thresholds=np.array(
            [
                0.5,
                0.8,
            ]
        ),
        reference_threshold=0.5,
    )

    assert best.threshold == 0.5


def test_invalid_reference_threshold_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="reference_threshold",
    ):
        search_best_threshold(
            y_true=[
                0,
                1,
            ],
            probabilities=[
                0.1,
                0.9,
            ],
            reference_threshold=1.1,
        )