import numpy as np
import pytest
import json

from src.evaluation.metrics import (
    calculate_attack_type_recall,
    calculate_binary_metrics,
    save_metrics_report,
)


def test_binary_metrics_are_calculated() -> None:
    y_true = np.array(
        [0, 0, 0, 1, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1, 0, 1]
    )

    probabilities = np.array(
        [
            0.1,
            0.7,
            0.2,
            0.9,
            0.4,
            0.8,
        ]
    )

    metrics = calculate_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    assert metrics.true_negative == 2
    assert metrics.false_positive == 1
    assert metrics.false_negative == 1
    assert metrics.true_positive == 2

    assert metrics.accuracy == pytest.approx(
        4 / 6
    )

    assert metrics.precision == pytest.approx(
        2 / 3
    )

    assert metrics.recall == pytest.approx(
        2 / 3
    )

    assert metrics.false_positive_rate == (
        pytest.approx(
            1 / 3
        )
    )

    assert metrics.specificity == (
        pytest.approx(
            2 / 3
        )
    )


def test_metrics_dictionary_contains_all_fields() -> None:
    metrics = calculate_binary_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        probabilities=[
            0.1,
            0.2,
            0.8,
            0.9,
        ],
    )

    result = metrics.to_dict()

    expected_fields = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "false_positive_rate",
        "specificity",
        "roc_auc",
        "pr_auc",
        "true_negative",
        "false_positive",
        "false_negative",
        "true_positive",
    }

    assert set(result.keys()) == (
        expected_fields
    )


def test_perfect_predictions_have_perfect_metrics() -> None:
    metrics = calculate_binary_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
        probabilities=[
            0.05,
            0.1,
            0.9,
            0.95,
        ],
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.false_positive_rate == 0.0
    assert metrics.specificity == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0


def test_attack_type_recall() -> None:
    attack_types = np.array(
        [
            "none",
            "constant",
            "constant",
            "random",
            "random",
        ]
    )

    y_true = np.array(
        [0, 1, 1, 1, 1]
    )

    y_pred = np.array(
        [0, 1, 0, 1, 1]
    )

    result = calculate_attack_type_recall(
        attack_types=attack_types,
        y_true=y_true,
        y_pred=y_pred,
    )

    assert result["constant"] == 0.5
    assert result["random"] == 1.0
    assert "none" not in result


def test_metric_inputs_must_have_equal_length() -> None:
    with pytest.raises(
        ValueError,
        match="same length",
    ):
        calculate_binary_metrics(
            y_true=[0, 1],
            y_pred=[0],
            probabilities=[
                0.1,
                0.9,
            ],
        )


def test_probabilities_must_be_valid() -> None:
    with pytest.raises(
        ValueError,
        match="between 0 and 1",
    ):
        calculate_binary_metrics(
            y_true=[0, 1],
            y_pred=[0, 1],
            probabilities=[
                -0.1,
                1.1,
            ],
        )


def test_true_labels_must_contain_both_classes() -> None:
    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        calculate_binary_metrics(
            y_true=[0, 0, 0],
            y_pred=[0, 0, 0],
            probabilities=[
                0.1,
                0.2,
                0.3,
            ],
        )


def test_attack_type_recall_rejects_invalid_labels() -> None:
    with pytest.raises(
        ValueError,
        match="non-attack labels",
    ):
        calculate_attack_type_recall(
            attack_types=[
                "constant",
                "constant",
            ],
            y_true=[
                1,
                0,
            ],
            y_pred=[
                1,
                0,
            ],
        )
    
def test_save_metrics_report(
    tmp_path,
) -> None:
    output_path = (
        tmp_path / "metrics.json"
    )

    report = {
        "test": {
            "accuracy": 0.99,
            "recall": 0.98,
        }
    }

    save_metrics_report(
        report=report,
        output_path=output_path,
    )

    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        loaded = json.load(file)

    assert loaded == report