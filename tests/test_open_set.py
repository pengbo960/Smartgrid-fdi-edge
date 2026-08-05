import numpy as np
import pandas as pd
import pytest

from src.detection.open_set import (
    apply_open_set_decision,
    quantile_threshold,
)
from scripts.train_open_set import (
    first_unknown_step_by_source,
    recall_by_attack_step,
)


def test_quantile_threshold() -> None:
    threshold = quantile_threshold(
        [
            0.1,
            0.2,
            0.3,
            0.4,
        ],
        0.5,
    )

    assert threshold == pytest.approx(
        0.25
    )


def test_low_confidence_is_rejected() -> None:
    decisions = apply_open_set_decision(
        predicted_labels=[
            "constant",
            "random",
        ],
        confidence_scores=[
            0.4,
            0.9,
        ],
        anomaly_scores=[
            0.1,
            0.1,
        ],
        confidence_threshold=0.5,
        anomaly_threshold=0.8,
    )

    assert decisions.tolist() == [
        "unknown",
        "random",
    ]


def test_anomalous_normal_is_rejected() -> None:
    decisions = apply_open_set_decision(
        predicted_labels=[
            "none",
            "constant",
        ],
        confidence_scores=[
            0.9,
            0.9,
        ],
        anomaly_scores=[
            0.9,
            0.9,
        ],
        confidence_threshold=0.5,
        anomaly_threshold=0.8,
    )

    assert decisions.tolist() == [
        "unknown",
        "constant",
    ]


def test_unknown_label_is_not_truncated() -> None:
    decisions = apply_open_set_decision(
        predicted_labels=[
            "none",
            "none",
        ],
        confidence_scores=[
            0.1,
            0.9,
        ],
        anomaly_scores=[
            0.1,
            0.9,
        ],
        confidence_threshold=0.5,
        anomaly_threshold=0.8,
    )

    assert decisions.tolist() == [
        "unknown",
        "unknown",
    ]


def test_score_lengths_must_match() -> None:
    with pytest.raises(
        ValueError,
        match="equal length",
    ):
        apply_open_set_decision(
            predicted_labels=[
                "none",
            ],
            confidence_scores=[
                0.9,
                0.8,
            ],
            anomaly_scores=[
                0.1,
            ],
            confidence_threshold=0.5,
            anomaly_threshold=0.8,
        )


def test_invalid_quantile_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="quantile",
    ):
        quantile_threshold(
            np.array(
                [
                    0.1,
                    0.2,
                ]
            ),
            1.1,
        )


def test_recall_by_attack_step() -> None:
    recall = recall_by_attack_step(
        attack_steps=pd.Series(
            [
                0,
                49,
                50,
                99,
            ]
        ),
        unknown_mask=np.array(
            [
                False,
                True,
                True,
                True,
            ]
        ),
    )

    assert recall == {
        "0-49": 0.5,
        "50-99": 1.0,
    }


def test_first_unknown_step_by_source() -> None:
    steps = first_unknown_step_by_source(
        source_files=pd.Series(
            [
                "run_1",
                "run_1",
                "run_2",
            ]
        ),
        attack_steps=pd.Series(
            [
                0,
                10,
                0,
            ]
        ),
        unknown_mask=np.array(
            [
                False,
                True,
                False,
            ]
        ),
    )

    assert steps == {
        "run_1": 10,
        "run_2": None,
    }
