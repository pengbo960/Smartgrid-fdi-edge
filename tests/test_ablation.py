from pathlib import Path

import pandas as pd
import pytest

from src.experiments.ablation import (
    AblationExperimentConfig,
    AblationExperimentResult,
    build_experiment_configs,
    build_predictions_dataframe,
    save_ablation_summary,
)


def test_build_experiment_configs() -> None:
    experiments = build_experiment_configs(
        [
            {
                "name": "value_only",
                "feature_groups": [
                    "value",
                ],
            },
            {
                "name": "all_views",
                "feature_groups": [
                    "value",
                    "temporal",
                    "protocol",
                ],
            },
        ]
    )

    assert experiments == (
        AblationExperimentConfig(
            name="value_only",
            feature_groups=(
                "value",
            ),
        ),
        AblationExperimentConfig(
            name="all_views",
            feature_groups=(
                "value",
                "temporal",
                "protocol",
            ),
        ),
    )


def test_duplicate_experiment_names_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Duplicate experiment name",
    ):
        build_experiment_configs(
            [
                {
                    "name": "value_only",
                    "feature_groups": [
                        "value",
                    ],
                },
                {
                    "name": "value_only",
                    "feature_groups": [
                        "temporal",
                    ],
                },
            ]
        )


def test_empty_feature_groups_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="at least one feature group",
    ):
        build_experiment_configs(
            [
                {
                    "name": "empty",
                    "feature_groups": [],
                }
            ]
        )


def test_predictions_dataframe() -> None:
    source = pd.DataFrame(
        {
            "source_file": [
                "normal_run_01.csv",
                "constant_run_01.csv",
            ],
            "attack_type": [
                "none",
                "constant",
            ],
            "is_attack": [
                0,
                1,
            ],
        }
    )

    predictions = build_predictions_dataframe(
        source_frame=source,
        y_true=[
            0,
            1,
        ],
        y_pred=[
            0,
            0,
        ],
        probabilities=[
            0.1,
            0.4,
        ],
    )

    assert predictions[
        "true_label"
    ].tolist() == [
        0,
        1,
    ]

    assert predictions[
        "predicted_label"
    ].tolist() == [
        0,
        0,
    ]

    assert predictions[
        "is_correct"
    ].tolist() == [
        1,
        0,
    ]


def build_result(
    name: str,
    macro_f1: float,
    recall: float,
) -> AblationExperimentResult:
    return AblationExperimentResult(
        experiment_name=name,
        feature_groups=(
            "value",
        ),
        feature_count=10,
        selected_threshold=0.5,
        accuracy=0.9,
        precision=0.9,
        recall=recall,
        f1=0.9,
        macro_f1=macro_f1,
        false_positive_rate=0.01,
        specificity=0.99,
        roc_auc=0.95,
        pr_auc=0.94,
        true_negative=99,
        false_positive=1,
        false_negative=2,
        true_positive=18,
        constant_recall=1.0,
        random_recall=0.8,
    )


def test_save_ablation_summary(
    tmp_path: Path,
) -> None:
    output_path = (
        tmp_path
        / "summary.csv"
    )

    summary = save_ablation_summary(
        results=[
            build_result(
                "lower",
                macro_f1=0.8,
                recall=0.85,
            ),
            build_result(
                "higher",
                macro_f1=0.9,
                recall=0.9,
            ),
        ],
        output_path=output_path,
    )

    assert output_path.exists()

    assert summary.iloc[0][
        "experiment_name"
    ] == "higher"

    assert summary.iloc[1][
        "experiment_name"
    ] == "lower"


def test_empty_ablation_summary_is_rejected(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        save_ablation_summary(
            results=[],
            output_path=(
                tmp_path / "summary.csv"
            ),
        )