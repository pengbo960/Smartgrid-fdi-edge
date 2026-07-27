from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.metrics import (
    calculate_attack_type_recall,
    calculate_binary_metrics,
    save_metrics_report,
)
from src.training.prepare_dataset import (
    load_feature_dataset,
    prepare_known_attack_dataset,
)
from src.training.split_data import (
    split_stratified_grouped_dataset,
)
from src.training.train_baseline import (
    save_baseline_artifacts,
    train_logistic_baseline,
)


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Baseline config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Baseline config must contain a YAML mapping"
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the Logistic Regression baseline."
        )
    )

    parser.add_argument(
        "--config",
        default="config/baseline.yaml",
        help="Path to the baseline YAML configuration.",
    )

    return parser.parse_args()


def build_predictions_dataframe(
    source_frame: pd.DataFrame,
    y_true,
    y_pred,
    probabilities,
) -> pd.DataFrame:
    """
    Build a test prediction table with metadata.
    """
    metadata_columns = [
        column
        for column in [
            "source_file",
            "scenario_id",
            "device_id",
            "sequence_number",
            "attack_type",
            "is_attack",
        ]
        if column in source_frame.columns
    ]

    predictions = source_frame[
        metadata_columns
    ].copy()

    predictions["true_label"] = y_true
    predictions["predicted_label"] = y_pred
    predictions["attack_probability"] = probabilities
    predictions["is_correct"] = (
        predictions["true_label"]
        == predictions["predicted_label"]
    ).astype(int)

    return predictions


def print_metrics(
    split_name: str,
    metrics: dict[str, Any],
) -> None:
    print(f"\n{split_name}")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.6f}"
            )
        else:
            print(
                f"{key}: {value}"
            )


def main() -> None:
    args = parse_arguments()

    config = load_config(
        args.config
    )

    dataset_config = config["dataset"]
    labels_config = config["labels"]
    feature_config = config["features"]
    split_config = config["split"]
    preprocessing_config = config["preprocessing"]
    model_config = config["model"]
    output_config = config["output"]

    dataframe = load_feature_dataset(
        dataset_config["path"]
    )

    prepared = prepare_known_attack_dataset(
        dataframe=dataframe,
        known_attack_types=(
            labels_config[
                "known_attack_types"
            ]
        ),
        excluded_attack_types=(
            labels_config.get(
                "excluded_attack_types",
                [],
            )
        ),
        target_column=labels_config.get(
            "target_column",
            "is_attack",
        ),
        group_column=split_config.get(
            "group_column",
            "source_file",
        ),
        include_groups=feature_config[
            "include_groups"
        ],
        drop_warmup_rows=(
            preprocessing_config.get(
                "drop_warmup_rows",
                False,
            )
        ),
        maximum_missing_fraction=float(
            preprocessing_config.get(
                "maximum_missing_fraction",
                0.0,
            )
        ),
    )

    split = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=int(
            split_config.get(
                "random_seed",
                42,
            )
        ),
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
        class_weight=model_config.get(
            "class_weight",
            "balanced",
        ),
        max_iter=int(
            model_config.get(
                "max_iter",
                1000,
            )
        ),
        random_seed=int(
            model_config.get(
                "random_seed",
                42,
            )
        ),
    )

    train_metrics = calculate_binary_metrics(
        y_true=result.y_train,
        y_pred=result.train_predictions,
        probabilities=result.train_probabilities,
    )

    validation_metrics = calculate_binary_metrics(
        y_true=result.y_validation,
        y_pred=result.validation_predictions,
        probabilities=result.validation_probabilities,
    )

    test_metrics = calculate_binary_metrics(
        y_true=result.y_test,
        y_pred=result.test_predictions,
        probabilities=result.test_probabilities,
    )

    attack_type_recall = (
        calculate_attack_type_recall(
            attack_types=(
                split.test[
                    "attack_type"
                ].to_numpy()
            ),
            y_true=result.y_test,
            y_pred=result.test_predictions,
        )
    )

    metrics_report = {
        "model": {
            "name": "logistic_regression",
            "feature_count": len(
                result.feature_columns
            ),
            "feature_groups": (
                feature_config[
                    "include_groups"
                ]
            ),
            "class_weight": (
                model_config.get(
                    "class_weight",
                    "balanced",
                )
            ),
            "max_iter": int(
                model_config.get(
                    "max_iter",
                    1000,
                )
            ),
            "iterations_used": (
                result.model.n_iter_
                .astype(int)
                .tolist()
            ),
        },
        "dataset": {
            "train_rows": len(
                split.train
            ),
            "validation_rows": len(
                split.validation
            ),
            "test_rows": len(
                split.test
            ),
            "train_groups": list(
                split.train_groups
            ),
            "validation_groups": list(
                split.validation_groups
            ),
            "test_groups": list(
                split.test_groups
            ),
        },
        "train": train_metrics.to_dict(),
        "validation": (
            validation_metrics.to_dict()
        ),
        "test": test_metrics.to_dict(),
        "test_attack_type_recall": (
            attack_type_recall
        ),
    }

    save_baseline_artifacts(
        result=result,
        model_path=(
            output_config["model_path"]
        ),
        scaler_path=(
            output_config["scaler_path"]
        ),
        feature_names_path=(
            output_config[
                "feature_names_path"
            ]
        ),
    )

    save_metrics_report(
        report=metrics_report,
        output_path=(
            output_config["metrics_path"]
        ),
    )

    predictions = (
        build_predictions_dataframe(
            source_frame=split.test,
            y_true=result.y_test,
            y_pred=result.test_predictions,
            probabilities=(
                result.test_probabilities
            ),
        )
    )

    predictions_path = Path(
        output_config[
            "predictions_path"
        ]
    )

    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    print_metrics(
        "TRAIN",
        train_metrics.to_dict(),
    )

    print_metrics(
        "VALIDATION",
        validation_metrics.to_dict(),
    )

    print_metrics(
        "TEST",
        test_metrics.to_dict(),
    )

    print(
        "\nTEST RECALL BY ATTACK TYPE"
    )

    for attack_type, recall in (
        attack_type_recall.items()
    ):
        print(
            f"{attack_type}: "
            f"{recall:.6f}"
        )

    print("\nSaved artifacts:")

    for label, path in [
        (
            "Model",
            output_config["model_path"],
        ),
        (
            "Scaler",
            output_config["scaler_path"],
        ),
        (
            "Feature names",
            output_config[
                "feature_names_path"
            ],
        ),
        (
            "Metrics",
            output_config[
                "metrics_path"
            ],
        ),
        (
            "Predictions",
            output_config[
                "predictions_path"
            ],
        ),
    ]:
        print(
            f"{label}: {path}"
        )


if __name__ == "__main__":
    main()