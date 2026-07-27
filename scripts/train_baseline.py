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
from src.evaluation.thresholds import (
    predictions_from_probabilities,
    search_best_threshold,
)
from src.evaluation.visualization import (
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
    save_threshold_curve,
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
    """
    Load the baseline YAML configuration.
    """
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Baseline config not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Baseline config path is not a file: {config_path}"
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
            "Train, evaluate and save the Logistic Regression baseline."
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
    y_true: Any,
    y_pred: Any,
    probabilities: Any,
) -> pd.DataFrame:
    """
    Build a prediction table containing source metadata and results.
    """
    if not (
        len(source_frame)
        == len(y_true)
        == len(y_pred)
        == len(probabilities)
    ):
        raise ValueError(
            "source_frame, y_true, y_pred and probabilities "
            "must have the same length"
        )

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

    predictions = (
        source_frame[
            metadata_columns
        ]
        .reset_index(drop=True)
        .copy()
    )

    predictions["true_label"] = y_true
    predictions["predicted_label"] = y_pred
    predictions["attack_probability"] = (
        probabilities
    )

    predictions["is_correct"] = (
        predictions["true_label"]
        == predictions["predicted_label"]
    ).astype(int)

    return predictions


def print_metrics(
    split_name: str,
    metrics: dict[str, Any],
) -> None:
    """
    Print one split's metrics in a readable form.
    """
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
    preprocessing_config = config[
        "preprocessing"
    ]
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
        target_column=(
            labels_config.get(
                "target_column",
                "is_attack",
            )
        ),
        group_column=(
            split_config.get(
                "group_column",
                "source_file",
            )
        ),
        include_groups=(
            feature_config[
                "include_groups"
            ]
        ),
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

    split = (
        split_stratified_grouped_dataset(
            prepared=prepared,
            random_seed=int(
                split_config.get(
                    "random_seed",
                    42,
                )
            ),
        )
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
        class_weight=(
            model_config.get(
                "class_weight",
                "balanced",
            )
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

    default_threshold = float(
        model_config.get(
            "default_threshold",
            0.5,
        )
    )

    threshold_metric = str(
        model_config.get(
            "threshold_metric",
            "macro_f1",
        )
    )

    best_threshold, threshold_evaluations = (
        search_best_threshold(
            y_true=result.y_validation,
            probabilities=(
                result.validation_probabilities
            ),
            metric=threshold_metric,
            reference_threshold=(
                default_threshold
            ),
        )
    )

    selected_threshold = (
        best_threshold.threshold
    )

    train_predictions = (
        predictions_from_probabilities(
            probabilities=(
                result.train_probabilities
            ),
            threshold=selected_threshold,
        )
    )

    validation_predictions = (
        predictions_from_probabilities(
            probabilities=(
                result.validation_probabilities
            ),
            threshold=selected_threshold,
        )
    )

    test_predictions = (
        predictions_from_probabilities(
            probabilities=(
                result.test_probabilities
            ),
            threshold=selected_threshold,
        )
    )

    train_metrics = calculate_binary_metrics(
        y_true=result.y_train,
        y_pred=train_predictions,
        probabilities=(
            result.train_probabilities
        ),
    )

    validation_metrics = (
        calculate_binary_metrics(
            y_true=result.y_validation,
            y_pred=validation_predictions,
            probabilities=(
                result.validation_probabilities
            ),
        )
    )

    test_metrics = calculate_binary_metrics(
        y_true=result.y_test,
        y_pred=test_predictions,
        probabilities=(
            result.test_probabilities
        ),
    )

    attack_type_recall = (
        calculate_attack_type_recall(
            attack_types=(
                split.test[
                    "attack_type"
                ].to_numpy()
            ),
            y_true=result.y_test,
            y_pred=test_predictions,
        )
    )

    metrics_report = {
        "model": {
            "name": "logistic_regression",
            "feature_count": len(
                result.feature_columns
            ),
            "feature_groups": list(
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
            "default_threshold": (
                default_threshold
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "threshold_metric": (
                threshold_metric
            ),
            "validation_threshold_metrics": {
                "accuracy": (
                    best_threshold.accuracy
                ),
                "precision": (
                    best_threshold.precision
                ),
                "recall": (
                    best_threshold.recall
                ),
                "f1": best_threshold.f1,
                "macro_f1": (
                    best_threshold.macro_f1
                ),
                "false_positive_rate": (
                    best_threshold
                    .false_positive_rate
                ),
            },
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
            output_config[
                "model_path"
            ]
        ),
        scaler_path=(
            output_config[
                "scaler_path"
            ]
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
            output_config[
                "metrics_path"
            ]
        ),
    )

    predictions = (
        build_predictions_dataframe(
            source_frame=split.test,
            y_true=result.y_test,
            y_pred=test_predictions,
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

    save_confusion_matrix(
        y_true=result.y_test,
        y_pred=test_predictions,
        output_path=(
            output_config[
                "confusion_matrix_path"
            ]
        ),
        title=(
            "Logistic Regression "
            "Test Confusion Matrix"
        ),
    )

    save_roc_curve(
        y_true=result.y_test,
        probabilities=(
            result.test_probabilities
        ),
        output_path=(
            output_config[
                "roc_curve_path"
            ]
        ),
        title=(
            "Logistic Regression "
            "Test ROC Curve"
        ),
    )

    save_precision_recall_curve(
        y_true=result.y_test,
        probabilities=(
            result.test_probabilities
        ),
        output_path=(
            output_config[
                "pr_curve_path"
            ]
        ),
        title=(
            "Logistic Regression "
            "Test Precision-Recall Curve"
        ),
    )

    save_threshold_curve(
        evaluations=(
            threshold_evaluations
        ),
        selected_threshold=(
            selected_threshold
        ),
        output_path=(
            output_config[
                "threshold_curve_path"
            ]
        ),
    )

    print(
        "\nThreshold selection:"
    )

    print(
        f"Default threshold: "
        f"{default_threshold:.2f}"
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation {threshold_metric}: "
        f"{getattr(best_threshold, threshold_metric):.6f}"
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

    saved_artifacts = [
        (
            "Model",
            output_config[
                "model_path"
            ],
        ),
        (
            "Scaler",
            output_config[
                "scaler_path"
            ],
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
        (
            "Confusion matrix",
            output_config[
                "confusion_matrix_path"
            ],
        ),
        (
            "ROC curve",
            output_config[
                "roc_curve_path"
            ],
        ),
        (
            "PR curve",
            output_config[
                "pr_curve_path"
            ],
        ),
        (
            "Threshold curve",
            output_config[
                "threshold_curve_path"
            ],
        ),
    ]

    for label, path in saved_artifacts:
        print(
            f"{label}: {path}"
        )


if __name__ == "__main__":
    main()