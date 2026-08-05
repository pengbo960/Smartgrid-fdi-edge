from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
)

from src.detection.open_set import (
    apply_open_set_decision,
)
from src.training.prepare_dataset import (
    load_feature_dataset,
    prepare_known_attack_dataset,
)
from src.training.split_data import (
    split_stratified_grouped_dataset,
)
from src.training.train_open_set import (
    save_open_set_artifacts,
    score_open_set_rows,
    train_open_set_detector,
)


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Open-set config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Open-set config must contain a YAML mapping"
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate the open-set "
            "known/unknown attack detector."
        )
    )
    parser.add_argument(
        "--config",
        default="config/open_set.yaml",
    )
    return parser.parse_args()


def per_class_recall(
    y_true: np.ndarray,
    decisions: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}

    for label in sorted(
        set(y_true)
    ):
        mask = y_true == label
        result[str(label)] = float(
            (
                decisions[mask]
                == label
            ).mean()
        )

    return result


def build_prediction_frame(
    source: pd.DataFrame,
    true_labels: np.ndarray,
    known_predictions: np.ndarray,
    decisions: np.ndarray,
    confidence: np.ndarray,
    anomaly_scores: np.ndarray,
    evaluation_set: str,
) -> pd.DataFrame:
    metadata_columns = [
        column
        for column in (
            "source_file",
            "scenario_id",
            "device_id",
            "sequence_number",
            "attack_type",
            "attack_step",
        )
        if column in source.columns
    ]

    frame = (
        source[metadata_columns]
        .reset_index(drop=True)
        .copy()
    )
    frame["evaluation_set"] = evaluation_set
    frame["true_label"] = true_labels
    frame["known_prediction"] = (
        known_predictions
    )
    frame["final_decision"] = decisions
    frame["confidence"] = confidence
    frame["anomaly_score"] = (
        anomaly_scores
    )
    frame["is_unknown"] = (
        decisions == "unknown"
    ).astype(int)

    return frame


def recall_by_attack_step(
    attack_steps: pd.Series,
    unknown_mask: np.ndarray,
    bin_size: int = 50,
) -> dict[str, float]:
    """
    Summarise unknown recall as an unseen attack progresses.
    """
    if bin_size <= 0:
        raise ValueError(
            "bin_size must be greater than zero"
        )

    steps = pd.to_numeric(
        attack_steps,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )
    unknown = np.asarray(
        unknown_mask,
        dtype=bool,
    )

    if len(steps) != len(unknown):
        raise ValueError(
            "attack_steps and unknown_mask "
            "must have equal length"
        )

    if not np.isfinite(steps).all():
        raise ValueError(
            "Unseen attack steps must be finite"
        )

    result: dict[str, float] = {}

    maximum_step = int(
        steps.max()
    )

    for start in range(
        0,
        maximum_step + 1,
        bin_size,
    ):
        end = start + bin_size - 1
        mask = (
            (steps >= start)
            & (steps <= end)
        )

        if mask.any():
            result[
                f"{start}-{end}"
            ] = float(
                unknown[mask].mean()
            )

    return result


def first_unknown_step_by_source(
    source_files: pd.Series,
    attack_steps: pd.Series,
    unknown_mask: np.ndarray,
) -> dict[str, int | None]:
    frame = pd.DataFrame(
        {
            "source_file": (
                source_files.astype(str)
            ),
            "attack_step": pd.to_numeric(
                attack_steps,
                errors="coerce",
            ),
            "is_unknown": np.asarray(
                unknown_mask,
                dtype=bool,
            ),
        }
    )

    if frame[
        "attack_step"
    ].isna().any():
        raise ValueError(
            "Unseen attack steps must be numeric"
        )

    result: dict[str, int | None] = {}

    for source_file, group in frame.groupby(
        "source_file",
        sort=True,
    ):
        detected = group[
            group["is_unknown"]
        ]
        result[str(source_file)] = (
            int(
                detected[
                    "attack_step"
                ].min()
            )
            if not detected.empty
            else None
        )

    return result


def main() -> None:
    args = parse_arguments()
    config = load_config(
        args.config
    )

    dataset_config = config["dataset"]
    labels_config = config["labels"]
    features_config = config["features"]
    split_config = config["split"]
    model_config = config["model"]
    rejection_config = config["rejection"]
    output_config = config["output"]

    known_attack_types = [
        str(label)
        for label in labels_config[
            "known_attack_types"
        ]
    ]
    unseen_attack_type = str(
        labels_config[
            "unseen_attack_type"
        ]
    )

    dataframe = load_feature_dataset(
        dataset_config["path"]
    )

    prepared = prepare_known_attack_dataset(
        dataframe=dataframe,
        known_attack_types=(
            known_attack_types
        ),
        excluded_attack_types=[
            unseen_attack_type
        ],
        target_column="is_attack",
        group_column=str(
            split_config.get(
                "group_column",
                "source_file",
            )
        ),
        include_groups=tuple(
            str(group)
            for group in features_config[
                "include_groups"
            ]
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

    result = train_open_set_detector(
        prepared=prepared,
        split=split,
        anomaly_feature_columns=tuple(
            str(feature)
            for feature in features_config[
                "anomaly_features"
            ]
        ),
        confidence_lower_quantile=float(
            rejection_config[
                "confidence_lower_quantile"
            ]
        ),
        anomaly_upper_quantile=float(
            rejection_config[
                "anomaly_upper_quantile"
            ]
        ),
        class_weight=model_config.get(
            "class_weight",
            "balanced",
        ),
        max_iter=int(
            model_config.get(
                "max_iter",
                2000,
            )
        ),
        isolation_estimators=int(
            model_config.get(
                "isolation_estimators",
                300,
            )
        ),
        random_seed=int(
            model_config.get(
                "random_seed",
                42,
            )
        ),
    )

    known_decisions = apply_open_set_decision(
        predicted_labels=(
            result.test_predictions
        ),
        confidence_scores=(
            result.test_confidence
        ),
        anomaly_scores=(
            result.test_anomaly_scores
        ),
        confidence_threshold=(
            result.confidence_threshold
        ),
        anomaly_threshold=(
            result.anomaly_threshold
        ),
    )

    unseen = dataframe[
        dataframe["attack_type"]
        .astype(str)
        .eq(unseen_attack_type)
    ].copy()

    if unseen.empty:
        raise ValueError(
            "No unseen attack rows were found"
        )

    (
        unseen_predictions,
        unseen_confidence,
        unseen_anomaly_scores,
    ) = score_open_set_rows(
        result=result,
        dataframe=unseen,
    )

    unseen_decisions = apply_open_set_decision(
        predicted_labels=(
            unseen_predictions
        ),
        confidence_scores=(
            unseen_confidence
        ),
        anomaly_scores=(
            unseen_anomaly_scores
        ),
        confidence_threshold=(
            result.confidence_threshold
        ),
        anomaly_threshold=(
            result.anomaly_threshold
        ),
    )

    known_unknown = (
        known_decisions == "unknown"
    )
    unseen_unknown = (
        unseen_decisions == "unknown"
    )

    unknown_true_positive = int(
        unseen_unknown.sum()
    )
    unknown_false_positive = int(
        known_unknown.sum()
    )

    unknown_precision = (
        unknown_true_positive
        / (
            unknown_true_positive
            + unknown_false_positive
        )
        if (
            unknown_true_positive
            + unknown_false_positive
        ) > 0
        else 0.0
    )

    confidence_only_unknown = (
        unseen_confidence
        < result.confidence_threshold
    )
    anomaly_unknown = (
        (unseen_predictions == "none")
        & (
            unseen_anomaly_scores
            > result.anomaly_threshold
        )
    )

    step_recall = recall_by_attack_step(
        attack_steps=unseen[
            "attack_step"
        ],
        unknown_mask=unseen_unknown,
    )
    first_detection_steps = (
        first_unknown_step_by_source(
            source_files=unseen[
                "source_file"
            ],
            attack_steps=unseen[
                "attack_step"
            ],
            unknown_mask=unseen_unknown,
        )
    )
    detected_steps = [
        step
        for step in (
            first_detection_steps.values()
        )
        if step is not None
    ]

    report = {
        "dataset": {
            "known_train_rows": len(
                split.train
            ),
            "known_validation_rows": len(
                split.validation
            ),
            "known_test_rows": len(
                split.test
            ),
            "unseen_test_rows": len(
                unseen
            ),
            "known_classes": list(
                result.classes
            ),
            "unseen_attack_type": (
                unseen_attack_type
            ),
        },
        "thresholds": {
            "confidence_threshold": (
                result.confidence_threshold
            ),
            "anomaly_threshold": (
                result.anomaly_threshold
            ),
            "confidence_lower_quantile": float(
                rejection_config[
                    "confidence_lower_quantile"
                ]
            ),
            "anomaly_upper_quantile": float(
                rejection_config[
                    "anomaly_upper_quantile"
                ]
            ),
        },
        "known_closed_set": {
            "accuracy": float(
                accuracy_score(
                    result.y_test,
                    result.test_predictions,
                )
            ),
            "macro_f1": float(
                f1_score(
                    result.y_test,
                    result.test_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "known_open_set": {
            "false_unknown_rate": float(
                known_unknown.mean()
            ),
            "acceptance_rate": float(
                (~known_unknown).mean()
            ),
            "overall_correct_rate": float(
                (
                    known_decisions
                    == result.y_test
                ).mean()
            ),
            "per_class_recall": (
                per_class_recall(
                    result.y_test,
                    known_decisions,
                )
            ),
        },
        "unseen": {
            "unknown_recall": float(
                unseen_unknown.mean()
            ),
            "unknown_precision": float(
                unknown_precision
            ),
            "confidence_only_recall": float(
                confidence_only_unknown.mean()
            ),
            "normal_anomaly_recall": float(
                anomaly_unknown.mean()
            ),
            "known_prediction_distribution": (
                pd.Series(
                    unseen_predictions
                )
                .value_counts()
                .to_dict()
            ),
            "unknown_recall_by_attack_step": (
                step_recall
            ),
            "first_unknown_step_by_source": (
                first_detection_steps
            ),
            "mean_first_unknown_step": (
                float(
                    np.mean(
                        detected_steps
                    )
                )
                if detected_steps
                else None
            ),
        },
    }

    save_open_set_artifacts(
        result=result,
        classifier_path=(
            output_config[
                "classifier_path"
            ]
        ),
        scaler_path=(
            output_config[
                "scaler_path"
            ]
        ),
        anomaly_detector_path=(
            output_config[
                "anomaly_detector_path"
            ]
        ),
        metadata_path=(
            output_config[
                "metadata_path"
            ]
        ),
    )

    metrics_path = Path(
        output_config[
            "metrics_path"
        ]
    )
    metrics_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with metrics_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=2,
        )

    known_frame = build_prediction_frame(
        source=split.test,
        true_labels=result.y_test,
        known_predictions=(
            result.test_predictions
        ),
        decisions=known_decisions,
        confidence=result.test_confidence,
        anomaly_scores=(
            result.test_anomaly_scores
        ),
        evaluation_set="known_test",
    )
    unseen_frame = build_prediction_frame(
        source=unseen,
        true_labels=np.full(
            len(unseen),
            unseen_attack_type,
            dtype=object,
        ),
        known_predictions=(
            unseen_predictions
        ),
        decisions=unseen_decisions,
        confidence=unseen_confidence,
        anomaly_scores=(
            unseen_anomaly_scores
        ),
        evaluation_set="unseen_test",
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
    pd.concat(
        [
            known_frame,
            unseen_frame,
        ],
        ignore_index=True,
    ).to_csv(
        predictions_path,
        index=False,
    )

    print(
        json.dumps(
            report,
            indent=2,
        )
    )
    print(
        "\nSaved open-set metrics to: "
        f"{metrics_path}"
    )


if __name__ == "__main__":
    main()
