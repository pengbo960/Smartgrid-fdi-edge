from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

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
    prepare_known_attack_dataset,
)
from src.training.split_data import (
    split_stratified_grouped_dataset,
)
from src.training.train_baseline import (
    train_logistic_baseline,
)


@dataclass(frozen=True)
class AblationExperimentConfig:
    name: str
    feature_groups: tuple[str, ...]


@dataclass(frozen=True)
class AblationExperimentResult:
    experiment_name: str
    feature_groups: tuple[str, ...]
    feature_count: int
    selected_threshold: float

    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    false_positive_rate: float
    specificity: float
    roc_auc: float
    pr_auc: float

    true_negative: int
    false_positive: int
    false_negative: int
    true_positive: int

    constant_recall: float | None
    random_recall: float | None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["feature_groups"] = "+".join(
            self.feature_groups
        )
        return result


def build_experiment_configs(
    raw_experiments: Iterable[dict[str, Any]],
) -> tuple[AblationExperimentConfig, ...]:
    """
    Convert YAML experiment entries into validated configurations.
    """
    experiments: list[
        AblationExperimentConfig
    ] = []

    experiment_names: set[str] = set()

    for raw_experiment in raw_experiments:
        if not isinstance(
            raw_experiment,
            dict,
        ):
            raise TypeError(
                "Each experiment must be a dictionary"
            )

        name = str(
            raw_experiment.get(
                "name",
                "",
            )
        ).strip()

        if not name:
            raise ValueError(
                "Experiment name must not be empty"
            )

        if name in experiment_names:
            raise ValueError(
                f"Duplicate experiment name: {name}"
            )

        raw_groups = raw_experiment.get(
            "feature_groups"
        )

        if not isinstance(
            raw_groups,
            list,
        ):
            raise TypeError(
                f"feature_groups for {name} "
                "must be a list"
            )

        feature_groups = tuple(
            str(group).strip()
            for group in raw_groups
            if str(group).strip()
        )

        if not feature_groups:
            raise ValueError(
                f"Experiment {name} must contain "
                "at least one feature group"
            )

        experiments.append(
            AblationExperimentConfig(
                name=name,
                feature_groups=feature_groups,
            )
        )

        experiment_names.add(name)

    if not experiments:
        raise ValueError(
            "At least one ablation experiment is required"
        )

    return tuple(experiments)


def build_predictions_dataframe(
    source_frame: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    probabilities: Any,
) -> pd.DataFrame:
    """
    Build one prediction table for an ablation experiment.
    """
    if not (
        len(source_frame)
        == len(y_true)
        == len(y_pred)
        == len(probabilities)
    ):
        raise ValueError(
            "Prediction inputs must have the same length"
        )

    metadata_columns = [
        column
        for column in (
            "source_file",
            "scenario_id",
            "device_id",
            "sequence_number",
            "attack_type",
            "is_attack",
        )
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


def run_ablation_experiment(
    dataframe: pd.DataFrame,
    experiment: AblationExperimentConfig,
    known_attack_types: Iterable[str],
    excluded_attack_types: Iterable[str],
    target_column: str,
    group_column: str,
    drop_warmup_rows: bool,
    maximum_missing_fraction: float,
    split_random_seed: int,
    class_weight: str | dict[int, float] | None,
    max_iter: int,
    model_random_seed: int,
    default_threshold: float,
    threshold_metric: str,
    metrics_directory: str | Path,
    predictions_directory: str | Path,
    figures_directory: str | Path,
) -> AblationExperimentResult:
    """
    Run one feature-group ablation experiment.
    """
    prepared = prepare_known_attack_dataset(
        dataframe=dataframe,
        known_attack_types=known_attack_types,
        excluded_attack_types=excluded_attack_types,
        target_column=target_column,
        group_column=group_column,
        include_groups=experiment.feature_groups,
        drop_warmup_rows=drop_warmup_rows,
        maximum_missing_fraction=(
            maximum_missing_fraction
        ),
    )

    split = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=split_random_seed,
    )

    training_result = train_logistic_baseline(
        prepared=prepared,
        split=split,
        class_weight=class_weight,
        max_iter=max_iter,
        random_seed=model_random_seed,
    )

    best_threshold, threshold_evaluations = (
        search_best_threshold(
            y_true=(
                training_result.y_validation
            ),
            probabilities=(
                training_result
                .validation_probabilities
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

    test_predictions = (
        predictions_from_probabilities(
            probabilities=(
                training_result
                .test_probabilities
            ),
            threshold=selected_threshold,
        )
    )

    test_metrics = calculate_binary_metrics(
        y_true=training_result.y_test,
        y_pred=test_predictions,
        probabilities=(
            training_result.test_probabilities
        ),
    )

    attack_type_recall = (
        calculate_attack_type_recall(
            attack_types=(
                split.test[
                    "attack_type"
                ].to_numpy()
            ),
            y_true=training_result.y_test,
            y_pred=test_predictions,
        )
    )

    metrics_path = (
        Path(metrics_directory)
        / f"{experiment.name}.json"
    )

    predictions_path = (
        Path(predictions_directory)
        / f"{experiment.name}.csv"
    )

    figures_path = Path(
        figures_directory
    )

    report = {
        "experiment": {
            "name": experiment.name,
            "feature_groups": list(
                experiment.feature_groups
            ),
            "feature_count": len(
                training_result.feature_columns
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
        "test": test_metrics.to_dict(),
        "test_attack_type_recall": (
            attack_type_recall
        ),
    }

    save_metrics_report(
        report=report,
        output_path=metrics_path,
    )

    predictions = (
        build_predictions_dataframe(
            source_frame=split.test,
            y_true=training_result.y_test,
            y_pred=test_predictions,
            probabilities=(
                training_result
                .test_probabilities
            ),
        )
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
        y_true=training_result.y_test,
        y_pred=test_predictions,
        output_path=(
            figures_path
            / (
                f"{experiment.name}"
                "_confusion_matrix.png"
            )
        ),
        title=(
            f"{experiment.name} "
            "Test Confusion Matrix"
        ),
    )

    save_roc_curve(
        y_true=training_result.y_test,
        probabilities=(
            training_result
            .test_probabilities
        ),
        output_path=(
            figures_path
            / f"{experiment.name}_roc_curve.png"
        ),
        title=(
            f"{experiment.name} "
            "Test ROC Curve"
        ),
    )

    save_precision_recall_curve(
        y_true=training_result.y_test,
        probabilities=(
            training_result
            .test_probabilities
        ),
        output_path=(
            figures_path
            / f"{experiment.name}_pr_curve.png"
        ),
        title=(
            f"{experiment.name} "
            "Test Precision-Recall Curve"
        ),
    )

    save_threshold_curve(
        evaluations=threshold_evaluations,
        selected_threshold=selected_threshold,
        output_path=(
            figures_path
            / (
                f"{experiment.name}"
                "_threshold_curve.png"
            )
        ),
    )

    return AblationExperimentResult(
        experiment_name=experiment.name,
        feature_groups=(
            experiment.feature_groups
        ),
        feature_count=len(
            training_result.feature_columns
        ),
        selected_threshold=(
            selected_threshold
        ),
        accuracy=test_metrics.accuracy,
        precision=test_metrics.precision,
        recall=test_metrics.recall,
        f1=test_metrics.f1,
        macro_f1=test_metrics.macro_f1,
        false_positive_rate=(
            test_metrics.false_positive_rate
        ),
        specificity=test_metrics.specificity,
        roc_auc=test_metrics.roc_auc,
        pr_auc=test_metrics.pr_auc,
        true_negative=(
            test_metrics.true_negative
        ),
        false_positive=(
            test_metrics.false_positive
        ),
        false_negative=(
            test_metrics.false_negative
        ),
        true_positive=(
            test_metrics.true_positive
        ),
        constant_recall=(
            attack_type_recall.get(
                "constant"
            )
        ),
        random_recall=(
            attack_type_recall.get(
                "random"
            )
        ),
    )


def save_ablation_summary(
    results: Iterable[
        AblationExperimentResult
    ],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Save all experiment results as one comparison CSV.
    """
    result_rows = [
        result.to_dict()
        for result in results
    ]

    if not result_rows:
        raise ValueError(
            "Ablation results must not be empty"
        )

    summary = pd.DataFrame(
        result_rows
    )

    summary = summary.sort_values(
        by=[
            "macro_f1",
            "recall",
            "false_positive_rate",
        ],
        ascending=[
            False,
            False,
            True,
        ],
        kind="stable",
    ).reset_index(drop=True)

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        path,
        index=False,
    )

    return summary