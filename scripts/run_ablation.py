from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.experiments.ablation import (
    build_experiment_configs,
    run_ablation_experiment,
    save_ablation_summary,
)
from src.training.prepare_dataset import (
    load_feature_dataset,
)


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Ablation config not found: {config_path}"
        )

    if not config_path.is_file():
        raise ValueError(
            f"Ablation config is not a file: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Ablation config must contain a YAML mapping"
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Logistic Regression "
            "multi-view ablation experiments."
        )
    )

    parser.add_argument(
        "--config",
        default="config/ablation.yaml",
        help="Path to ablation YAML configuration.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = load_config(
        args.config
    )

    dataset_config = config["dataset"]
    labels_config = config["labels"]
    split_config = config["split"]
    preprocessing_config = config[
        "preprocessing"
    ]
    model_config = config["model"]
    output_config = config["output"]

    experiments = build_experiment_configs(
        config["experiments"]
    )

    dataframe = load_feature_dataset(
        dataset_config["path"]
    )

    results = []

    for index, experiment in enumerate(
        experiments,
        start=1,
    ):
        print(
            f"\n[{index}/{len(experiments)}] "
            f"Running {experiment.name}"
        )

        result = run_ablation_experiment(
            dataframe=dataframe,
            experiment=experiment,
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
            split_random_seed=int(
                split_config.get(
                    "random_seed",
                    42,
                )
            ),
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
            model_random_seed=int(
                model_config.get(
                    "random_seed",
                    42,
                )
            ),
            default_threshold=float(
                model_config.get(
                    "default_threshold",
                    0.5,
                )
            ),
            threshold_metric=str(
                model_config.get(
                    "threshold_metric",
                    "macro_f1",
                )
            ),
            metrics_directory=(
                output_config[
                    "metrics_directory"
                ]
            ),
            predictions_directory=(
                output_config[
                    "predictions_directory"
                ]
            ),
            figures_directory=(
                output_config[
                    "figures_directory"
                ]
            ),
        )

        results.append(
            result
        )

        print(
            f"Feature groups: "
            f"{'+'.join(result.feature_groups)}"
        )

        print(
            f"Feature count: "
            f"{result.feature_count}"
        )

        print(
            f"Selected threshold: "
            f"{result.selected_threshold:.2f}"
        )

        print(
            f"Test macro F1: "
            f"{result.macro_f1:.6f}"
        )

        print(
            f"Test recall: "
            f"{result.recall:.6f}"
        )

        print(
            f"Test FPR: "
            f"{result.false_positive_rate:.6f}"
        )

    summary = save_ablation_summary(
        results=results,
        output_path=(
            output_config[
                "summary_csv"
            ]
        ),
    )

    display_columns = [
        "experiment_name",
        "feature_groups",
        "feature_count",
        "selected_threshold",
        "precision",
        "recall",
        "macro_f1",
        "false_positive_rate",
        "constant_recall",
        "random_recall",
    ]

    print("\nABLATION SUMMARY")

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSummary saved to: "
        f"{output_config['summary_csv']}"
    )


if __name__ == "__main__":
    main()