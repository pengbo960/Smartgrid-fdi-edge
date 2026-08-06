from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.training.model_comparison import (
    run_model_comparison,
    save_model_comparison,
    save_model_comparison_figure,
)
from src.training.prepare_dataset import (
    load_feature_dataset,
    prepare_known_attack_dataset,
)
from src.training.split_data import split_stratified_grouped_dataset


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("Model comparison config must be a YAML mapping")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Logistic Regression and Random Forest fairly."
    )
    parser.add_argument("--config", default="config/model_comparison.yaml")
    args = parser.parse_args()
    config = load_config(args.config)

    dataframe = load_feature_dataset(config["dataset"]["path"])
    labels = config["labels"]
    preprocessing = config["preprocessing"]
    split_config = config["split"]
    prepared = prepare_known_attack_dataset(
        dataframe=dataframe,
        known_attack_types=labels["known_attack_types"],
        excluded_attack_types=labels.get("excluded_attack_types", []),
        target_column=labels.get("target_column", "is_attack"),
        group_column=split_config.get("group_column", "source_file"),
        include_groups=config["features"]["include_groups"],
        drop_warmup_rows=preprocessing.get("drop_warmup_rows", False),
        maximum_missing_fraction=float(
            preprocessing.get("maximum_missing_fraction", 0.0)
        ),
    )
    split = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=int(split_config.get("random_seed", 42)),
    )
    evaluation = config["evaluation"]
    output = config["output"]
    model_paths = {
        "logistic_regression": output["logistic_model"],
        "random_forest": output["random_forest_model"],
    }
    results = []
    for model_name in ("logistic_regression", "random_forest"):
        print(f"\nTraining {model_name}...")
        result = run_model_comparison(
            prepared=prepared,
            split=split,
            model_name=model_name,
            model_config=config["models"][model_name],
            default_threshold=float(evaluation.get("default_threshold", 0.5)),
            threshold_metric=str(evaluation.get("threshold_metric", "macro_f1")),
            latency_sample_size=int(evaluation.get("latency_sample_size", 1000)),
            model_path=model_paths[model_name],
        )
        results.append(result)
        print(
            f"macro_f1={result.metrics.macro_f1:.6f}, "
            f"recall={result.metrics.recall:.6f}, "
            f"fpr={result.metrics.false_positive_rate:.6f}, "
            f"latency={result.inference_mean_ms:.4f} ms, "
            f"size={result.model_size_mb:.4f} MB"
        )

    summary = save_model_comparison(results, output["summary_csv"])
    save_model_comparison_figure(
        summary=summary,
        output_path=output["comparison_figure"],
    )
    columns = [
        "model_name", "selected_threshold", "macro_f1", "precision",
        "recall", "false_positive_rate", "constant_recall",
        "random_recall", "replay_recall", "topic_spoof_recall",
        "training_seconds", "inference_mean_ms", "inference_p95_ms",
        "model_size_mb",
    ]
    print("\nMODEL COMPARISON")
    print(summary[columns].to_string(index=False))
    print(f"\nSaved to: {output['summary_csv']}")
    print(f"Figure saved to: {output['comparison_figure']}")


if __name__ == "__main__":
    main()
