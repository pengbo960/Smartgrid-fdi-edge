from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.evaluation.repeated_experiments import (
    aggregate_binary_confusion_matrix,
    aggregate_repeated_runs,
    configure_ablation_run,
    configure_model_comparison_run,
    configure_open_set_run,
    extract_open_set_row,
    save_table,
    validate_folds,
    validate_seeds,
)
from src.evaluation.visualization import save_confusion_matrix_from_counts


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    with resolved.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {resolved}")
    return config


def run_script(script: str, config: dict[str, Any], workspace: Path) -> None:
    config_path = workspace / "run_config.yaml"
    with config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / script), "--config", str(config_path)],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )


RunSpec = tuple[int | None, int]


def run_ablation(
    run_specs: tuple[RunSpec, ...],
    base: dict[str, Any],
    validation_offset: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold, seed in run_specs:
        label = f"fold {fold}" if fold is not None else f"seed {seed}"
        print(f"\n=== Repeated ablation: {label} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"ablation_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_ablation_run(
                base,
                seed,
                workspace,
                fold_index=fold - 1 if fold is not None else None,
                validation_offset=validation_offset,
            )
            run_script("scripts/run_ablation.py", config, workspace)
            frame = pd.read_csv(workspace / "summary.csv")
            frame.insert(0, "seed", seed)
            if fold is not None:
                frame.insert(0, "fold", fold)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_model_comparison(
    run_specs: tuple[RunSpec, ...],
    base: dict[str, Any],
    validation_offset: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for fold, seed in run_specs:
        label = f"fold {fold}" if fold is not None else f"seed {seed}"
        print(f"\n=== Repeated model comparison: {label} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"models_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_model_comparison_run(
                base,
                seed,
                workspace,
                fold_index=fold - 1 if fold is not None else None,
                validation_offset=validation_offset,
            )
            run_script("scripts/compare_models.py", config, workspace)
            frame = pd.read_csv(workspace / "summary.csv")
            frame.insert(0, "seed", seed)
            if fold is not None:
                frame.insert(0, "fold", fold)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_open_set(
    run_specs: tuple[RunSpec, ...],
    base: dict[str, Any],
    validation_offset: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold, seed in run_specs:
        label = f"fold {fold}" if fold is not None else f"seed {seed}"
        print(f"\n=== Repeated open set: {label} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"open_set_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_open_set_run(
                base,
                seed,
                workspace,
                fold_index=fold - 1 if fold is not None else None,
                validation_offset=validation_offset,
            )
            run_script("scripts/train_open_set.py", config, workspace)
            with (workspace / "metrics.json").open("r", encoding="utf-8") as file:
                report = json.load(file)
            rows.append(extract_open_set_row(report, seed, fold=fold))
    return pd.DataFrame(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the core dissertation experiments across grouped folds."
    )
    parser.add_argument("--config", default="config/repeated_experiments.yaml")
    parser.add_argument(
        "--sections", nargs="+",
        choices=("ablation", "model_comparison", "open_set"),
        default=("ablation", "model_comparison", "open_set"),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional seed override for legacy repeated-holdout configurations.",
    )
    parser.add_argument(
        "--folds", nargs="+", type=int, default=None,
        help="Optional one-based fold override for smoke tests or partial reruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_yaml(args.config)
    split_strategy = str(
        config.get("split_strategy", "repeated_holdout")
    ).strip().lower()
    validation_offset = int(config.get("validation_offset", 1))
    if split_strategy in {"grouped_kfold", "grouped_fold"}:
        fold_count = int(config.get("fold_count", 5))
        folds = validate_folds(
            args.folds if args.folds is not None else config["folds"],
            fold_count,
        )
        model_seed_base = int(config.get("model_seed_base", 42))
        run_specs: tuple[RunSpec, ...] = tuple(
            (fold, model_seed_base + fold - 1)
            for fold in folds
        )
    else:
        seeds = validate_seeds(
            args.seeds if args.seeds is not None else config["seeds"]
        )
        run_specs = tuple((None, seed) for seed in seeds)
    experiments = config["experiments"]
    output_config = config["output"]
    output_root = PROJECT_ROOT / output_config["root_directory"]

    if "ablation" in args.sections:
        runs = run_ablation(
            run_specs,
            load_yaml(experiments["ablation_config"]),
            validation_offset,
        )
        summary = aggregate_repeated_runs(
            runs, ("experiment_name", "feature_groups", "feature_count")
        )
        save_table(runs, output_root / "ablation_runs.csv")
        save_table(summary, output_root / "ablation_summary.csv")
        confusion_matrix = aggregate_binary_confusion_matrix(
            runs,
            experiment_name="all_views",
        )
        for raw_output_path in output_config.get(
            "aggregate_confusion_matrix_outputs",
            [],
        ):
            output_path = Path(raw_output_path)
            if not output_path.is_absolute():
                output_path = PROJECT_ROOT / output_path
            save_confusion_matrix_from_counts(
                confusion_matrix=confusion_matrix,
                output_path=output_path,
                title="All-View LR — Five-Fold Aggregate",
            )

    if "model_comparison" in args.sections:
        runs = run_model_comparison(
            run_specs,
            load_yaml(experiments["model_comparison_config"]),
            validation_offset,
        )
        summary = aggregate_repeated_runs(runs, ("model_name", "feature_count"))
        save_table(runs, output_root / "model_comparison_runs.csv")
        save_table(summary, output_root / "model_comparison_summary.csv")

    if "open_set" in args.sections:
        runs = run_open_set(
            run_specs,
            load_yaml(experiments["open_set_config"]),
            validation_offset,
        )
        runs_for_summary = runs.copy()
        runs_for_summary["experiment"] = "gradual_unseen"
        summary = aggregate_repeated_runs(runs_for_summary, ("experiment",))
        save_table(runs, output_root / "open_set_runs.csv")
        save_table(summary, output_root / "open_set_summary.csv")

    print(f"\nRepeated experiment results saved under {output_root}")


if __name__ == "__main__":
    main()
