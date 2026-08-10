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
    aggregate_repeated_runs,
    configure_ablation_run,
    configure_model_comparison_run,
    configure_open_set_run,
    extract_open_set_row,
    save_table,
    validate_seeds,
)


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


def run_ablation(seeds: tuple[int, ...], base: dict[str, Any]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"\n=== Repeated ablation: seed {seed} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"ablation_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_ablation_run(base, seed, workspace)
            run_script("scripts/run_ablation.py", config, workspace)
            frame = pd.read_csv(workspace / "summary.csv")
            frame.insert(0, "seed", seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_model_comparison(
    seeds: tuple[int, ...], base: dict[str, Any],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"\n=== Repeated model comparison: seed {seed} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"models_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_model_comparison_run(base, seed, workspace)
            run_script("scripts/compare_models.py", config, workspace)
            frame = pd.read_csv(workspace / "summary.csv")
            frame.insert(0, "seed", seed)
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def run_open_set(seeds: tuple[int, ...], base: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        print(f"\n=== Repeated open set: seed {seed} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"open_set_{seed}_") as directory:
            workspace = Path(directory)
            config = configure_open_set_run(base, seed, workspace)
            run_script("scripts/train_open_set.py", config, workspace)
            with (workspace / "metrics.json").open("r", encoding="utf-8") as file:
                report = json.load(file)
            rows.append(extract_open_set_row(report, seed))
    return pd.DataFrame(rows)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the core dissertation experiments across seeds."
    )
    parser.add_argument("--config", default="config/repeated_experiments.yaml")
    parser.add_argument(
        "--sections", nargs="+",
        choices=("ablation", "model_comparison", "open_set"),
        default=("ablation", "model_comparison", "open_set"),
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=None,
        help="Optional seed override, useful for smoke tests or partial reruns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    config = load_yaml(args.config)
    seeds = validate_seeds(
        args.seeds if args.seeds is not None else config["seeds"]
    )
    experiments = config["experiments"]
    output_root = PROJECT_ROOT / config["output"]["root_directory"]

    if "ablation" in args.sections:
        runs = run_ablation(seeds, load_yaml(experiments["ablation_config"]))
        summary = aggregate_repeated_runs(
            runs, ("experiment_name", "feature_groups", "feature_count")
        )
        save_table(runs, output_root / "ablation_runs.csv")
        save_table(summary, output_root / "ablation_summary.csv")

    if "model_comparison" in args.sections:
        runs = run_model_comparison(
            seeds, load_yaml(experiments["model_comparison_config"])
        )
        summary = aggregate_repeated_runs(runs, ("model_name", "feature_count"))
        save_table(runs, output_root / "model_comparison_runs.csv")
        save_table(summary, output_root / "model_comparison_summary.csv")

    if "open_set" in args.sections:
        runs = run_open_set(seeds, load_yaml(experiments["open_set_config"]))
        runs_for_summary = runs.copy()
        runs_for_summary["experiment"] = "gradual_unseen"
        summary = aggregate_repeated_runs(runs_for_summary, ("experiment",))
        save_table(runs, output_root / "open_set_runs.csv")
        save_table(summary, output_root / "open_set_summary.csv")

    print(f"\nRepeated experiment results saved under {output_root}")


if __name__ == "__main__":
    main()
