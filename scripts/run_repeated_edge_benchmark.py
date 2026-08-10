from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.evaluation.repeated_experiments import aggregate_repeated_runs, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed LR/RF deployment artifacts in fresh processes."
    )
    parser.add_argument("--config", default="config/repeated_edge_benchmark.yaml")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    repetitions = args.repetitions or int(config["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    models = args.models or list(config["models"])
    unknown = set(models) - set(config["models"])
    if unknown:
        raise ValueError(f"Unknown benchmark models: {sorted(unknown)}")

    rows: list[dict[str, object]] = []
    for run_id in range(1, repetitions + 1):
        # Alternate order to reduce systematic temperature/order bias.
        run_models = models if run_id % 2 else list(reversed(models))
        for model_name in run_models:
            print(f"Benchmark run {run_id}/{repetitions}: {model_name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="edge_benchmark_") as directory:
                output = Path(directory) / "report.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts/benchmark_comparison_edge.py"),
                        "--config", str(args.config),
                        "--model", model_name,
                        "--run-id", str(run_id),
                        "--output", str(output),
                    ],
                    cwd=PROJECT_ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                with output.open("r", encoding="utf-8") as file:
                    rows.append(json.load(file))

    runs = pd.DataFrame(rows).sort_values(["model_name", "run_id"])
    summary = aggregate_repeated_runs(
        runs,
        ("model_name", "model_class", "benchmark_mode", "feature_count"),
    )
    save_table(runs, config["output"]["runs_csv"])
    save_table(summary, config["output"]["summary_csv"])
    print(f"Saved runs to {config['output']['runs_csv']}")
    print(f"Saved summary to {config['output']['summary_csv']}")


if __name__ == "__main__":
    main()
