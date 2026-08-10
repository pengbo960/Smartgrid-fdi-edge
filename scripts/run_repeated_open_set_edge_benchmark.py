from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.evaluation.open_set_edge_benchmark import flatten_open_set_benchmark
from src.evaluation.repeated_experiments import aggregate_repeated_runs, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat the complete open-set streaming benchmark."
    )
    parser.add_argument(
        "--config", default="config/repeated_open_set_edge_benchmark.yaml"
    )
    parser.add_argument("--repetitions", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    repetitions = args.repetitions or int(config["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")

    rows: list[dict[str, object]] = []
    for run_id in range(1, repetitions + 1):
        print(f"Open-set edge benchmark {run_id}/{repetitions}", flush=True)
        with tempfile.TemporaryDirectory(prefix="open_set_edge_") as directory:
            report_path = Path(directory) / "report.json"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/benchmark_edge.py"),
                    "--config", str(config["edge_config"]),
                    "--input", str(config["input"]),
                    "--warmup-per-device",
                    str(config["warmup_messages_per_device"]),
                    "--output", str(report_path),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            with report_path.open("r", encoding="utf-8") as file:
                rows.append(flatten_open_set_benchmark(json.load(file), run_id))

    runs = pd.DataFrame(rows).sort_values("run_id")
    summary = aggregate_repeated_runs(
        runs,
        (
            "benchmark_mode", "classifier_class", "anomaly_detector_class",
            "platform_label",
        ),
    )
    save_table(runs, config["output"]["runs_csv"])
    save_table(summary, config["output"]["summary_csv"])
    print(f"Saved runs to {config['output']['runs_csv']}")
    print(f"Saved summary to {config['output']['summary_csv']}")


if __name__ == "__main__":
    main()
