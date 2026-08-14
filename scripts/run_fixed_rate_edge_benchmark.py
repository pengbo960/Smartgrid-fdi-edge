from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.evaluation.fixed_rate_benchmark import validate_message_rates
from src.evaluation.repeated_experiments import aggregate_repeated_runs, save_table


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat fixed-rate CPU benchmarks in fresh processes."
    )
    parser.add_argument("--config", default="config/fixed_rate_edge_benchmark.yaml")
    parser.add_argument("--repetitions", type=int, default=None)
    parser.add_argument("--pipelines", nargs="+", default=None)
    parser.add_argument("--message-rates", nargs="+", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    repetitions = args.repetitions or int(config["repetitions"])
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    pipelines = args.pipelines or list(config["pipelines"])
    unknown = set(pipelines) - set(config["pipelines"])
    if unknown:
        raise ValueError(f"Unknown fixed-rate pipelines: {sorted(unknown)}")
    rates = validate_message_rates(args.message_rates or config["message_rates"])

    rows: list[dict[str, object]] = []
    for run_id in range(1, repetitions + 1):
        run_pipelines = pipelines if run_id % 2 else list(reversed(pipelines))
        run_rates = rates if run_id % 2 else tuple(reversed(rates))
        for message_rate in run_rates:
            for pipeline in run_pipelines:
                print(
                    f"Fixed-rate run {run_id}/{repetitions}: "
                    f"{pipeline} at {message_rate:g} messages/s",
                    flush=True,
                )
                with tempfile.TemporaryDirectory(prefix="fixed_rate_edge_") as directory:
                    output = Path(directory) / "report.json"
                    subprocess.run(
                        [
                            sys.executable,
                            str(PROJECT_ROOT / "scripts/benchmark_fixed_rate_edge.py"),
                            "--config", str(args.config),
                            "--pipeline", pipeline,
                            "--message-rate", str(message_rate),
                            "--run-id", str(run_id),
                            "--output", str(output),
                        ],
                        cwd=PROJECT_ROOT,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                    with output.open("r", encoding="utf-8") as file:
                        rows.append(json.load(file))

    runs = pd.DataFrame(rows).sort_values(
        ["pipeline", "configured_message_rate", "run_id"]
    )
    metrics = [
        "achieved_message_rate",
        "deadline_misses",
        "deadline_miss_rate",
        "elapsed_seconds",
        "total_latency_mean_ms",
        "total_latency_p95_ms",
        "model_inference_mean_ms",
        "process_cpu_seconds",
        "cpu_time_per_message_ms",
        "cpu_percent_single_core_equivalent",
        "cpu_percent_total_machine_capacity",
        "process_peak_memory_before_mb",
        "process_peak_memory_after_mb",
        "temperature_before_c",
        "temperature_after_c",
    ]
    summary = aggregate_repeated_runs(
        runs,
        (
            "pipeline",
            "pipeline_mode",
            "benchmark_mode",
            "platform_label",
            "configured_message_rate",
        ),
        metric_columns=metrics,
    )
    save_table(runs, config["output"]["runs_csv"])
    save_table(summary, config["output"]["summary_csv"])
    print(f"Saved runs to {config['output']['runs_csv']}")
    print(f"Saved summary to {config['output']['summary_csv']}")


if __name__ == "__main__":
    main()
