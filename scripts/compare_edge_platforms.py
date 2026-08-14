from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.evaluation.platform_comparison import (
    build_platform_comparison,
    save_platform_comparison_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare repeated MacBook and Raspberry Pi edge benchmarks."
    )
    parser.add_argument("--config", default="config/platform_comparison.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    inputs = config["inputs"]
    comparison = build_platform_comparison(
        mac_model_summary=pd.read_csv(inputs["mac_model_summary"]),
        pi_model_summary=pd.read_csv(inputs["pi_model_summary"]),
        mac_open_set_summary=pd.read_csv(inputs["mac_open_set_summary"]),
        pi_open_set_summary=pd.read_csv(inputs["pi_open_set_summary"]),
    )
    output = config["output"]
    csv_path = Path(output["comparison_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    save_platform_comparison_figure(comparison, output["comparison_figure"])
    display_columns = [
        "pipeline_label",
        "mac_total_latency_mean_ms_mean",
        "pi_total_latency_mean_ms_mean",
        "pi_to_mac_latency_ratio",
        "mac_throughput_messages_per_second_mean",
        "pi_throughput_messages_per_second_mean",
        "mac_to_pi_throughput_ratio",
        "mac_process_peak_memory_after_mb_mean",
        "pi_process_peak_memory_after_mb_mean",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved table to: {csv_path}")
    print(f"Saved figure to: {output['comparison_figure']}")


if __name__ == "__main__":
    main()
