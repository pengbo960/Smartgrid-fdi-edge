from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.evaluation.fixed_rate_platform_comparison import (
    build_fixed_rate_platform_comparison,
    save_fixed_rate_cpu_figure,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare MacBook and Raspberry Pi fixed-rate CPU results."
    )
    parser.add_argument(
        "--config", default="config/fixed_rate_platform_comparison.yaml"
    )
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    comparison = build_fixed_rate_platform_comparison(
        pd.read_csv(config["inputs"]["mac_summary"]),
        pd.read_csv(config["inputs"]["pi_summary"]),
    )
    output = config["output"]
    csv_path = Path(output["comparison_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    save_fixed_rate_cpu_figure(comparison, output["cpu_figure"])
    columns = [
        "pipeline_label",
        "configured_message_rate",
        "cpu_percent_single_core_equivalent_mean_mac",
        "cpu_percent_single_core_equivalent_std_mac",
        "cpu_percent_single_core_equivalent_mean_pi",
        "cpu_percent_single_core_equivalent_std_pi",
        "pi_to_mac_cpu_ratio",
        "deadline_misses_mean_mac",
        "deadline_misses_mean_pi",
    ]
    print(comparison[columns].to_string(index=False))
    print(f"\nSaved table to: {csv_path}")
    print(f"Saved figure to: {output['cpu_figure']}")


if __name__ == "__main__":
    main()
