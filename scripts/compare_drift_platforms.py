from __future__ import annotations

import argparse
from pathlib import Path

from src.common.config import load_yaml_config
from src.evaluation.drift_platform_comparison import (
    build_drift_platform_comparison,
    save_drift_platform_figure,
)
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MacBook and Raspberry Pi live MQTT drift trials."
    )
    parser.add_argument(
        "--config",
        default="config/drift_platform_comparison.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    comparison = build_drift_platform_comparison(
        mac_summary=pd.read_csv(config["inputs"]["mac_summary"]),
        pi_summary=pd.read_csv(config["inputs"]["pi_summary"]),
    )
    csv_path = Path(config["output"]["comparison_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(csv_path, index=False)
    save_drift_platform_figure(
        comparison,
        config["output"]["comparison_figure"],
    )
    display_columns = [
        "scenario_label",
        "platform",
        "runs",
        "mean_detection_delay_messages_mean",
        "mean_detection_delay_messages_std",
        "active_alert_reduction_percent_mean",
        "active_alert_reduction_percent_std",
        "adaptation_updates_mean",
        "latency_mean_ms_mean",
        "latency_mean_ms_std",
        "latency_p95_ms_mean",
        "pi_to_mac_latency_ratio",
    ]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSaved table to: {csv_path}")
    print(f"Saved figure to: {config['output']['comparison_figure']}")


if __name__ == "__main__":
    main()
