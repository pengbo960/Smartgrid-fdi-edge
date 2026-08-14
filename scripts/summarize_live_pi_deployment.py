from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.live_deployment import summarize_live_deployment_logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise Raspberry Pi real MQTT deployment logs."
    )
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/edge/raspberry_pi/live_mqtt_summary.csv"),
    )
    args = parser.parse_args()
    summary = summarize_live_deployment_logs(args.logs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    main()
