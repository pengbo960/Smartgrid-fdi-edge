from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.drift.live_evaluation import evaluate_live_drift


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarise labelled live MQTT drift experiments."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/drift/live_mqtt_summary.json"),
    )
    args = parser.parse_args()

    report = {
        path.stem: evaluate_live_drift(pd.read_csv(path))
        for path in args.inputs
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    for name, metrics in report.items():
        print(f"\n{name}")
        for key, value in metrics.items():
            print(f"{key}: {value}")
    print(f"\nSummary saved to: {args.output}")


if __name__ == "__main__":
    main()
