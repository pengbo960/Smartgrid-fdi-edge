from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.drift.live_evaluation import build_live_drift_phase_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create phase-wise metrics from labelled live MQTT drift trials."
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/drift/live_mqtt_phase_metrics.csv"),
    )
    args = parser.parse_args()
    frames: list[pd.DataFrame] = []
    for path in args.inputs:
        metrics = build_live_drift_phase_metrics(pd.read_csv(path))
        metrics.insert(0, "experiment", path.stem)
        frames.append(metrics)
    result = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nPhase metrics saved to: {args.output}")


if __name__ == "__main__":
    main()
