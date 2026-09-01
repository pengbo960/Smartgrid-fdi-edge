from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot illustrative guarded-adaptation and poisoning-reference "
            "trajectories from the archived drift experiment."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/drift/drift_time_series.csv"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("results/drift/drift_metrics.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "dissertation/images/guarded_adaptation_poisoning.png"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    with args.metrics.open("r", encoding="utf-8") as file:
        metrics = json.load(file)

    measurement = frame.dropna(
        subset=["measurement_value", "adapted_reference"]
    )
    poisoning = frame.dropna(
        subset=[
            "poisoning_value",
            "poisoning_target_mean",
            "guarded_reference",
            "unguarded_reference",
        ]
    )

    change_step = int(metrics["measurement_drift"]["change_step"])
    detection_step = int(
        metrics["measurement_drift"]["first_detection_step"]
    )
    confirmation_step = int(
        metrics["poisoning_resistance"]["drift_confirmation_step"]
    )

    target = poisoning["poisoning_target_mean"].to_numpy(dtype=float)
    changed = np.flatnonzero(~np.isclose(target, target[0]))
    poisoning_start = max(0, int(changed[0]) - 1) if len(changed) else 0

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 7.2))

    axes[0].plot(
        measurement["step"],
        measurement["measurement_value"],
        color="#4C78A8",
        linewidth=1.0,
        alpha=0.85,
        label="Voltage stream",
    )
    axes[0].plot(
        measurement["step"],
        measurement["adapted_reference"],
        color="#F58518",
        linewidth=2.2,
        label="Guarded reference",
    )
    axes[0].axvline(
        change_step,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="True drift",
    )
    axes[0].axvline(
        detection_step,
        color="#E45756",
        linestyle=":",
        linewidth=1.8,
        label="Drift detected",
    )
    axes[0].set_title(
        "(a) Guarded reference adaptation after measurement drift",
        fontweight="bold",
    )
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_xlabel("Message step")
    axes[0].legend(ncol=2, frameon=True, loc="lower right")

    axes[1].plot(
        poisoning["step"],
        poisoning["poisoning_value"],
        color="#B8B8B8",
        linewidth=0.9,
        alpha=0.75,
        label="Poisoning stream",
    )
    axes[1].plot(
        poisoning["step"],
        poisoning["guarded_reference"],
        color="#4C78A8",
        linewidth=2.2,
        label="Guarded reference",
    )
    axes[1].plot(
        poisoning["step"],
        poisoning["unguarded_reference"],
        color="#F58518",
        linewidth=2.2,
        label="Unguarded reference",
    )
    axes[1].axvline(
        poisoning_start,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label="Poisoning begins",
    )
    axes[1].axvline(
        confirmation_step,
        color="#E45756",
        linestyle=":",
        linewidth=1.8,
        label="Drift confirmed",
    )
    axes[1].set_title(
        "(b) Guarded and unguarded reference movement under poisoning",
        fontweight="bold",
    )
    axes[1].set_xlabel("Message step")
    axes[1].set_ylabel("Voltage (V)")
    axes[1].legend(ncol=2, frameon=True, loc="upper left")

    for axis in axes:
        axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(h_pad=2.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved figure to {args.output}")


if __name__ == "__main__":
    main()
