from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCENARIO_LABELS = {
    "measurement": "Measurement drift",
    "communication": "Communication drift",
}

METRICS = (
    "rows",
    "mean_detection_delay_messages",
    "active_alert_reduction_percent",
    "adaptation_updates",
    "latency_mean_ms",
    "latency_p95_ms",
    "latency_p99_ms",
)


def _select_scenario(
    summary: pd.DataFrame,
    scenario_type: str,
    platform: str,
) -> pd.Series:
    if "scenario_type" not in summary.columns:
        raise ValueError(f"{platform} summary is missing scenario_type")
    selected = summary.loc[summary["scenario_type"] == scenario_type]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {scenario_type} row in {platform} summary, "
            f"found {len(selected)}"
        )
    required = {"runs"}
    for metric in METRICS:
        required.update({f"{metric}_mean", f"{metric}_std"})
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"{platform} summary is missing columns: {missing}")
    return selected.iloc[0]


def build_drift_platform_comparison(
    mac_summary: pd.DataFrame,
    pi_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a five-run MacBook versus Raspberry Pi drift comparison."""
    rows: list[dict[str, object]] = []
    for scenario_type, scenario_label in SCENARIO_LABELS.items():
        platform_rows = {
            "MacBook": _select_scenario(
                mac_summary, scenario_type, "MacBook"
            ),
            "Raspberry Pi 5": _select_scenario(
                pi_summary, scenario_type, "Raspberry Pi 5"
            ),
        }
        mac_latency = float(platform_rows["MacBook"]["latency_mean_ms_mean"])
        pi_latency = float(
            platform_rows["Raspberry Pi 5"]["latency_mean_ms_mean"]
        )
        for platform, source in platform_rows.items():
            row: dict[str, object] = {
                "scenario": f"{scenario_type}_drift",
                "scenario_type": scenario_type,
                "scenario_label": scenario_label,
                "platform": platform,
                "runs": int(source["runs"]),
                "pi_to_mac_latency_ratio": pi_latency / mac_latency,
            }
            for metric in METRICS:
                row[f"{metric}_mean"] = float(source[f"{metric}_mean"])
                row[f"{metric}_std"] = float(source[f"{metric}_std"])
            rows.append(row)
    return pd.DataFrame(rows)


def save_drift_platform_figure(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Save labelled five-run means with sample-standard-deviation bars."""
    if comparison.empty:
        raise ValueError("Drift platform comparison must not be empty")

    scenarios = list(SCENARIO_LABELS)
    labels = [SCENARIO_LABELS[item] for item in scenarios]
    positions = np.arange(len(scenarios), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = (
        (
            "mean_detection_delay_messages",
            "Detection delay (messages)",
            lambda value: f"{value:.1f}",
        ),
        (
            "active_alert_reduction_percent",
            "Active-period alert reduction (%)",
            lambda value: f"{value:.1f}%",
        ),
        (
            "latency_mean_ms",
            "Mean processing latency (ms)",
            lambda value: f"{value:.2f}",
        ),
    )

    for axis, (metric, ylabel, formatter) in zip(axes, panels):
        for offset, platform, colour in (
            (-width / 2, "MacBook", "#3478BF"),
            (width / 2, "Raspberry Pi 5", "#D9534F"),
        ):
            platform_rows = comparison[comparison["platform"] == platform]
            values = []
            errors = []
            for scenario_type in scenarios:
                selected = platform_rows[
                    platform_rows["scenario_type"] == scenario_type
                ].iloc[0]
                values.append(float(selected[f"{metric}_mean"]))
                errors.append(float(selected[f"{metric}_std"]))
            bars = axis.bar(
                positions + offset,
                values,
                width,
                yerr=errors,
                capsize=3,
                label=platform,
                color=colour,
                alpha=0.9,
            )
            axis.bar_label(
                bars,
                labels=[formatter(value) for value in values],
                padding=5,
                fontsize=8,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=14, ha="right")
        axis.grid(axis="y", alpha=0.25)
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower, upper * 1.16)

    axes[0].legend(frameon=False)
    figure.suptitle(
        "Cross-platform live MQTT drift evaluation "
        "(mean ± sample SD, five runs)"
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
