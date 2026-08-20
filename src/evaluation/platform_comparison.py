from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "open_set": "Open-set (LR + Isolation Forest)",
}

METRICS = {
    "throughput_messages_per_second": "throughput_messages_per_second",
    "total_latency_mean_ms": "total_latency_mean_ms",
    "total_latency_p95_ms": "total_latency_p95_ms",
    "model_inference_mean_ms": "model_inference_mean_ms",
    "cpu_percent_single_core_equivalent": "cpu_percent_single_core_equivalent",
    "process_peak_memory_after_mb": "process_peak_memory_after_mb",
}


def _extract_summary_row(
    frame: pd.DataFrame,
    pipeline: str,
) -> dict[str, float]:
    if frame.empty:
        raise ValueError(f"Summary for {pipeline} must not be empty")
    row = frame.iloc[0]
    output: dict[str, float] = {}
    for output_name, source_prefix in METRICS.items():
        mean_column = f"{source_prefix}_mean"
        std_column = f"{source_prefix}_std"
        missing = [
            column for column in (mean_column, std_column)
            if column not in frame.columns
        ]
        if missing:
            raise ValueError(
                f"Summary for {pipeline} is missing columns: {missing}"
            )
        output[f"{output_name}_mean"] = float(row[mean_column])
        output[f"{output_name}_std"] = float(row[std_column])
    return output


def _select_model_summary(
    frame: pd.DataFrame,
    model_name: str,
) -> pd.DataFrame:
    if "model_name" not in frame.columns:
        raise ValueError("Model benchmark summary is missing model_name")
    selected = frame.loc[frame["model_name"] == model_name]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {model_name}, found {len(selected)}"
        )
    return selected


def build_platform_comparison(
    mac_model_summary: pd.DataFrame,
    pi_model_summary: pd.DataFrame,
    mac_open_set_summary: pd.DataFrame,
    pi_open_set_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build one thesis-ready MacBook versus Raspberry Pi comparison table."""
    sources: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]] = {
        "logistic_regression": (
            _select_model_summary(mac_model_summary, "logistic_regression"),
            _select_model_summary(pi_model_summary, "logistic_regression"),
        ),
        "random_forest": (
            _select_model_summary(mac_model_summary, "random_forest"),
            _select_model_summary(pi_model_summary, "random_forest"),
        ),
        "open_set": (mac_open_set_summary, pi_open_set_summary),
    }
    rows: list[dict[str, object]] = []
    for pipeline, (mac_frame, pi_frame) in sources.items():
        mac = _extract_summary_row(mac_frame, pipeline)
        pi = _extract_summary_row(pi_frame, pipeline)
        row: dict[str, object] = {
            "pipeline": pipeline,
            "pipeline_label": PIPELINE_LABELS[pipeline],
        }
        for metric in METRICS:
            row[f"mac_{metric}_mean"] = mac[f"{metric}_mean"]
            row[f"mac_{metric}_std"] = mac[f"{metric}_std"]
            row[f"pi_{metric}_mean"] = pi[f"{metric}_mean"]
            row[f"pi_{metric}_std"] = pi[f"{metric}_std"]
        row["pi_to_mac_latency_ratio"] = (
            pi["total_latency_mean_ms_mean"]
            / mac["total_latency_mean_ms_mean"]
        )
        row["mac_to_pi_throughput_ratio"] = (
            mac["throughput_messages_per_second_mean"]
            / pi["throughput_messages_per_second_mean"]
        )
        row["pi_to_mac_inference_ratio"] = (
            pi["model_inference_mean_ms_mean"]
            / mac["model_inference_mean_ms_mean"]
        )
        row["pi_memory_change_percent"] = (
            pi["process_peak_memory_after_mb_mean"]
            / mac["process_peak_memory_after_mb_mean"]
            - 1.0
        ) * 100.0
        rows.append(row)
    return pd.DataFrame(rows)


def save_platform_comparison_figure(
    comparison: pd.DataFrame,
    output_path: str,
) -> None:
    """Save processing-latency, throughput and memory comparisons."""
    if comparison.empty:
        raise ValueError("Platform comparison must not be empty")
    labels = comparison["pipeline_label"].tolist()
    positions = np.arange(len(labels), dtype=float)
    width = 0.36
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    panels = (
        (
            "total_latency_mean_ms",
            "Mean processing latency\n(receive-to-decision, ms)",
            True,
            lambda value: f"{value:.2f}",
        ),
        (
            "throughput_messages_per_second",
            "Throughput (messages/s)",
            True,
            lambda value: f"{value:,.0f}",
        ),
        (
            "process_peak_memory_after_mb",
            "Peak detector-process RSS (MB)",
            False,
            lambda value: f"{value:.1f}",
        ),
    )
    for axis, (metric, ylabel, use_log_scale, formatter) in zip(axes, panels):
        for offset, platform, label, colour in (
            (-width / 2, "mac", "MacBook", "#3478BF"),
            (width / 2, "pi", "Raspberry Pi 5", "#D9534F"),
        ):
            values = comparison[f"{platform}_{metric}_mean"]
            bars = axis.bar(
                positions + offset,
                values,
                width,
                yerr=comparison[f"{platform}_{metric}_std"],
                capsize=3,
                label=label,
                color=colour,
                alpha=0.9,
            )
            axis.bar_label(
                bars,
                labels=[formatter(float(value)) for value in values],
                padding=4,
                fontsize=8,
            )
        axis.set_ylabel(ylabel)
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if use_log_scale:
            axis.set_yscale("log")
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower, upper * (1.35 if use_log_scale else 1.12))
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Cross-platform edge inference cost (mean ± sample SD, five runs)"
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
