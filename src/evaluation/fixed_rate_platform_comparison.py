from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PIPELINE_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "open_set": "Open-set (LR + Isolation Forest)",
}


def concatenate_fixed_rate_summaries(
    summaries: list[pd.DataFrame],
    platform: str,
) -> pd.DataFrame:
    """Combine independently generated rate summaries without duplicates."""
    if not summaries:
        raise ValueError(f"No {platform} fixed-rate summaries were provided")
    combined = pd.concat(summaries, ignore_index=True)
    _validate_summary(combined, platform)
    return combined


def _validate_summary(frame: pd.DataFrame, platform: str) -> None:
    required = {
        "pipeline",
        "configured_message_rate",
        "achieved_message_rate_mean",
        "cpu_percent_single_core_equivalent_mean",
        "cpu_percent_single_core_equivalent_std",
        "cpu_time_per_message_ms_mean",
        "total_latency_mean_ms_mean",
        "deadline_misses_mean",
        "process_peak_memory_after_mb_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            f"{platform} fixed-rate summary is missing columns: {sorted(missing)}"
        )
    if frame.empty:
        raise ValueError(f"{platform} fixed-rate summary must not be empty")
    keys = frame[["pipeline", "configured_message_rate"]]
    if keys.duplicated().any():
        raise ValueError(f"{platform} fixed-rate summary contains duplicate groups")


def build_fixed_rate_platform_comparison(
    mac_summary: pd.DataFrame,
    pi_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Join fixed-rate platform summaries and calculate deployment ratios."""
    _validate_summary(mac_summary, "MacBook")
    _validate_summary(pi_summary, "Raspberry Pi")
    metrics = [
        "achieved_message_rate_mean",
        "cpu_percent_single_core_equivalent_mean",
        "cpu_percent_single_core_equivalent_std",
        "cpu_time_per_message_ms_mean",
        "total_latency_mean_ms_mean",
        "deadline_misses_mean",
        "process_peak_memory_after_mb_mean",
    ]
    pi_optional = [
        column for column in (
            "temperature_before_c_mean",
            "temperature_after_c_mean",
        )
        if column in pi_summary.columns
    ]
    keys = ["pipeline", "configured_message_rate"]
    comparison = mac_summary[keys + metrics].merge(
        pi_summary[keys + metrics + pi_optional],
        on=keys,
        how="outer",
        validate="one_to_one",
        suffixes=("_mac", "_pi"),
        indicator=True,
    )
    if not comparison["_merge"].eq("both").all():
        unmatched = comparison.loc[
            comparison["_merge"] != "both", keys + ["_merge"]
        ]
        raise ValueError(
            "MacBook and Raspberry Pi fixed-rate groups do not match: "
            f"{unmatched.to_dict(orient='records')}"
        )
    comparison = comparison.drop(columns="_merge")
    comparison.insert(
        1,
        "pipeline_label",
        comparison["pipeline"].map(PIPELINE_LABELS).fillna(
            comparison["pipeline"]
        ),
    )
    comparison.insert(
        3,
        "traffic_profile",
        np.where(
            comparison["configured_message_rate"].eq(6.0),
            "formal_normal_mqtt_load",
            "controlled_rate_sweep",
        ),
    )
    comparison["pi_to_mac_cpu_ratio"] = (
        comparison["cpu_percent_single_core_equivalent_mean_pi"]
        / comparison["cpu_percent_single_core_equivalent_mean_mac"]
    )
    comparison["pi_to_mac_cpu_time_per_message_ratio"] = (
        comparison["cpu_time_per_message_ms_mean_pi"]
        / comparison["cpu_time_per_message_ms_mean_mac"]
    )
    comparison["pi_to_mac_latency_ratio"] = (
        comparison["total_latency_mean_ms_mean_pi"]
        / comparison["total_latency_mean_ms_mean_mac"]
    )
    ordering = {name: index for index, name in enumerate(PIPELINE_LABELS)}
    comparison["_order"] = comparison["pipeline"].map(ordering)
    return comparison.sort_values(
        ["_order", "configured_message_rate"]
    ).drop(columns="_order").reset_index(drop=True)


def save_fixed_rate_cpu_figure(
    comparison: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """Plot single-core-equivalent process CPU against incoming rate."""
    if comparison.empty:
        raise ValueError("Fixed-rate platform comparison must not be empty")
    pipelines = list(PIPELINE_LABELS)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.7), sharey=True)
    for axis, pipeline in zip(axes, pipelines):
        selected = comparison.loc[
            comparison["pipeline"] == pipeline
        ].sort_values("configured_message_rate")
        if selected.empty:
            raise ValueError(f"Missing fixed-rate results for {pipeline}")
        rates = selected["configured_message_rate"].to_numpy(dtype=float)
        if np.isclose(rates, 6.0).any():
            axis.axvspan(
                5.55,
                6.45,
                color="#6C757D",
                alpha=0.10,
                label="Normal load (6 msg/s)" if pipeline == pipelines[0] else None,
            )
        for platform, label, colour, marker, x_offset, label_offset in (
            ("mac", "MacBook", "#3478BF", "o", -0.12, (-8, 7)),
            ("pi", "Raspberry Pi 5", "#D9534F", "s", 0.12, (8, 7)),
        ):
            values = selected[
                f"cpu_percent_single_core_equivalent_mean_{platform}"
            ].to_numpy(dtype=float)
            errors = selected[
                f"cpu_percent_single_core_equivalent_std_{platform}"
            ].to_numpy(dtype=float)
            axis.errorbar(
                rates + x_offset,
                values,
                yerr=errors,
                marker=marker,
                markersize=6,
                linewidth=2,
                capsize=3,
                label=label,
                color=colour,
            )
            for rate, value in zip(rates + x_offset, values):
                axis.annotate(
                    f"{value:.1f}%",
                    (rate, value),
                    xytext=label_offset,
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=colour,
                )
        axis.set_title(PIPELINE_LABELS[pipeline])
        axis.set_xlabel("Incoming message rate (messages/s)")
        axis.set_xticks(rates)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Process CPU (% of one logical core)")
    axes[0].legend(frameon=False, loc="upper left")
    maximum = float(
        comparison[[
            "cpu_percent_single_core_equivalent_mean_mac",
            "cpu_percent_single_core_equivalent_mean_pi",
        ]].to_numpy(dtype=float).max()
    )
    axes[0].set_ylim(0.0, max(10.0, maximum * 1.18))
    figure.suptitle(
        "CPU utilisation under fixed incoming load "
        "(6 msg/s = formal normal MQTT rate; mean ± sample SD, five runs)"
    )
    figure.tight_layout()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(figure)
