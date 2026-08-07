from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.drift.guarded_adaptation import ReferenceAdapter
from src.drift.page_hinkley import PageHinkley


@dataclass(frozen=True)
class DriftDetectionMetrics:
    change_step: int
    first_detection_step: int | None
    detection_delay: int | None
    false_alarms_before_change: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "change_step": self.change_step,
            "first_detection_step": self.first_detection_step,
            "detection_delay": self.detection_delay,
            "false_alarms_before_change": self.false_alarms_before_change,
        }


def generate_step_drift(
    baseline_samples: int,
    drift_samples: int,
    baseline_mean: float,
    drift_mean: float,
    standard_deviation: float,
    random_seed: int,
) -> np.ndarray:
    if baseline_samples <= 0 or drift_samples <= 0:
        raise ValueError("Stream sections must contain samples")
    if standard_deviation < 0:
        raise ValueError("standard_deviation must be non-negative")
    rng = np.random.default_rng(random_seed)
    return np.concatenate(
        [
            rng.normal(baseline_mean, standard_deviation, baseline_samples),
            rng.normal(drift_mean, standard_deviation, drift_samples),
        ]
    )


def evaluate_detector(
    values: np.ndarray,
    change_step: int,
    detector: PageHinkley,
) -> tuple[DriftDetectionMetrics, list[int]]:
    detections: list[int] = []
    for step, value in enumerate(values):
        if detector.update(float(value)).drift_detected:
            detections.append(step)
    after_change = [step for step in detections if step >= change_step]
    first = after_change[0] if after_change else None
    return (
        DriftDetectionMetrics(
            change_step=change_step,
            first_detection_step=first,
            detection_delay=None if first is None else first - change_step,
            false_alarms_before_change=sum(
                step < change_step for step in detections
            ),
        ),
        detections,
    )


def evaluate_legal_adaptation(
    values: np.ndarray,
    change_step: int,
    drift_detection_step: int,
    baseline_mean: float,
    alert_deviation: float,
    drift_mean: float,
    reference_tolerance: float,
    adapter_config: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray]:
    adapter = ReferenceAdapter(
        initial_mean=baseline_mean,
        window_size=int(adapter_config["window_size"]),
        minimum_samples=int(adapter_config["minimum_samples"]),
        blend_rate=float(adapter_config["blend_rate"]),
        maximum_update_step=float(adapter_config["maximum_update_step"]),
        guarded=True,
    )
    references: list[float] = []
    static_alerts: list[bool] = []
    adapted_alerts: list[bool] = []
    for step, value in enumerate(values):
        confirmed = step >= drift_detection_step
        adapter.update(
            float(value),
            trusted_sample=True,
            drift_confirmed=confirmed,
        )
        references.append(adapter.reference_mean)
        static_alerts.append(abs(value - baseline_mean) > alert_deviation)
        adapted_alerts.append(
            abs(value - adapter.reference_mean) > alert_deviation
        )

    post = slice(change_step, None)
    within_tolerance = np.flatnonzero(
        np.abs(np.asarray(references) - drift_mean) <= reference_tolerance
    )
    after_change = within_tolerance[within_tolerance >= change_step]
    return (
        {
            "static_post_drift_alert_rate": float(np.mean(static_alerts[post])),
            "guarded_post_drift_alert_rate": float(np.mean(adapted_alerts[post])),
            "guarded_final_100_alert_rate": float(
                np.mean(adapted_alerts[-100:])
            ),
            "adaptation_delay": (
                None
                if len(after_change) == 0
                else int(after_change[0] - change_step)
            ),
            "final_reference_mean": adapter.reference_mean,
            "reference_updates": adapter.update_count,
        },
        np.asarray(references),
    )


def evaluate_poisoning(
    baseline_samples: int,
    poisoning_samples: int,
    initial_mean: float,
    final_mean: float,
    standard_deviation: float,
    random_seed: int,
    adapter_config: dict[str, Any],
    trusted_deviation: float,
    detector_config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    means = np.concatenate(
        [
            np.full(baseline_samples, initial_mean),
            np.linspace(initial_mean, final_mean, poisoning_samples),
        ]
    )
    values = means + rng.normal(0.0, standard_deviation, len(means))

    common = {
        "initial_mean": initial_mean,
        "window_size": int(adapter_config["window_size"]),
        "minimum_samples": int(adapter_config["minimum_samples"]),
        "blend_rate": float(adapter_config["blend_rate"]),
        "maximum_update_step": float(adapter_config["maximum_update_step"]),
    }
    guarded = ReferenceAdapter(**common, guarded=True)
    unguarded = ReferenceAdapter(**common, guarded=False)
    guarded_references: list[float] = []
    unguarded_references: list[float] = []
    detector = PageHinkley(
        delta=float(detector_config["delta"]),
        threshold=float(detector_config["threshold"]),
        minimum_instances=int(detector_config["minimum_instances"]),
        reset_after_drift=False,
    )
    drift_confirmed = False
    confirmation_step: int | None = None
    rejected_after_confirmation = 0
    for step, value in enumerate(values):
        if detector.update(float(value)).drift_detected:
            drift_confirmed = True
            if confirmation_step is None:
                confirmation_step = step
        trusted = abs(float(value) - initial_mean) <= trusted_deviation
        guarded_result = guarded.update(
            float(value),
            trusted_sample=trusted,
            drift_confirmed=drift_confirmed,
        )
        if drift_confirmed and not guarded_result.accepted:
            rejected_after_confirmation += 1
        unguarded.update(
            float(value), trusted_sample=False, drift_confirmed=False
        )
        guarded_references.append(guarded.reference_mean)
        unguarded_references.append(unguarded.reference_mean)

    metrics = {
        "guarded_reference_shift": guarded.reference_mean - initial_mean,
        "unguarded_reference_shift": unguarded.reference_mean - initial_mean,
        "guarded_updates": guarded.update_count,
        "unguarded_updates": unguarded.update_count,
        "drift_confirmation_step": confirmation_step,
        "guarded_rejected_after_confirmation": rejected_after_confirmation,
    }
    frame = pd.DataFrame(
        {
            "step": np.arange(len(values)),
            "poisoning_value": values,
            "poisoning_target_mean": means,
            "guarded_reference": guarded_references,
            "unguarded_reference": unguarded_references,
        }
    )
    return metrics, frame


def save_drift_outputs(
    measurement_values: np.ndarray,
    measurement_detections: list[int],
    adapted_references: np.ndarray,
    communication_values: np.ndarray,
    communication_detections: list[int],
    poisoning_frame: pd.DataFrame,
    change_step: int,
    metrics: dict[str, Any],
    metrics_path: str | Path,
    time_series_path: str | Path,
    figure_path: str | Path,
) -> None:
    paths = [Path(metrics_path), Path(time_series_path), Path(figure_path)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    with paths[0].open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    length = max(len(measurement_values), len(poisoning_frame))
    combined = pd.DataFrame({"step": np.arange(length)})
    measurement = pd.DataFrame(
        {
            "step": np.arange(len(measurement_values)),
            "measurement_value": measurement_values,
            "adapted_reference": adapted_references,
        }
    )
    communication = pd.DataFrame(
        {
            "step": np.arange(len(communication_values)),
            "communication_value": communication_values,
        }
    )
    combined = combined.merge(measurement, on="step", how="left")
    combined = combined.merge(communication, on="step", how="left")
    combined = combined.merge(poisoning_frame, on="step", how="left")
    combined.to_csv(paths[1], index=False)

    figure, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    axes[0].plot(measurement_values, label="Voltage stream", linewidth=1)
    axes[0].plot(adapted_references, label="Guarded reference", linewidth=2)
    axes[0].axvline(change_step, color="black", linestyle="--", label="True drift")
    for step in measurement_detections:
        axes[0].axvline(step, color="red", alpha=0.4)
    axes[0].set_ylabel("Voltage (V)")
    axes[0].set_title("Measurement drift and guarded adaptation")
    axes[0].legend()

    axes[1].plot(communication_values, label="Publish interval", color="#2563eb")
    axes[1].axvline(change_step, color="black", linestyle="--", label="True drift")
    for step in communication_detections:
        axes[1].axvline(step, color="red", alpha=0.4)
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Communication-rate drift")
    axes[1].legend()

    axes[2].plot(
        poisoning_frame["poisoning_value"],
        color="lightgray",
        label="Poisoning stream",
    )
    axes[2].plot(
        poisoning_frame["guarded_reference"],
        label="Guarded reference",
        linewidth=2,
    )
    axes[2].plot(
        poisoning_frame["unguarded_reference"],
        label="Unguarded reference",
        linewidth=2,
    )
    axes[2].set_xlabel("Message step")
    axes[2].set_ylabel("Voltage (V)")
    axes[2].set_title("Resistance to unconfirmed gradual poisoning")
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(paths[2], dpi=200, bbox_inches="tight")
    plt.close(figure)
