from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.common.config import load_yaml_config
from src.drift.experiments import (
    evaluate_detector,
    evaluate_legal_adaptation,
    evaluate_poisoning,
    generate_step_drift,
)
from src.drift.page_hinkley import PageHinkley


def detector(config: dict[str, object]) -> PageHinkley:
    return PageHinkley(
        delta=float(config["delta"]),
        threshold=float(config["threshold"]),
        minimum_instances=int(config["minimum_instances"]),
    )


def run_once(config: dict[str, object], seed: int) -> dict[str, float | int]:
    streams = config["streams"]
    baseline = int(streams["baseline_samples"])
    drift_samples = int(streams["drift_samples"])
    measurement_config = streams["measurement"]
    communication_config = streams["communication"]

    measurement = generate_step_drift(
        baseline, drift_samples,
        float(measurement_config["baseline_mean"]),
        float(measurement_config["drift_mean"]),
        float(measurement_config["standard_deviation"]), seed,
    )
    communication = generate_step_drift(
        baseline, drift_samples,
        float(communication_config["baseline_mean"]),
        float(communication_config["drift_mean"]),
        float(communication_config["standard_deviation"]), seed + 100,
    )
    measurement_metrics, _ = evaluate_detector(
        measurement, baseline, detector(measurement_config["page_hinkley"])
    )
    communication_metrics, _ = evaluate_detector(
        communication,
        baseline,
        detector(communication_config["page_hinkley"]),
    )
    if measurement_metrics.first_detection_step is None:
        raise RuntimeError(f"Measurement drift not detected for seed {seed}")
    if communication_metrics.first_detection_step is None:
        raise RuntimeError(f"Communication drift not detected for seed {seed}")

    adaptation, _ = evaluate_legal_adaptation(
        values=measurement,
        change_step=baseline,
        drift_detection_step=measurement_metrics.first_detection_step,
        baseline_mean=float(measurement_config["baseline_mean"]),
        alert_deviation=float(config["adaptation"]["alert_deviation"]),
        drift_mean=float(measurement_config["drift_mean"]),
        reference_tolerance=float(
            config["adaptation"]["reference_tolerance"]
        ),
        adapter_config=config["adaptation"],
    )
    poisoning_config = config["poisoning"]
    poisoning, _ = evaluate_poisoning(
        baseline_samples=int(poisoning_config["baseline_samples"]),
        poisoning_samples=int(poisoning_config["poisoning_samples"]),
        initial_mean=float(poisoning_config["initial_mean"]),
        final_mean=float(poisoning_config["final_mean"]),
        standard_deviation=float(poisoning_config["standard_deviation"]),
        random_seed=seed + 200,
        adapter_config=config["adaptation"],
        trusted_deviation=float(poisoning_config["trusted_deviation"]),
        detector_config=poisoning_config["page_hinkley"],
    )
    return {
        "seed": seed,
        "measurement_detection_delay": int(measurement_metrics.detection_delay),
        "measurement_false_alarms": measurement_metrics.false_alarms_before_change,
        "communication_detection_delay": int(
            communication_metrics.detection_delay
        ),
        "communication_false_alarms": (
            communication_metrics.false_alarms_before_change
        ),
        "adaptation_delay": int(adaptation["adaptation_delay"]),
        "final_100_alert_rate": float(
            adaptation["guarded_final_100_alert_rate"]
        ),
        "guarded_reference_shift": float(
            poisoning["guarded_reference_shift"]
        ),
        "unguarded_reference_shift": float(
            poisoning["unguarded_reference_shift"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat controlled drift experiments across random seeds."
    )
    parser.add_argument("--config", default="config/drift.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    base_seed = int(config["random_seed"])
    run_count = int(config.get("repeated_runs", 5))
    if run_count <= 0:
        raise ValueError("repeated_runs must be greater than zero")

    frame = pd.DataFrame(
        [run_once(config, base_seed + index) for index in range(run_count)]
    )
    numeric = frame.drop(columns=["seed"])
    summary = {
        column: {
            "mean": float(numeric[column].mean()),
            "standard_deviation": float(numeric[column].std(ddof=1)),
            "minimum": float(numeric[column].min()),
            "maximum": float(numeric[column].max()),
        }
        for column in numeric.columns
    }
    output = config["output"]
    csv_path = Path(output["repeated_metrics"])
    json_path = Path(output["repeated_summary"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"runs": run_count, "seeds": frame["seed"].tolist(), "summary": summary},
            file,
            indent=2,
        )
    print("REPEATED DRIFT SUMMARY")
    print(frame.to_string(index=False))
    print("\nMean and standard deviation")
    print(numeric.agg(["mean", "std"]).to_string())
    print(f"\nSaved to: {json_path}")


if __name__ == "__main__":
    main()
