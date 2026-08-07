from __future__ import annotations

import argparse

from src.common.config import load_yaml_config
from src.drift.experiments import (
    evaluate_detector,
    evaluate_legal_adaptation,
    evaluate_poisoning,
    generate_step_drift,
    save_drift_outputs,
)
from src.drift.page_hinkley import PageHinkley


def build_detector(config: dict[str, object]) -> PageHinkley:
    return PageHinkley(
        delta=float(config["delta"]),
        threshold=float(config["threshold"]),
        minimum_instances=int(config["minimum_instances"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate drift detection and guarded adaptation."
    )
    parser.add_argument("--config", default="config/drift.yaml")
    args = parser.parse_args()
    config = load_yaml_config(args.config)
    streams = config["streams"]
    baseline_samples = int(streams["baseline_samples"])
    drift_samples = int(streams["drift_samples"])
    seed = int(config["random_seed"])

    measurement_config = streams["measurement"]
    communication_config = streams["communication"]
    measurement = generate_step_drift(
        baseline_samples, drift_samples,
        float(measurement_config["baseline_mean"]),
        float(measurement_config["drift_mean"]),
        float(measurement_config["standard_deviation"]), seed,
    )
    communication = generate_step_drift(
        baseline_samples, drift_samples,
        float(communication_config["baseline_mean"]),
        float(communication_config["drift_mean"]),
        float(communication_config["standard_deviation"]), seed + 1,
    )
    measurement_metrics, measurement_detections = evaluate_detector(
        measurement, baseline_samples,
        build_detector(measurement_config["page_hinkley"]),
    )
    communication_metrics, communication_detections = evaluate_detector(
        communication, baseline_samples,
        build_detector(communication_config["page_hinkley"]),
    )
    if measurement_metrics.first_detection_step is None:
        raise RuntimeError("Measurement drift was not detected")

    adaptation_metrics, references = evaluate_legal_adaptation(
        values=measurement,
        change_step=baseline_samples,
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
    poisoning_metrics, poisoning_frame = evaluate_poisoning(
        baseline_samples=int(poisoning_config["baseline_samples"]),
        poisoning_samples=int(poisoning_config["poisoning_samples"]),
        initial_mean=float(poisoning_config["initial_mean"]),
        final_mean=float(poisoning_config["final_mean"]),
        standard_deviation=float(poisoning_config["standard_deviation"]),
        random_seed=seed + 2,
        adapter_config=config["adaptation"],
        trusted_deviation=float(poisoning_config["trusted_deviation"]),
        detector_config=poisoning_config["page_hinkley"],
    )
    metrics = {
        "measurement_drift": measurement_metrics.to_dict(),
        "communication_drift": communication_metrics.to_dict(),
        "guarded_adaptation": adaptation_metrics,
        "poisoning_resistance": poisoning_metrics,
    }
    output = config["output"]
    save_drift_outputs(
        measurement, measurement_detections, references,
        communication, communication_detections, poisoning_frame,
        baseline_samples, metrics, output["metrics"],
        output["time_series"], output["figure"],
    )
    print("DRIFT EVALUATION")
    for section, values in metrics.items():
        print(f"\n{section}")
        for name, value in values.items():
            print(f"{name}: {value}")
    print(f"\nMetrics saved to: {output['metrics']}")
    print(f"Figure saved to: {output['figure']}")


if __name__ == "__main__":
    main()
