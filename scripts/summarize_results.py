from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCES = {
    "ablation": Path("results/v2/ablation/ablation_summary.csv"),
    "open_set": Path("results/open_set/metrics/open_set_metrics.json"),
    "edge": Path("results/edge/macbook_benchmark.json"),
    "edge_mac_repeated": Path(
        "results/edge/repeated/open_set_benchmark_summary.csv"
    ),
    "edge_pi_repeated": Path(
        "results/edge/raspberry_pi/open_set_benchmark_summary.csv"
    ),
    "platform_comparison": Path(
        "results/edge/platform_comparison/platform_comparison.csv"
    ),
    "live_mqtt": Path(
        "results/edge/raspberry_pi/live_mqtt_matched/"
        "live_mqtt_formal_summary.csv"
    ),
    "model_comparison": Path("results/model_comparison/model_comparison.csv"),
    "drift": Path("results/drift/drift_repeated_summary.json"),
    "pi_live_drift": Path(
        "results/drift/raspberry_pi_repeated_live/live_drift_summary.csv"
    ),
    "pi_drift_thermal": Path(
        "results/drift/raspberry_pi_repeated_live/thermal_observations.csv"
    ),
    "drift_platform_comparison": Path(
        "results/drift/platform_comparison.csv"
    ),
    "normal_load_cpu": Path(
        "results/edge/normal_load_cpu/platform_comparison.csv"
    ),
    "normal_load_thermal": Path(
        "results/edge/normal_load_cpu/thermal_observation.csv"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_summary() -> tuple[dict[str, Any], pd.DataFrame]:
    missing = [str(path) for path in SOURCES.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing final result sources: " + ", ".join(missing))

    ablation = pd.read_csv(SOURCES["ablation"])
    models = pd.read_csv(SOURCES["model_comparison"])
    open_set = load_json(SOURCES["open_set"])
    edge = load_json(SOURCES["edge"])
    edge_mac_repeated = pd.read_csv(SOURCES["edge_mac_repeated"]).iloc[0]
    edge_pi_repeated = pd.read_csv(SOURCES["edge_pi_repeated"]).iloc[0]
    platform_comparison = pd.read_csv(SOURCES["platform_comparison"])
    live_mqtt = pd.read_csv(SOURCES["live_mqtt"])
    drift = load_json(SOURCES["drift"])
    pi_live_drift = pd.read_csv(SOURCES["pi_live_drift"])
    pi_drift_thermal = pd.read_csv(SOURCES["pi_drift_thermal"])
    drift_platform_comparison = pd.read_csv(
        SOURCES["drift_platform_comparison"]
    )
    normal_load_cpu = pd.read_csv(SOURCES["normal_load_cpu"])
    normal_load_thermal = pd.read_csv(SOURCES["normal_load_thermal"])

    all_views = ablation.loc[ablation["experiment_name"] == "all_views"].iloc[0]
    logistic = models.loc[models["model_name"] == "logistic_regression"].iloc[0]
    forest = models.loc[models["model_name"] == "random_forest"].iloc[0]
    drift_summary = drift["summary"]
    pi_live_by_scenario = pi_live_drift.set_index("scenario_type")
    pi_measurement_drift = pi_live_by_scenario.loc["measurement"]
    pi_communication_drift = pi_live_by_scenario.loc["communication"]
    live_by_scenario = live_mqtt.set_index("scenario_id")
    known_live = live_mqtt[
        live_mqtt["scenario_id"].isin(
            ["constant_01", "replay_01", "topic_spoof_01"]
        )
    ]
    gradual_live = live_by_scenario.loc["gradual_run_01"]
    live_normal_messages = live_mqtt["normal_messages"].sum()
    live_normal_alerts = (
        live_mqtt["normal_messages"] * live_mqtt["normal_alert_rate"]
    ).sum()
    live_messages = live_mqtt["messages"].sum()
    normal_cpu_by_pipeline = normal_load_cpu.set_index("pipeline")

    summary = {
        "multi_view": {
            "macro_f1": float(all_views["macro_f1"]),
            "recall": float(all_views["recall"]),
            "false_positive_rate": float(all_views["false_positive_rate"]),
        },
        "open_set": {
            "unseen_attack": open_set["dataset"]["unseen_attack_type"],
            "unknown_recall": float(open_set["unseen"]["unknown_recall"]),
            "unknown_precision": float(open_set["unseen"]["unknown_precision"]),
            "known_false_unknown_rate": float(
                open_set["known_open_set"]["false_unknown_rate"]
            ),
        },
        "model_comparison": {
            "logistic_macro_f1": float(logistic["macro_f1"]),
            "random_forest_macro_f1": float(forest["macro_f1"]),
            "logistic_inference_ms": float(logistic["inference_mean_ms"]),
            "random_forest_inference_ms": float(forest["inference_mean_ms"]),
            "logistic_model_size_mb": float(logistic["model_size_mb"]),
            "random_forest_model_size_mb": float(forest["model_size_mb"]),
        },
        "edge_gateway": {
            "platform": edge["platform_label"],
            "mean_latency_ms": float(edge["latency_ms"]["mean"]),
            "p95_latency_ms": float(edge["latency_ms"]["p95"]),
            "throughput_messages_per_second": float(
                edge["throughput_messages_per_second"]
            ),
            "peak_memory_mb": float(edge["peak_memory_mb_after"]),
        },
        "raspberry_pi_gateway": {
            "platform": str(edge_pi_repeated["platform_label"]),
            "runs": int(edge_pi_repeated["runs"]),
            "mean_latency_ms": float(
                edge_pi_repeated["total_latency_mean_ms_mean"]
            ),
            "p95_latency_ms": float(
                edge_pi_repeated["total_latency_p95_ms_mean"]
            ),
            "throughput_messages_per_second": float(
                edge_pi_repeated["throughput_messages_per_second_mean"]
            ),
            "cpu_percent_single_core_equivalent": float(
                edge_pi_repeated[
                    "cpu_percent_single_core_equivalent_mean"
                ]
            ),
            "peak_memory_mb": float(
                edge_pi_repeated["process_peak_memory_after_mb_mean"]
            ),
        },
        "cross_platform": {
            "mac_runs": int(edge_mac_repeated["runs"]),
            "pi_runs": int(edge_pi_repeated["runs"]),
            "logistic_pi_to_mac_latency_ratio": float(
                platform_comparison.loc[
                    platform_comparison["pipeline"] == "logistic_regression",
                    "pi_to_mac_latency_ratio",
                ].iloc[0]
            ),
            "random_forest_pi_to_mac_latency_ratio": float(
                platform_comparison.loc[
                    platform_comparison["pipeline"] == "random_forest",
                    "pi_to_mac_latency_ratio",
                ].iloc[0]
            ),
            "open_set_pi_to_mac_latency_ratio": float(
                platform_comparison.loc[
                    platform_comparison["pipeline"] == "open_set",
                    "pi_to_mac_latency_ratio",
                ].iloc[0]
            ),
        },
        "live_mqtt_deployment": {
            "scenarios": int(len(live_mqtt)),
            "messages": int(live_messages),
            "known_attack_alert_rate": float(
                (
                    known_live["attack_messages"]
                    * known_live["attack_alert_rate"]
                ).sum()
                / known_live["attack_messages"].sum()
            ),
            "known_exact_classification_rate": float(
                (
                    known_live["attack_messages"]
                    * known_live["exact_attack_class_rate"]
                ).sum()
                / known_live["attack_messages"].sum()
            ),
            "unseen_attack": "gradual",
            "unseen_unknown_recall": float(
                gradual_live["unknown_rate_on_attack"]
            ),
            "normal_alert_rate": float(
                live_normal_alerts / live_normal_messages
            ),
            "mean_latency_ms": float(
                (
                    live_mqtt["messages"] * live_mqtt["mean_latency_ms"]
                ).sum()
                / live_messages
            ),
            "maximum_scenario_p95_latency_ms": float(
                live_mqtt["p95_latency_ms"].max()
            ),
            "maximum_latency_ms": float(
                live_mqtt["maximum_latency_ms"].max()
            ),
        },
        "drift": {
            "runs": int(drift["runs"]),
            "measurement_detection_delay_mean": float(
                drift_summary["measurement_detection_delay"]["mean"]
            ),
            "communication_detection_delay_mean": float(
                drift_summary["communication_detection_delay"]["mean"]
            ),
            "adaptation_delay_mean": float(
                drift_summary["adaptation_delay"]["mean"]
            ),
            "final_alert_rate_mean": float(
                drift_summary["final_100_alert_rate"]["mean"]
            ),
            "guarded_poisoning_shift_mean": float(
                drift_summary["guarded_reference_shift"]["mean"]
            ),
            "unguarded_poisoning_shift_mean": float(
                drift_summary["unguarded_reference_shift"]["mean"]
            ),
        },
        "raspberry_pi_live_drift": {
            "runs_per_scenario": int(pi_measurement_drift["runs"]),
            "measurement_detection_delay_messages": float(
                pi_measurement_drift["mean_detection_delay_messages_mean"]
            ),
            "measurement_detection_delay_messages_std": float(
                pi_measurement_drift["mean_detection_delay_messages_std"]
            ),
            "measurement_active_alert_reduction_percent": float(
                pi_measurement_drift["active_alert_reduction_percent_mean"]
            ),
            "measurement_active_alert_reduction_percent_std": float(
                pi_measurement_drift["active_alert_reduction_percent_std"]
            ),
            "measurement_adaptation_updates_mean": float(
                pi_measurement_drift["adaptation_updates_mean"]
            ),
            "measurement_mean_latency_ms": float(
                pi_measurement_drift["latency_mean_ms_mean"]
            ),
            "measurement_mean_latency_ms_std": float(
                pi_measurement_drift["latency_mean_ms_std"]
            ),
            "communication_detection_delay_messages": float(
                pi_communication_drift["mean_detection_delay_messages_mean"]
            ),
            "communication_detection_delay_messages_std": float(
                pi_communication_drift["mean_detection_delay_messages_std"]
            ),
            "communication_active_alert_reduction_percent": float(
                pi_communication_drift["active_alert_reduction_percent_mean"]
            ),
            "communication_active_alert_reduction_percent_std": float(
                pi_communication_drift["active_alert_reduction_percent_std"]
            ),
            "communication_adaptation_updates_mean": float(
                pi_communication_drift["adaptation_updates_mean"]
            ),
            "communication_mean_latency_ms": float(
                pi_communication_drift["latency_mean_ms_mean"]
            ),
            "communication_mean_latency_ms_std": float(
                pi_communication_drift["latency_mean_ms_std"]
            ),
            "measurement_pi_to_mac_latency_ratio": float(
                drift_platform_comparison.loc[
                    drift_platform_comparison["scenario"]
                    == "measurement_drift",
                    "pi_to_mac_latency_ratio",
                ].iloc[0]
            ),
            "communication_pi_to_mac_latency_ratio": float(
                drift_platform_comparison.loc[
                    drift_platform_comparison["scenario"]
                    == "communication_drift",
                    "pi_to_mac_latency_ratio",
                ].iloc[0]
            ),
            "thermal_observations": int(len(pi_drift_thermal)),
            "maximum_observed_temperature_c": float(
                pi_drift_thermal["temperature_c"].max()
            ),
            "thermal_throttling_observed": bool(
                (pi_drift_thermal["throttled_status"] != "0x0").any()
            ),
            "throttled_status_all_checkpoints": "0x0",
        },
        "normal_mqtt_cpu": {
            "message_rate": 6.0,
            "runs_per_pipeline": 5,
            "logistic_mac_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "logistic_regression",
                    "cpu_percent_single_core_equivalent_mean_mac",
                ]
            ),
            "logistic_pi_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "logistic_regression",
                    "cpu_percent_single_core_equivalent_mean_pi",
                ]
            ),
            "random_forest_mac_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "random_forest",
                    "cpu_percent_single_core_equivalent_mean_mac",
                ]
            ),
            "random_forest_pi_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "random_forest",
                    "cpu_percent_single_core_equivalent_mean_pi",
                ]
            ),
            "open_set_mac_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "open_set",
                    "cpu_percent_single_core_equivalent_mean_mac",
                ]
            ),
            "open_set_pi_single_core_percent": float(
                normal_cpu_by_pipeline.loc[
                    "open_set",
                    "cpu_percent_single_core_equivalent_mean_pi",
                ]
            ),
            "deadline_misses_mac": float(
                normal_load_cpu["deadline_misses_mean_mac"].sum()
            ),
            "deadline_misses_pi": float(
                normal_load_cpu["deadline_misses_mean_pi"].sum()
            ),
            "pi_temperature_before_c": float(
                normal_load_thermal.loc[
                    normal_load_thermal["checkpoint"]
                    == "before_normal_load_benchmark",
                    "temperature_c",
                ].iloc[0]
            ),
            "pi_temperature_after_c": float(
                normal_load_thermal.loc[
                    normal_load_thermal["checkpoint"]
                    == "after_normal_load_benchmark",
                    "temperature_c",
                ].iloc[0]
            ),
            "thermal_throttling_observed": bool(
                (normal_load_thermal["throttled_status"] != "0x0").any()
            ),
        },
        "limitations": {
            "edge_hardware": "MacBook and Raspberry Pi 5 Model B",
            "raspberry_pi_measured": True,
            "raspberry_pi_drift_measured": True,
            "energy_measured": False,
            "thermal_throttling_checked": True,
            "normal_load_cpu_measured": True,
        },
    }

    rows: list[dict[str, Any]] = []
    for section, metrics in summary.items():
        for metric, value in metrics.items():
            rows.append({"section": section, "metric": metric, "value": value})
    return summary, pd.DataFrame(rows)


def main() -> None:
    summary, table = build_summary()
    output_directory = Path("results/final")
    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "experiment_summary.json"
    csv_path = output_directory / "experiment_summary.csv"
    live_table_path = output_directory / "live_mqtt_deployment_table.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    table.to_csv(csv_path, index=False)
    pd.read_csv(SOURCES["live_mqtt"]).to_csv(live_table_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved to: {json_path}")
    print(f"Saved to: {csv_path}")
    print(f"Saved to: {live_table_path}")


if __name__ == "__main__":
    main()
