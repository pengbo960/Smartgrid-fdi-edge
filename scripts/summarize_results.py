from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


SOURCES = {
    "ablation": Path("results/v2/ablation/ablation_summary.csv"),
    "open_set": Path("results/open_set/metrics/open_set_metrics.json"),
    "edge": Path("results/edge/macbook_benchmark.json"),
    "model_comparison": Path("results/model_comparison/model_comparison.csv"),
    "drift": Path("results/drift/drift_repeated_summary.json"),
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
    drift = load_json(SOURCES["drift"])

    all_views = ablation.loc[ablation["experiment_name"] == "all_views"].iloc[0]
    logistic = models.loc[models["model_name"] == "logistic_regression"].iloc[0]
    forest = models.loc[models["model_name"] == "random_forest"].iloc[0]
    drift_summary = drift["summary"]

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
        "limitations": {
            "edge_hardware": "MacBook emulated edge gateway",
            "raspberry_pi_measured": False,
            "energy_measured": False,
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
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    table.to_csv(csv_path, index=False)
    print(table.to_string(index=False))
    print(f"\nSaved to: {json_path}")
    print(f"Saved to: {csv_path}")


if __name__ == "__main__":
    main()
