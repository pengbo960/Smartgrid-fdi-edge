from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from src.common.config import load_yaml_config
from src.detection.comparison_model_loader import ComparisonModelBundle
from src.detection.edge_detector import EdgeDetector
from src.evaluation.resource_monitor import ResourceMonitor
from src.evaluation.edge_warmup import partition_device_warmup
from src.features.data_loader import load_raw_dataset
from src.features.feature_pipeline import StreamingFeaturePipeline


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("Latency values must not be empty")
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one binary edge model.")
    parser.add_argument("--config", default="config/repeated_edge_benchmark.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-id", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    if args.model not in config["models"]:
        raise ValueError(f"Unknown benchmark model: {args.model}")
    features = config["features"]
    artifact_path = Path(config["models"][args.model])
    model = ComparisonModelBundle.load(artifact_path, args.model)
    detector = EdgeDetector(
        model=model,
        feature_pipeline=StreamingFeaturePipeline(
            window_size=int(features["window_size"]),
            minimum_history=int(features["minimum_history"]),
            power_factor=float(features["power_factor"]),
            repeated_value_field=str(features["repeated_value_field"]),
            value_tolerance=float(features["value_tolerance"]),
        ),
    )
    rows = load_raw_dataset(config["input"]).to_dict(orient="records")
    warmup_rows, measured_rows, warmup_counts = partition_device_warmup(
        rows, int(config.get("warmup_messages_per_device", 0))
    )
    for row in warmup_rows:
        detector.process(row)

    monitor = ResourceMonitor()
    before = monitor.snapshot()
    total_latencies: list[float] = []
    feature_latencies: list[float] = []
    inference_latencies: list[float] = []
    started = perf_counter()
    for row in measured_rows:
        result = detector.process(row)
        total_latencies.append(float(result["total_detection_ms"]))
        feature_latencies.append(float(result["feature_extraction_ms"]))
        inference_latencies.append(float(result["model_inference_ms"]))
    elapsed = perf_counter() - started
    after = monitor.snapshot()
    measured = len(total_latencies)
    report = {
        "run_id": args.run_id,
        "model_name": args.model,
        "model_class": type(model.model).__name__,
        "benchmark_mode": "known_attack_streaming_pipeline",
        "platform_label": config["platform_label"],
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "input_file": str(config["input"]),
        "artifact_path": str(artifact_path),
        "artifact_size_mb": artifact_path.stat().st_size / (1024 ** 2),
        "feature_count": len(model.feature_columns),
        "threshold": model.threshold,
        "warmup_messages_per_device": int(
            config.get("warmup_messages_per_device", 0)
        ),
        "warmup_messages_total": len(warmup_rows),
        "warmup_device_counts": warmup_counts,
        "measured_messages": measured,
        "failed_messages": detector.failed_messages,
        "elapsed_seconds": elapsed,
        "throughput_messages_per_second": measured / elapsed,
        "total_latency_mean_ms": float(np.mean(total_latencies)),
        "total_latency_median_ms": percentile(total_latencies, 50),
        "total_latency_p95_ms": percentile(total_latencies, 95),
        "total_latency_p99_ms": percentile(total_latencies, 99),
        "total_latency_max_ms": float(np.max(total_latencies)),
        "feature_extraction_mean_ms": float(np.mean(feature_latencies)),
        "feature_extraction_p95_ms": percentile(feature_latencies, 95),
        "model_inference_mean_ms": float(np.mean(inference_latencies)),
        "model_inference_p95_ms": percentile(inference_latencies, 95),
        "cpu_percent_single_core_equivalent": (
            (after.cpu_time_seconds - before.cpu_time_seconds) / elapsed * 100.0
        ),
        "process_peak_memory_before_mb": before.memory_mb,
        "process_peak_memory_after_mb": after.memory_mb,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
