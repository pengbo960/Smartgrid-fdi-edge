from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from src.common.config import load_yaml_config
from src.detection.edge_detector import EdgeDetector
from src.detection.model_loader import OpenSetModelBundle
from src.evaluation.resource_monitor import ResourceMonitor
from src.evaluation.edge_warmup import partition_device_warmup
from src.evaluation.open_set_edge_benchmark import calculate_measured_throughput
from src.features.data_loader import load_raw_dataset
from src.features.feature_pipeline import StreamingFeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark online edge inference by replaying a raw CSV."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--warmup-per-device", type=int, default=None)
    return parser.parse_args()


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), value))


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    artifacts = config["artifacts"]
    features = config["features"]
    model = OpenSetModelBundle.load(
        artifacts["classifier"], artifacts["scaler"],
        artifacts["anomaly_detector"], artifacts["metadata"],
    )
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
    dataframe = load_raw_dataset(args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be greater than zero")
        dataframe = dataframe.head(args.limit)

    rows = dataframe.to_dict(orient="records")
    if args.warmup_per_device is not None:
        warmup_rows, measured_rows, warmup_counts = partition_device_warmup(
            rows, args.warmup_per_device
        )
    else:
        if args.warmup < 0 or args.warmup >= len(rows):
            raise ValueError("warmup must leave at least one measured message")
        warmup_rows = rows[:args.warmup]
        measured_rows = rows[args.warmup:]
        warmup_counts = {}
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
        total_latencies.append(result["total_detection_ms"])
        feature_latencies.append(result["feature_extraction_ms"])
        inference_latencies.append(result["model_inference_ms"])
    elapsed = perf_counter() - started
    after = monitor.snapshot()
    cpu_percent = (
        (after.cpu_time_seconds - before.cpu_time_seconds)
        / elapsed
        * 100.0
    )

    report = {
        "platform_label": config["platform_label"],
        "benchmark_mode": "full_open_set_streaming_pipeline",
        "classifier_class": type(model.classifier).__name__,
        "anomaly_detector_class": type(model.anomaly_detector).__name__,
        "classifier_path": str(artifacts["classifier"]),
        "anomaly_detector_path": str(artifacts["anomaly_detector"]),
        "metadata_path": str(artifacts["metadata"]),
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "input_file": str(args.input),
        "input_messages": len(rows),
        "warmup_messages": len(warmup_rows),
        "warmup_messages_per_device": args.warmup_per_device,
        "warmup_device_counts": warmup_counts,
        "measured_messages": len(total_latencies),
        "processed_messages": detector.processed_messages,
        "failed_messages": detector.failed_messages,
        "elapsed_seconds": elapsed,
        "throughput_messages_per_second": calculate_measured_throughput(
            measured_messages=len(total_latencies),
            elapsed_seconds=elapsed,
        ),
        "latency_ms": {
            "mean": float(np.mean(total_latencies)),
            "median": percentile(total_latencies, 50),
            "p95": percentile(total_latencies, 95),
            "p99": percentile(total_latencies, 99),
            "maximum": float(np.max(total_latencies)),
        },
        "feature_extraction_ms_mean": float(np.mean(feature_latencies)),
        "model_inference_ms_mean": float(np.mean(inference_latencies)),
        "process_cpu_percent_single_core_equivalent": cpu_percent,
        "peak_memory_mb_before": before.memory_mb,
        "peak_memory_mb_after": after.memory_mb,
    }
    output = args.output or Path(config["output"]["benchmark"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nBenchmark saved to: {output}")


if __name__ == "__main__":
    main()
