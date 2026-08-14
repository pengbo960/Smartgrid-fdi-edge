from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from itertools import cycle, islice
from pathlib import Path
from time import perf_counter

import numpy as np

from src.common.config import load_yaml_config
from src.detection.comparison_model_loader import ComparisonModelBundle
from src.detection.edge_detector import EdgeDetector
from src.detection.model_loader import OpenSetModelBundle
from src.evaluation.edge_warmup import partition_device_warmup
from src.evaluation.fixed_rate_benchmark import (
    calculate_cpu_metrics,
    calculate_target_messages,
    parse_temperature,
    parse_throttled,
)
from src.evaluation.resource_monitor import ResourceMonitor
from src.features.data_loader import load_raw_dataset
from src.features.feature_pipeline import StreamingFeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure edge CPU at a fixed incoming message rate."
    )
    parser.add_argument("--config", default="config/fixed_rate_edge_benchmark.yaml")
    parser.add_argument("--pipeline", required=True)
    parser.add_argument("--message-rate", required=True, type=float)
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_pi_status() -> dict[str, object | None]:
    if shutil.which("vcgencmd") is None:
        return {"temperature_c": None, "throttled": None}
    temperature = subprocess.run(
        ["vcgencmd", "measure_temp"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    throttled = subprocess.run(
        ["vcgencmd", "get_throttled"],
        check=False,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "temperature_c": parse_temperature(temperature),
        "throttled": parse_throttled(throttled),
    }


def load_model(config: dict[str, object], pipeline: str):
    pipelines = config["pipelines"]
    if not isinstance(pipelines, dict) or pipeline not in pipelines:
        raise ValueError(f"Unknown fixed-rate pipeline: {pipeline}")
    specification = pipelines[pipeline]
    if not isinstance(specification, dict):
        raise ValueError(f"Pipeline configuration for {pipeline} must be a mapping")
    mode = specification["mode"]
    if mode == "comparison":
        return ComparisonModelBundle.load(specification["artifact"], pipeline)
    if mode == "open_set":
        artifacts = specification["artifacts"]
        return OpenSetModelBundle.load(
            artifacts["classifier"],
            artifacts["scaler"],
            artifacts["anomaly_detector"],
            artifacts["metadata"],
        )
    raise ValueError(f"Unsupported pipeline mode: {mode}")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), quantile))


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    duration = float(config["duration_seconds"])
    target_messages = calculate_target_messages(args.message_rate, duration)
    features = config["features"]
    detector = EdgeDetector(
        model=load_model(config, args.pipeline),
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
    if not measured_rows:
        raise ValueError("Fixed-rate benchmark has no measured input rows")
    for row in warmup_rows:
        detector.process(row)

    scheduled_rows = list(islice(cycle(measured_rows), target_messages))
    monitor = ResourceMonitor()
    hardware_before = read_pi_status()
    before = monitor.snapshot()
    total_latencies: list[float] = []
    inference_latencies: list[float] = []
    deadline_misses = 0
    interval = 1.0 / args.message_rate
    started = perf_counter()
    for index, row in enumerate(scheduled_rows):
        deadline = started + index * interval
        remaining = deadline - perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        result = detector.process(row)
        total_latencies.append(float(result["total_detection_ms"]))
        inference_latencies.append(float(result["model_inference_ms"]))
        if perf_counter() > deadline + interval:
            deadline_misses += 1
    scheduled_end = started + duration
    remaining = scheduled_end - perf_counter()
    if remaining > 0.0:
        time.sleep(remaining)
    elapsed = perf_counter() - started
    after = monitor.snapshot()
    hardware_after = read_pi_status()
    cpu_metrics = calculate_cpu_metrics(
        cpu_seconds=after.cpu_time_seconds - before.cpu_time_seconds,
        elapsed_seconds=elapsed,
        measured_messages=target_messages,
        logical_cpu_count=os.cpu_count() or 1,
    )
    report = {
        "run_id": args.run_id,
        "pipeline": args.pipeline,
        "pipeline_mode": config["pipelines"][args.pipeline]["mode"],
        "benchmark_mode": "fixed_incoming_message_rate",
        "platform_label": config["platform_label"],
        "system": platform.system(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "logical_cpu_count": os.cpu_count() or 1,
        "input_file": str(config["input"]),
        "configured_message_rate": args.message_rate,
        "duration_seconds": duration,
        "target_messages": target_messages,
        "measured_messages": len(total_latencies),
        "achieved_message_rate": len(total_latencies) / elapsed,
        "warmup_messages_total": len(warmup_rows),
        "warmup_device_counts": warmup_counts,
        "failed_messages": detector.failed_messages,
        "deadline_misses": deadline_misses,
        "deadline_miss_rate": deadline_misses / target_messages,
        "elapsed_seconds": elapsed,
        "total_latency_mean_ms": float(np.mean(total_latencies)),
        "total_latency_p95_ms": percentile(total_latencies, 95),
        "model_inference_mean_ms": float(np.mean(inference_latencies)),
        "process_peak_memory_before_mb": before.memory_mb,
        "process_peak_memory_after_mb": after.memory_mb,
        "temperature_before_c": hardware_before["temperature_c"],
        "temperature_after_c": hardware_after["temperature_c"],
        "throttled_before": hardware_before["throttled"],
        "throttled_after": hardware_after["throttled"],
        **cpu_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)


if __name__ == "__main__":
    main()
