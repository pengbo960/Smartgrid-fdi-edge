from __future__ import annotations

from typing import Any


def calculate_measured_throughput(
    measured_messages: int,
    elapsed_seconds: float,
) -> float:
    """Calculate throughput from timed messages only, excluding warm-up."""
    if measured_messages <= 0:
        raise ValueError("measured_messages must be greater than zero")
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be greater than zero")
    return measured_messages / elapsed_seconds


def flatten_open_set_benchmark(
    report: dict[str, Any], run_id: int,
) -> dict[str, Any]:
    """Flatten one benchmark JSON report for CSV aggregation."""
    latency = report["latency_ms"]
    return {
        "run_id": run_id,
        "benchmark_mode": report["benchmark_mode"],
        "classifier_class": report["classifier_class"],
        "anomaly_detector_class": report["anomaly_detector_class"],
        "platform_label": report["platform_label"],
        "system": report["system"],
        "machine": report["machine"],
        "python_version": report["python_version"],
        "input_file": report["input_file"],
        "classifier_path": report["classifier_path"],
        "anomaly_detector_path": report["anomaly_detector_path"],
        "metadata_path": report["metadata_path"],
        "input_messages": report["input_messages"],
        "warmup_messages": report["warmup_messages"],
        "measured_messages": report["measured_messages"],
        "processed_messages": report["processed_messages"],
        "failed_messages": report["failed_messages"],
        "elapsed_seconds": report["elapsed_seconds"],
        "throughput_messages_per_second": report[
            "throughput_messages_per_second"
        ],
        "total_latency_mean_ms": latency["mean"],
        "total_latency_median_ms": latency["median"],
        "total_latency_p95_ms": latency["p95"],
        "total_latency_p99_ms": latency["p99"],
        "total_latency_max_ms": latency["maximum"],
        "feature_extraction_mean_ms": report["feature_extraction_ms_mean"],
        "model_inference_mean_ms": report["model_inference_ms_mean"],
        "cpu_percent_single_core_equivalent": report[
            "process_cpu_percent_single_core_equivalent"
        ],
        "process_peak_memory_before_mb": report["peak_memory_mb_before"],
        "process_peak_memory_after_mb": report["peak_memory_mb_after"],
    }
