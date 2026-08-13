import pytest

from src.evaluation.open_set_edge_benchmark import (
    calculate_measured_throughput,
    flatten_open_set_benchmark,
)


def test_flatten_open_set_benchmark() -> None:
    report = {
        "benchmark_mode": "full_open_set_streaming_pipeline",
        "classifier_class": "LogisticRegression",
        "anomaly_detector_class": "IsolationForest",
        "platform_label": "macbook",
        "system": "Darwin",
        "machine": "arm64",
        "python_version": "3.11",
        "input_file": "input.csv",
        "classifier_path": "classifier.joblib",
        "anomaly_detector_path": "anomaly.joblib",
        "metadata_path": "metadata.json",
        "input_messages": 100,
        "warmup_messages": 10,
        "measured_messages": 90,
        "processed_messages": 100,
        "failed_messages": 0,
        "elapsed_seconds": 1.0,
        "throughput_messages_per_second": 90.0,
        "latency_ms": {
            "mean": 1.0, "median": 0.9, "p95": 1.2,
            "p99": 1.4, "maximum": 2.0,
        },
        "feature_extraction_ms_mean": 0.2,
        "model_inference_ms_mean": 0.8,
        "process_cpu_percent_single_core_equivalent": 95.0,
        "peak_memory_mb_before": 100.0,
        "peak_memory_mb_after": 110.0,
    }
    row = flatten_open_set_benchmark(report, 3)
    assert row["run_id"] == 3
    assert row["total_latency_p95_ms"] == 1.2
    assert row["classifier_class"] == "LogisticRegression"
    assert row["anomaly_detector_class"] == "IsolationForest"


def test_throughput_excludes_warmup_messages() -> None:
    # 102 warm-up messages are deliberately absent from the numerator.
    assert calculate_measured_throughput(1674, 10.0) == pytest.approx(167.4)


def test_throughput_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="measured_messages"):
        calculate_measured_throughput(0, 1.0)
    with pytest.raises(ValueError, match="elapsed_seconds"):
        calculate_measured_throughput(1, 0.0)
