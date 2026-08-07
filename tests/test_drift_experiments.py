import numpy as np

from src.drift.experiments import (
    evaluate_detector,
    evaluate_poisoning,
    generate_step_drift,
)
from src.drift.page_hinkley import PageHinkley


def test_step_drift_is_reproducible() -> None:
    first = generate_step_drift(10, 10, 0.0, 2.0, 0.1, 42)
    second = generate_step_drift(10, 10, 0.0, 2.0, 0.1, 42)
    assert np.array_equal(first, second)


def test_evaluate_detector_reports_delay() -> None:
    values = np.asarray([0.0] * 50 + [2.0] * 50)
    metrics, detections = evaluate_detector(
        values,
        change_step=50,
        detector=PageHinkley(
            delta=0.01, threshold=2.0, minimum_instances=20
        ),
    )
    assert detections
    assert metrics.first_detection_step is not None
    assert metrics.detection_delay is not None
    assert metrics.detection_delay >= 0
    assert metrics.false_alarms_before_change == 0


def test_guard_blocks_unconfirmed_poisoning() -> None:
    config = {
        "window_size": 10,
        "minimum_samples": 5,
        "blend_rate": 0.5,
        "maximum_update_step": 1.0,
    }
    metrics, _ = evaluate_poisoning(
        baseline_samples=20,
        poisoning_samples=40,
        initial_mean=230.0,
        final_mean=238.0,
        standard_deviation=0.0,
        random_seed=42,
        adapter_config=config,
        trusted_deviation=2.0,
        detector_config={
            "delta": 0.01,
            "threshold": 1.0,
            "minimum_instances": 10,
        },
    )
    assert metrics["guarded_reference_shift"] < 3.0
    assert metrics["unguarded_reference_shift"] > 0.0
    assert (
        metrics["guarded_reference_shift"]
        < metrics["unguarded_reference_shift"]
    )
