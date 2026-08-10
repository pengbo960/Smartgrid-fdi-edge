import pandas as pd
import pytest

from src.drift.live_evaluation import build_live_drift_phase_metrics, evaluate_live_drift


def test_live_evaluation_separates_active_drift_and_recovery() -> None:
    frame = pd.DataFrame(
        {
            "device_id": ["meter_01"] * 6,
            "sequence_number": range(6),
            "true_drift_type": ["none", "none", "shift", "shift", "none", "none"],
            "decision": ["none", "none", "random", "random", "none", "none"],
            "drift_aware_decision": [
                "none", "none", "random", "normal_drift", "normal_drift", "none"
            ],
            "drift_detected": [0, 0, 0, 1, 1, 0],
            "adaptation_updated": [0, 0, 0, 0, 1, 0],
            "total_detection_ms": [1.0] * 6,
        }
    )
    metrics = evaluate_live_drift(frame)
    assert metrics["active_drift_rows"] == 2
    assert metrics["recovery_rows"] == 2
    assert metrics["detection_delay_messages"] == {"meter_01": 1}
    assert metrics["raw_active_alerts"] == 2
    assert metrics["drift_aware_active_alerts"] == 1
    assert metrics["active_alert_reduction_percent"] == 50.0
    assert metrics["recovery_normal_drift_rows"] == 1


def test_live_evaluation_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        evaluate_live_drift(pd.DataFrame({"device_id": ["meter_01"]}))


def test_phase_metrics_separate_detection_and_adaptation() -> None:
    frame = pd.DataFrame({
        "device_id": ["meter_01"] * 8,
        "sequence_number": range(8),
        "true_drift_type": ["none", "none", "shift", "shift", "shift", "shift", "none", "none"],
        "decision": ["none", "none", "random", "random", "random", "random", "none", "none"],
        "drift_aware_decision": [
            "none", "none", "random", "random", "random", "normal_drift", "normal_drift", "none",
        ],
        "drift_detected": [0, 0, 0, 1, 0, 0, 0, 0],
        "adaptation_updated": [0, 0, 0, 0, 0, 1, 0, 0],
        "total_detection_ms": [1.0] * 8,
    })
    metrics = build_live_drift_phase_metrics(frame).set_index("evaluation_phase")
    assert metrics.loc["baseline", "rows"] == 2
    assert metrics.loc["drift_pre_detection", "rows"] == 1
    assert metrics.loc["drift_detected_pre_adaptation", "rows"] == 2
    assert metrics.loc["drift_post_adaptation", "rows"] == 1
    assert metrics.loc["drift_post_adaptation", "drift_aware_false_alert_rate"] == 0.0
    assert metrics.loc["recovery", "rows"] == 2
