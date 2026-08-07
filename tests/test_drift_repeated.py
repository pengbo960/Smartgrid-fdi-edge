from src.common.config import load_yaml_config
from scripts.evaluate_drift_repeated import run_once


def test_repeated_drift_run_returns_expected_metrics() -> None:
    config = load_yaml_config("config/drift.yaml")
    result = run_once(config, 42)
    assert result["measurement_detection_delay"] >= 0
    assert result["communication_detection_delay"] >= 0
    assert result["measurement_false_alarms"] == 0
    assert result["communication_false_alarms"] == 0
    assert result["guarded_reference_shift"] < result["unguarded_reference_shift"]
