import pytest

from src.simulation.drift_manager import NormalDriftManager


def measurements() -> dict[str, float]:
    return {
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
    }


def test_measurement_drift_preserves_normal_attack_semantics() -> None:
    manager = NormalDriftManager.from_config(
        [
            {
                "device_id": "meter_02",
                "drift_type": "measurement_shift",
                "start_step": 10,
                "end_step": 20,
                "voltage_offset": 5.0,
                "transition_steps": 1,
            }
        ]
    )
    before, label_before, _ = manager.apply_measurements(
        "meter_02", 9, measurements()
    )
    during, label, drift_step = manager.apply_measurements(
        "meter_02", 10, measurements()
    )
    assert before["voltage"] == 230.0
    assert label_before == "none"
    assert during["voltage"] == 235.0
    assert during["power"] == 1116.25
    assert label == "measurement_shift"
    assert drift_step == 0


def test_measurement_drift_is_device_specific() -> None:
    manager = NormalDriftManager.from_config(
        [
            {
                "device_id": "meter_02",
                "drift_type": "measurement_shift",
                "start_step": 10,
                "end_step": 20,
                "voltage_offset": 5.0,
            }
        ]
    )
    result, label, _ = manager.apply_measurements(
        "meter_01", 10, measurements()
    )
    assert result == measurements()
    assert label == "none"


def test_publish_interval_drift() -> None:
    manager = NormalDriftManager.from_config(
        [
            {
                "drift_type": "publish_interval",
                "start_step": 10,
                "end_step": 20,
                "target_interval": 0.8,
            }
        ]
    )
    assert manager.publish_interval(9, 0.5) == (0.5, "none", None)
    interval, label, drift_step = manager.publish_interval(10, 0.5)
    assert interval == 0.8
    assert label == "publish_interval"
    assert drift_step == 0


def test_invalid_drift_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        NormalDriftManager.from_config(
            [
                {
                    "drift_type": "unknown",
                    "start_step": 1,
                    "end_step": 2,
                }
            ]
        )
