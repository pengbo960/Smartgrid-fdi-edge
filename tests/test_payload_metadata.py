import pytest

from src.collection.payload_metadata import (
    calculate_operational_payload_size,
)


def build_payload() -> dict[str, object]:
    return {
        "scenario_id": "normal_01",
        "device_id": "meter_02",
        "client_id": "simulator-normal_01",
        "timestamp": "2026-07-28T10:00:00+00:00",
        "sequence_number": 10,
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
        "attack_type": "none",
        "is_attack": 0,
        "attack_step": None,
    }


def test_ground_truth_labels_do_not_change_payload_size() -> None:
    normal = build_payload()
    attacked = build_payload()
    attacked["scenario_id"] = "replay_attack_run"
    attacked["attack_type"] = "replay"
    attacked["is_attack"] = 1
    attacked["attack_step"] = 123

    assert calculate_operational_payload_size(
        normal
    ) == calculate_operational_payload_size(
        attacked
    )


def test_operational_value_can_change_payload_size() -> None:
    original = build_payload()
    changed = build_payload()
    changed["voltage"] = 12345.6789

    assert calculate_operational_payload_size(
        original
    ) != calculate_operational_payload_size(
        changed
    )


def test_missing_operational_field_is_rejected() -> None:
    payload = build_payload()
    del payload["voltage"]

    with pytest.raises(
        ValueError,
        match="missing fields",
    ):
        calculate_operational_payload_size(
            payload
        )
