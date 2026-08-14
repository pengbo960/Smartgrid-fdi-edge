import pytest

from src.evaluation.fixed_rate_benchmark import (
    calculate_cpu_metrics,
    calculate_target_messages,
    parse_temperature,
    parse_throttled,
    validate_message_rates,
)


def test_validate_message_rates() -> None:
    assert validate_message_rates([3, 10.0, 25]) == (3.0, 10.0, 25.0)
    with pytest.raises(ValueError, match="At least one"):
        validate_message_rates([])
    with pytest.raises(ValueError, match="unique"):
        validate_message_rates([3, 3])
    with pytest.raises(ValueError, match="greater than zero"):
        validate_message_rates([0])


def test_calculate_target_messages() -> None:
    assert calculate_target_messages(3.0, 20.0) == 60
    assert calculate_target_messages(0.1, 1.0) == 1
    with pytest.raises(ValueError, match="duration_seconds"):
        calculate_target_messages(3.0, 0.0)


def test_calculate_cpu_metrics() -> None:
    metrics = calculate_cpu_metrics(
        cpu_seconds=2.0,
        elapsed_seconds=10.0,
        measured_messages=100,
        logical_cpu_count=4,
    )
    assert metrics["cpu_percent_single_core_equivalent"] == pytest.approx(20.0)
    assert metrics["cpu_percent_total_machine_capacity"] == pytest.approx(5.0)
    assert metrics["cpu_time_per_message_ms"] == pytest.approx(20.0)


def test_parse_raspberry_pi_status() -> None:
    assert parse_temperature("temp=56.5'C\n") == 56.5
    assert parse_temperature("unavailable") is None
    assert parse_throttled("throttled=0x0\n") == "0x0"
    assert parse_throttled("throttled=0x50005\n") == "0x50005"
    assert parse_throttled("unavailable") is None
