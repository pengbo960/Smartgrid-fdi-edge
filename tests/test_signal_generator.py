import pytest

from src.simulation.signal_generator import (
    DeviceConfig,
    SignalGenerator,
    build_device_config,
    build_device_configs,
)


def test_generator_returns_expected_measurements() -> None:
    device = DeviceConfig(
        device_id="meter_01",
        phase=0.0,
    )

    generator = SignalGenerator(random_seed=42)

    measurement = generator.generate(
        step=0,
        device=device,
    )

    assert set(measurement.keys()) == {
        "voltage",
        "current",
        "power",
        "frequency",
    }

    assert isinstance(measurement["voltage"], float)
    assert isinstance(measurement["current"], float)
    assert isinstance(measurement["power"], float)
    assert isinstance(measurement["frequency"], float)


def test_generator_is_reproducible() -> None:
    device = DeviceConfig(
        device_id="meter_01",
        phase=0.0,
    )

    first_generator = SignalGenerator(random_seed=42)
    second_generator = SignalGenerator(random_seed=42)

    first_result = first_generator.generate(
        step=10,
        device=device,
    )

    second_result = second_generator.generate(
        step=10,
        device=device,
    )

    assert first_result == second_result


def test_different_devices_have_different_patterns() -> None:
    generator = SignalGenerator(random_seed=42)

    meter_01 = DeviceConfig(
        device_id="meter_01",
        phase=0.0,
    )

    meter_02 = DeviceConfig(
        device_id="meter_02",
        phase=0.7,
    )

    first = generator.generate(
        step=20,
        device=meter_01,
    )

    second = generator.generate(
        step=20,
        device=meter_02,
    )

    assert first != second


def test_negative_step_is_rejected() -> None:
    generator = SignalGenerator(random_seed=42)
    device = DeviceConfig(
        device_id="meter_01",
        phase=0.0,
    )

    with pytest.raises(
        ValueError,
        match="step must be zero or greater",
    ):
        generator.generate(
            step=-1,
            device=device,
        )


def test_build_device_config_uses_defaults() -> None:
    device = build_device_config(
        {
            "device_id": "meter_01",
            "phase": 0.0,
        }
    )

    assert device.device_id == "meter_01"
    assert device.phase == 0.0
    assert device.voltage_base == 230.0
    assert device.current_base == 4.5
    assert device.frequency_base == 50.0
    assert device.power_factor == 0.95


def test_duplicate_device_ids_are_rejected() -> None:
    raw_devices = [
        {
            "device_id": "meter_01",
            "phase": 0.0,
        },
        {
            "device_id": "meter_01",
            "phase": 0.7,
        },
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate device_id",
    ):
        build_device_configs(raw_devices)