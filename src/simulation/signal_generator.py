from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceConfig:
    """
    Configuration for one simulated smart meter.

    Attributes:
        device_id:
            Unique identity of the simulated device.

        phase:
            Phase offset used to create slightly different measurement
            patterns for different devices.

        voltage_base:
            Normal central voltage value in volts.

        current_base:
            Normal central current value in amperes.

        frequency_base:
            Normal central frequency value in hertz.

        power_factor:
            Approximate power factor used to calculate active power.
    """

    device_id: str
    phase: float
    voltage_base: float = 230.0
    current_base: float = 4.5
    frequency_base: float = 50.0
    power_factor: float = 0.95


class SignalGenerator:
    """
    Generate reproducible normal smart-meter measurements.

    Each SignalGenerator instance owns an independent random-number
    generator. This avoids changes in one part of the program affecting
    the random values produced elsewhere.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.rng = random.Random(random_seed)

    def generate(
        self,
        step: int,
        device: DeviceConfig,
    ) -> dict[str, float]:
        """
        Generate one normal measurement set for a device.

        Args:
            step:
                Current simulation step. It is used to produce gradual
                periodic changes over time.

            device:
                Configuration of the simulated smart meter.

        Returns:
            A dictionary containing voltage, current, power and frequency.
        """
        if step < 0:
            raise ValueError("step must be zero or greater")

        voltage = (
            device.voltage_base
            + 2.0 * math.sin(step / 25.0 + device.phase)
            + self.rng.gauss(0.0, 0.25)
        )

        current = (
            device.current_base
            + 0.6 * math.sin(step / 18.0 + device.phase)
            + self.rng.gauss(0.0, 0.08)
        )

        power = (
            voltage
            * current
            * device.power_factor
            + self.rng.gauss(0.0, 3.0)
        )

        frequency = (
            device.frequency_base
            + 0.025 * math.sin(step / 35.0 + device.phase)
            + self.rng.gauss(0.0, 0.005)
        )

        return {
            "voltage": round(voltage, 4),
            "current": round(current, 4),
            "power": round(power, 4),
            "frequency": round(frequency, 4),
        }


def build_device_config(raw_config: dict[str, Any]) -> DeviceConfig:
    """
    Convert a device mapping loaded from YAML into DeviceConfig.

    Expected YAML structure:

        device_id: meter_01
        phase: 0.0
        voltage_base: 230.0
        current_base: 4.5
        frequency_base: 50.0
        power_factor: 0.95

    Only device_id and phase are required. Other fields use defaults.
    """
    if not isinstance(raw_config, dict):
        raise TypeError("Device configuration must be a dictionary")

    if "device_id" not in raw_config:
        raise ValueError("Missing required device field: device_id")

    if "phase" not in raw_config:
        raise ValueError("Missing required device field: phase")

    device_id = str(raw_config["device_id"]).strip()

    if not device_id:
        raise ValueError("device_id must not be empty")

    try:
        phase = float(raw_config["phase"])
        voltage_base = float(raw_config.get("voltage_base", 230.0))
        current_base = float(raw_config.get("current_base", 4.5))
        frequency_base = float(
            raw_config.get("frequency_base", 50.0)
        )
        power_factor = float(raw_config.get("power_factor", 0.95))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid numeric configuration for device {device_id}"
        ) from exc

    if voltage_base <= 0:
        raise ValueError("voltage_base must be greater than zero")

    if current_base < 0:
        raise ValueError("current_base must be zero or greater")

    if frequency_base <= 0:
        raise ValueError("frequency_base must be greater than zero")

    if not 0 < power_factor <= 1:
        raise ValueError(
            "power_factor must be greater than zero and no greater than one"
        )

    return DeviceConfig(
        device_id=device_id,
        phase=phase,
        voltage_base=voltage_base,
        current_base=current_base,
        frequency_base=frequency_base,
        power_factor=power_factor,
    )


def build_device_configs(
    raw_devices: list[dict[str, Any]],
) -> list[DeviceConfig]:
    """
    Convert a list of YAML device mappings into DeviceConfig objects.

    Duplicate device IDs are rejected because every MQTT device must have
    a unique identity.
    """
    if not isinstance(raw_devices, list):
        raise TypeError("devices configuration must be a list")

    if not raw_devices:
        raise ValueError("At least one device must be configured")

    devices = [
        build_device_config(raw_device)
        for raw_device in raw_devices
    ]

    device_ids = [device.device_id for device in devices]

    if len(device_ids) != len(set(device_ids)):
        raise ValueError("Duplicate device_id values are not allowed")

    return devices