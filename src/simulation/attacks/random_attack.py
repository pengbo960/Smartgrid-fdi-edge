from __future__ import annotations

import random


from src.simulation.attacks.base import (
    Attack,
    Measurements,
)


class RandomAttack(Attack):
    """
    Add bounded random manipulation to selected measurements.
    """

    attack_name = "random"

    def __init__(
        self,
        random_seed: int = 42,
        voltage_range: float = 12.0,
        current_range: float = 1.5,
        recalculate_power: bool = True,
        power_factor: float = 0.95,
    ) -> None:
        if voltage_range < 0:
            raise ValueError(
                "voltage_range must be zero or greater"
            )

        if current_range < 0:
            raise ValueError(
                "current_range must be zero or greater"
            )

        if not 0 < power_factor <= 1:
            raise ValueError(
                "power_factor must be greater than zero "
                "and no greater than one"
            )

        self.rng = random.Random(random_seed)
        self.voltage_range = voltage_range
        self.current_range = current_range
        self.recalculate_power = recalculate_power
        self.power_factor = power_factor

    def apply(
        self,
        measurements: Measurements,
        attack_step: int,
    ) -> Measurements:
        if attack_step < 0:
            raise ValueError(
                "attack_step must be zero or greater"
            )

        required_fields = {
            "voltage",
            "current",
        }

        missing_fields = (
            required_fields - measurements.keys()
        )

        if missing_fields:
            raise KeyError(
                f"Missing measurement fields: "
                f"{sorted(missing_fields)}"
            )

        attacked = measurements.copy()

        attacked_voltage = (
            float(attacked["voltage"])
            + self.rng.uniform(
                -self.voltage_range,
                self.voltage_range,
            )
        )

        attacked_current = (
            float(attacked["current"])
            + self.rng.uniform(
                -self.current_range,
                self.current_range,
            )
        )

        attacked["voltage"] = round(
            attacked_voltage,
            4,
        )

        attacked["current"] = round(
            attacked_current,
            4,
        )

        if (
            self.recalculate_power
            and "power" in attacked
        ):
            attacked["power"] = round(
                attacked["voltage"]
                * attacked["current"]
                * self.power_factor,
                4,
            )

        return {
            key: round(float(value), 4)
            for key, value in attacked.items()
        }