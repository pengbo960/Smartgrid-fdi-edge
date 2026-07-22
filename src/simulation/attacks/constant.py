from __future__ import annotations

from dataclasses import dataclass

from src.simulation.attacks.base import (
    Attack,
    Measurements,
)


@dataclass(frozen=True)
class ConstantAttack(Attack):
    """
    Replace one selected measurement with a fixed value.
    """

    target_field: str = "voltage"
    fixed_value: float = 242.0
    attack_name: str = "constant"

    def apply(
        self,
        measurements: Measurements,
        attack_step: int,
    ) -> Measurements:
        if attack_step < 0:
            raise ValueError(
                "attack_step must be zero or greater"
            )

        if self.target_field not in measurements:
            raise KeyError(
                f"Measurement field not found: "
                f"{self.target_field}"
            )

        attacked = measurements.copy()
        attacked[self.target_field] = round(
            float(self.fixed_value),
            4,
        )

        return attacked