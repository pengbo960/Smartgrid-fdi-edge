from __future__ import annotations

from dataclasses import dataclass

from src.simulation.attacks.base import (
    Attack,
    Measurements,
)


@dataclass(frozen=True)
class GradualAttack(Attack):
    """
    Slowly increase one measurement by a bounded bias.
    """

    target_field: str = "voltage"
    bias_per_step: float = 0.08
    maximum_bias: float = 10.0
    attack_name: str = "gradual"

    def __post_init__(self) -> None:
        if self.bias_per_step < 0:
            raise ValueError(
                "bias_per_step must be zero or greater"
            )

        if self.maximum_bias < 0:
            raise ValueError(
                "maximum_bias must be zero or greater"
            )

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

        bias = min(
            attack_step * self.bias_per_step,
            self.maximum_bias,
        )

        attacked[self.target_field] = round(
            attacked[self.target_field] + bias,
            4,
        )

        return attacked