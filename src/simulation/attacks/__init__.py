from __future__ import annotations

from src.simulation.attacks.base import Attack
from src.simulation.attacks.constant import (
    ConstantAttack,
)
from src.simulation.attacks.gradual import (
    GradualAttack,
)
from src.simulation.attacks.random_attack import (
    RandomAttack,
)


SUPPORTED_ATTACKS = {
    "constant",
    "random",
    "gradual",
}


def create_attack(
    attack_type: str,
    random_seed: int = 42,
) -> Attack:
    """
    Create an attack implementation from its configuration name.
    """
    if attack_type == "constant":
        return ConstantAttack()

    if attack_type == "random":
        return RandomAttack(
            random_seed=random_seed
        )

    if attack_type == "gradual":
        return GradualAttack()

    raise ValueError(
        f"Unsupported attack type: {attack_type}"
    )


__all__ = [
    "Attack",
    "ConstantAttack",
    "RandomAttack",
    "GradualAttack",
    "SUPPORTED_ATTACKS",
    "create_attack",
]