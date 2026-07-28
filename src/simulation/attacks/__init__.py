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
from src.simulation.attacks.replay import (
    ReplayBuffer,
)


VALUE_ATTACKS = {
    "constant",
    "random",
    "gradual",
}

COMMUNICATION_ATTACKS = {
    "replay",
}

SUPPORTED_ATTACKS = (
    VALUE_ATTACKS
    | COMMUNICATION_ATTACKS
)


def create_attack(
    attack_type: str,
    random_seed: int = 42,
) -> Attack:
    """
    Create a value-manipulation attack.

    Communication attacks such as replay are handled by the simulator
    because they operate on complete messages rather than measurements.
    """
    if attack_type == "constant":
        return ConstantAttack()

    if attack_type == "random":
        return RandomAttack(
            random_seed=random_seed
        )

    if attack_type == "gradual":
        return GradualAttack()

    if attack_type in COMMUNICATION_ATTACKS:
        raise ValueError(
            f"{attack_type} is a communication attack "
            "and cannot be created by create_attack"
        )

    raise ValueError(
        f"Unsupported attack type: {attack_type}"
    )


__all__ = [
    "Attack",
    "ConstantAttack",
    "RandomAttack",
    "GradualAttack",
    "ReplayBuffer",
    "VALUE_ATTACKS",
    "COMMUNICATION_ATTACKS",
    "SUPPORTED_ATTACKS",
    "create_attack",
]