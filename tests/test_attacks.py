import pytest

from src.simulation.attacks import create_attack
from src.simulation.attacks.constant import (
    ConstantAttack,
)
from src.simulation.attacks.gradual import (
    GradualAttack,
)
from src.simulation.attacks.random_attack import (
    RandomAttack,
)


def normal_measurements() -> dict[str, float]:
    return {
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
    }


def test_constant_attack_changes_target_field() -> None:
    original = normal_measurements()

    attack = ConstantAttack(
        target_field="voltage",
        fixed_value=242.0,
    )

    attacked = attack.apply(
        measurements=original,
        attack_step=0,
    )

    assert attacked["voltage"] == 242.0
    assert attacked["current"] == 5.0
    assert attacked["frequency"] == 50.0


def test_constant_attack_does_not_modify_original() -> None:
    original = normal_measurements()
    attack = ConstantAttack()

    attack.apply(
        measurements=original,
        attack_step=0,
    )

    assert original["voltage"] == 230.0


def test_random_attack_is_reproducible() -> None:
    original = normal_measurements()

    first_attack = RandomAttack(
        random_seed=42
    )

    second_attack = RandomAttack(
        random_seed=42
    )

    first_result = first_attack.apply(
        measurements=original,
        attack_step=0,
    )

    second_result = second_attack.apply(
        measurements=original,
        attack_step=0,
    )

    assert first_result == second_result


def test_random_attack_changes_measurements() -> None:
    original = normal_measurements()

    attack = RandomAttack(
        random_seed=42
    )

    attacked = attack.apply(
        measurements=original,
        attack_step=0,
    )

    assert attacked["voltage"] != (
        original["voltage"]
    )

    assert attacked["current"] != (
        original["current"]
    )

    expected_power = round(
        attacked["voltage"]
        * attacked["current"]
        * 0.95,
        4,
    )

    assert attacked["power"] == expected_power


def test_gradual_attack_increases_over_time() -> None:
    original = normal_measurements()

    attack = GradualAttack(
        bias_per_step=0.1,
        maximum_bias=10.0,
    )

    step_zero = attack.apply(
        measurements=original,
        attack_step=0,
    )

    step_ten = attack.apply(
        measurements=original,
        attack_step=10,
    )

    assert step_zero["voltage"] == 230.0
    assert step_ten["voltage"] == 231.0


def test_gradual_attack_respects_maximum_bias() -> None:
    original = normal_measurements()

    attack = GradualAttack(
        bias_per_step=0.1,
        maximum_bias=5.0,
    )

    attacked = attack.apply(
        measurements=original,
        attack_step=1000,
    )

    assert attacked["voltage"] == 235.0


def test_negative_attack_step_is_rejected() -> None:
    attack = GradualAttack()

    with pytest.raises(
        ValueError,
        match="attack_step must be zero or greater",
    ):
        attack.apply(
            measurements=normal_measurements(),
            attack_step=-1,
        )


@pytest.mark.parametrize(
    "attack_type",
    [
        "constant",
        "random",
        "gradual",
    ],
)
def test_attack_factory_creates_supported_attacks(
    attack_type: str,
) -> None:
    attack = create_attack(
        attack_type=attack_type,
        random_seed=42,
    )

    assert attack.attack_name == attack_type


def test_attack_factory_rejects_unknown_attack() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported attack type",
    ):
        create_attack("unsupported")