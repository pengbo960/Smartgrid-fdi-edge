import pytest

from src.simulation.scenario_manager import (
    AttackSchedule,
    ScenarioConfig,
    ScenarioManager,
    build_scenario_config,
)


def normal_measurements() -> dict[str, float]:
    return {
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
    }


def build_constant_scenario() -> ScenarioConfig:
    return ScenarioConfig(
        scenario_id="constant_test",
        duration=20,
        publish_interval=1.0,
        random_seed=42,
        attacks=(
            AttackSchedule(
                device_id="meter_02",
                attack_type="constant",
                start_step=5,
                end_step=10,
            ),
        ),
    )


def test_scenario_returns_normal_before_attack() -> None:
    manager = ScenarioManager(
        build_constant_scenario()
    )

    result = manager.apply(
        device_id="meter_02",
        step=4,
        measurements=normal_measurements(),
    )

    assert result.attack_type == "none"
    assert result.is_attack == 0
    assert result.attack_step is None
    assert result.measurements["voltage"] == 230.0


def test_scenario_applies_attack_at_start_step() -> None:
    manager = ScenarioManager(
        build_constant_scenario()
    )

    result = manager.apply(
        device_id="meter_02",
        step=5,
        measurements=normal_measurements(),
    )

    assert result.attack_type == "constant"
    assert result.is_attack == 1
    assert result.attack_step == 0
    assert result.measurements["voltage"] == 242.0


def test_scenario_increments_attack_step() -> None:
    manager = ScenarioManager(
        build_constant_scenario()
    )

    result = manager.apply(
        device_id="meter_02",
        step=8,
        measurements=normal_measurements(),
    )

    assert result.attack_step == 3


def test_scenario_returns_normal_at_end_step() -> None:
    manager = ScenarioManager(
        build_constant_scenario()
    )

    result = manager.apply(
        device_id="meter_02",
        step=10,
        measurements=normal_measurements(),
    )

    assert result.attack_type == "none"
    assert result.is_attack == 0


def test_other_device_remains_normal() -> None:
    manager = ScenarioManager(
        build_constant_scenario()
    )

    result = manager.apply(
        device_id="meter_01",
        step=7,
        measurements=normal_measurements(),
    )

    assert result.attack_type == "none"
    assert result.is_attack == 0


def test_gradual_attack_uses_relative_attack_step() -> None:
    scenario = ScenarioConfig(
        scenario_id="gradual_test",
        duration=30,
        publish_interval=1.0,
        random_seed=42,
        attacks=(
            AttackSchedule(
                device_id="meter_02",
                attack_type="gradual",
                start_step=10,
                end_step=20,
            ),
        ),
    )

    manager = ScenarioManager(scenario)

    first = manager.apply(
        device_id="meter_02",
        step=10,
        measurements=normal_measurements(),
    )

    later = manager.apply(
        device_id="meter_02",
        step=15,
        measurements=normal_measurements(),
    )

    assert first.attack_step == 0
    assert first.measurements["voltage"] == 230.0

    assert later.attack_step == 5
    assert later.measurements["voltage"] == 230.4


def test_overlapping_attacks_are_rejected() -> None:
    scenario = ScenarioConfig(
        scenario_id="overlap_test",
        duration=30,
        publish_interval=1.0,
        random_seed=42,
        attacks=(
            AttackSchedule(
                device_id="meter_02",
                attack_type="constant",
                start_step=5,
                end_step=15,
            ),
            AttackSchedule(
                device_id="meter_02",
                attack_type="random",
                start_step=10,
                end_step=20,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="Overlapping attacks",
    ):
        ScenarioManager(scenario)


def test_build_scenario_config_without_attacks() -> None:
    raw_config = {
        "scenario": {
            "scenario_id": "normal_test",
            "duration": 60,
            "publish_interval": 1.0,
            "random_seed": 42,
        },
        "attacks": [],
    }

    scenario = build_scenario_config(
        raw_config
    )

    assert scenario.scenario_id == "normal_test"
    assert scenario.attacks == ()


def test_invalid_attack_period_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="end_step must be greater",
    ):
        AttackSchedule(
            device_id="meter_02",
            attack_type="constant",
            start_step=10,
            end_step=10,
        )