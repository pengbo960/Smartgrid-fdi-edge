from src.simulation.scenario_manager import (
    AttackSchedule,
    ScenarioConfig,
    ScenarioManager,
)
from src.simulation.signal_generator import (
    DeviceConfig,
    SignalGenerator,
)


def test_signal_generator_and_scenario_manager_integrate() -> None:
    device = DeviceConfig(
        device_id="meter_02",
        phase=0.7,
    )

    generator = SignalGenerator(
        random_seed=42
    )

    scenario = ScenarioConfig(
        scenario_id="constant_integration_test",
        duration=10,
        publish_interval=1.0,
        random_seed=42,
        attacks=(
            AttackSchedule(
                device_id="meter_02",
                attack_type="constant",
                start_step=2,
                end_step=5,
            ),
        ),
    )

    manager = ScenarioManager(scenario)

    before = manager.apply(
        device_id=device.device_id,
        step=1,
        measurements=generator.generate(
            step=1,
            device=device,
        ),
    )

    during = manager.apply(
        device_id=device.device_id,
        step=2,
        measurements=generator.generate(
            step=2,
            device=device,
        ),
    )

    after = manager.apply(
        device_id=device.device_id,
        step=5,
        measurements=generator.generate(
            step=5,
            device=device,
        ),
    )

    assert before.attack_type == "none"

    assert during.attack_type == "constant"
    assert during.measurements["voltage"] == 242.0
    assert during.attack_step == 0

    assert after.attack_type == "none"