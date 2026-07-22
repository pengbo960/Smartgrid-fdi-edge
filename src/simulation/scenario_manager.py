from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.simulation.attacks import Attack, create_attack
from src.simulation.attacks.base import Measurements


@dataclass(frozen=True)
class AttackSchedule:
    """
    Attack configuration for one simulated device.
    """

    device_id: str
    attack_type: str
    start_step: int
    end_step: int
    random_seed: int = 42

    def __post_init__(self) -> None:
        if not self.device_id:
            raise ValueError(
                "device_id must not be empty"
            )

        if self.start_step < 0:
            raise ValueError(
                "start_step must be zero or greater"
            )

        if self.end_step <= self.start_step:
            raise ValueError(
                "end_step must be greater than start_step"
            )


@dataclass(frozen=True)
class ScenarioConfig:
    """
    Complete simulation scenario configuration.
    """

    scenario_id: str
    duration: float
    publish_interval: float
    random_seed: int
    attacks: tuple[AttackSchedule, ...]

    def __post_init__(self) -> None:
        if not self.scenario_id:
            raise ValueError(
                "scenario_id must not be empty"
            )

        if self.duration <= 0:
            raise ValueError(
                "duration must be greater than zero"
            )

        if self.publish_interval <= 0:
            raise ValueError(
                "publish_interval must be greater than zero"
            )


@dataclass
class AttackRuntime:
    """
    Runtime state for one configured attack.
    """

    schedule: AttackSchedule
    attack: Attack


@dataclass(frozen=True)
class ScenarioResult:
    """
    Result after applying the scenario to one measurement.
    """

    measurements: Measurements
    attack_type: str
    is_attack: int
    attack_step: int | None


class ScenarioManager:
    """
    Apply configured attacks according to device and simulation step.
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
    ) -> None:
        self.scenario = scenario

        self._attacks_by_device: dict[
            str,
            list[AttackRuntime],
        ] = {}

        for schedule in scenario.attacks:
            runtime = AttackRuntime(
                schedule=schedule,
                attack=create_attack(
                    attack_type=schedule.attack_type,
                    random_seed=schedule.random_seed,
                ),
            )

            self._attacks_by_device.setdefault(
                schedule.device_id,
                [],
            ).append(runtime)

        for runtimes in self._attacks_by_device.values():
            runtimes.sort(
                key=lambda item: item.schedule.start_step
            )

        self._validate_no_overlapping_attacks()

    def _validate_no_overlapping_attacks(self) -> None:
        for device_id, runtimes in (
            self._attacks_by_device.items()
        ):
            for previous, current in zip(
                runtimes,
                runtimes[1:],
            ):
                if (
                    current.schedule.start_step
                    < previous.schedule.end_step
                ):
                    raise ValueError(
                        f"Overlapping attacks configured "
                        f"for device {device_id}"
                    )

    def apply(
        self,
        device_id: str,
        step: int,
        measurements: Measurements,
    ) -> ScenarioResult:
        """
        Apply the active attack for one device and simulation step.
        """
        if step < 0:
            raise ValueError(
                "step must be zero or greater"
            )

        runtimes = self._attacks_by_device.get(
            device_id,
            [],
        )

        for runtime in runtimes:
            schedule = runtime.schedule

            if (
                schedule.start_step
                <= step
                < schedule.end_step
            ):
                attack_step = (
                    step - schedule.start_step
                )

                attacked_measurements = (
                    runtime.attack.apply(
                        measurements=measurements,
                        attack_step=attack_step,
                    )
                )

                return ScenarioResult(
                    measurements=attacked_measurements,
                    attack_type=schedule.attack_type,
                    is_attack=1,
                    attack_step=attack_step,
                )

        return ScenarioResult(
            measurements=measurements.copy(),
            attack_type="none",
            is_attack=0,
            attack_step=None,
        )


def build_attack_schedule(
    raw_config: dict[str, Any],
    default_seed: int,
) -> AttackSchedule:
    """
    Convert one YAML attack entry into AttackSchedule.
    """
    if not isinstance(raw_config, dict):
        raise TypeError(
            "Attack configuration must be a dictionary"
        )

    required_fields = {
        "device_id",
        "attack_type",
        "start_step",
        "end_step",
    }

    missing_fields = (
        required_fields - raw_config.keys()
    )

    if missing_fields:
        raise ValueError(
            f"Missing attack configuration fields: "
            f"{sorted(missing_fields)}"
        )

    return AttackSchedule(
        device_id=str(
            raw_config["device_id"]
        ).strip(),
        attack_type=str(
            raw_config["attack_type"]
        ).strip(),
        start_step=int(
            raw_config["start_step"]
        ),
        end_step=int(
            raw_config["end_step"]
        ),
        random_seed=int(
            raw_config.get(
                "random_seed",
                default_seed,
            )
        ),
    )


def build_scenario_config(
    raw_config: dict[str, Any],
) -> ScenarioConfig:
    """
    Convert a YAML mapping into ScenarioConfig.
    """
    if not isinstance(raw_config, dict):
        raise TypeError(
            "Scenario configuration must be a dictionary"
        )

    if "scenario" not in raw_config:
        raise ValueError(
            "Missing scenario section"
        )

    scenario_section = raw_config["scenario"]

    if not isinstance(scenario_section, dict):
        raise TypeError(
            "scenario section must be a dictionary"
        )

    required_fields = {
        "scenario_id",
        "duration",
        "publish_interval",
        "random_seed",
    }

    missing_fields = (
        required_fields - scenario_section.keys()
    )

    if missing_fields:
        raise ValueError(
            f"Missing scenario fields: "
            f"{sorted(missing_fields)}"
        )

    default_seed = int(
        scenario_section["random_seed"]
    )

    raw_attacks = raw_config.get(
        "attacks",
        [],
    )

    if not isinstance(raw_attacks, list):
        raise TypeError(
            "attacks section must be a list"
        )

    attacks = tuple(
        build_attack_schedule(
            raw_config=attack,
            default_seed=default_seed + index,
        )
        for index, attack in enumerate(raw_attacks)
    )

    return ScenarioConfig(
        scenario_id=str(
            scenario_section["scenario_id"]
        ).strip(),
        duration=float(
            scenario_section["duration"]
        ),
        publish_interval=float(
            scenario_section["publish_interval"]
        ),
        random_seed=default_seed,
        attacks=attacks,
    )