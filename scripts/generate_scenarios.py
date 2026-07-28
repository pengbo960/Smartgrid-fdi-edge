from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


ATTACK_SETTINGS: dict[str, dict[str, Any]] = {
    "normal": {},
    "constant": {
        "attack_type": "constant",
    },
    "random": {
        "attack_type": "random",
    },
    "gradual": {
        "attack_type": "gradual",
    },
    "replay": {
        "attack_type": "replay",
        "replay_lag_steps": 10,
    },
    "topic_spoof": {
        "attack_type": "topic_spoof",
        "spoofed_device_id": "meter_01",
    },
}

SEED_BASES = {
    "normal": 100,
    "constant": 200,
    "random": 300,
    "gradual": 400,
    "replay": 500,
    "topic_spoof": 600,
}


def build_scenario(
    scenario_type: str,
    run_number: int,
    duration: float,
    interval: float,
) -> dict[str, Any]:
    if scenario_type not in ATTACK_SETTINGS:
        raise ValueError(
            f"Unsupported scenario type: {scenario_type}"
        )

    if run_number <= 0:
        raise ValueError(
            "run_number must be greater than zero"
        )

    if duration <= 0:
        raise ValueError(
            "duration must be greater than zero"
        )

    if interval <= 0:
        raise ValueError(
            "interval must be greater than zero"
        )

    scenario_id = (
        f"{scenario_type}_run_{run_number:02d}"
    )

    random_seed = (
        SEED_BASES[scenario_type]
        + run_number
    )

    total_steps = int(
        duration / interval
    )

    attack_start = int(
        total_steps * 0.25
    )

    attack_end = int(
        total_steps * 0.75
    )

    if attack_end <= attack_start:
        raise ValueError(
            "Scenario duration and interval do not "
            "provide a valid attack window"
        )

    config: dict[str, Any] = {
        "scenario": {
            "scenario_id": scenario_id,
            "duration": duration,
            "publish_interval": interval,
            "random_seed": random_seed,
        },
        "attacks": [],
    }

    if scenario_type != "normal":
        attack_config: dict[str, Any] = {
            "device_id": "meter_02",
            "start_step": attack_start,
            "end_step": attack_end,
        }

        attack_config.update(
            ATTACK_SETTINGS[scenario_type]
        )

        if scenario_type == "random":
            attack_config["random_seed"] = (
                1000 + random_seed
            )

        if scenario_type == "replay":
            replay_lag_steps = int(
                attack_config[
                    "replay_lag_steps"
                ]
            )

            if attack_start < replay_lag_steps:
                raise ValueError(
                    "Replay attack window starts before "
                    "enough history can be captured"
                )

        config["attacks"].append(
            attack_config
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible dataset "
            "scenario YAML files."
        )
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=300,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "config/scenarios/dataset"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    if args.runs <= 0:
        raise ValueError(
            "runs must be greater than zero"
        )

    if args.duration <= 0:
        raise ValueError(
            "duration must be greater than zero"
        )

    if args.interval <= 0:
        raise ValueError(
            "interval must be greater than zero"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_count = 0

    for scenario_type in ATTACK_SETTINGS:
        for run_number in range(
            1,
            args.runs + 1,
        ):
            config = build_scenario(
                scenario_type=scenario_type,
                run_number=run_number,
                duration=args.duration,
                interval=args.interval,
            )

            scenario_id = config[
                "scenario"
            ]["scenario_id"]

            output_path = (
                args.output_dir
                / f"{scenario_id}.yaml"
            )

            with output_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                yaml.safe_dump(
                    config,
                    file,
                    sort_keys=False,
                )

            print(
                f"Created {output_path}"
            )

            generated_count += 1

    print(
        f"\nGenerated {generated_count} "
        f"scenario files."
    )


if __name__ == "__main__":
    main()
