from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


def load_scenario_id(
    scenario_path: Path,
) -> str:
    with scenario_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid scenario config: "
            f"{scenario_path}"
        )

    scenario = config.get("scenario")

    if not isinstance(scenario, dict):
        raise ValueError(
            f"Missing scenario section: "
            f"{scenario_path}"
        )

    scenario_id = str(
        scenario.get("scenario_id", "")
    ).strip()

    if not scenario_id:
        raise ValueError(
            f"Missing scenario_id: "
            f"{scenario_path}"
        )

    return scenario_id


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one MQTT scenario and "
            "collect its CSV automatically."
        )
    )

    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/raw/training_runs"
        ),
    )

    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help=(
            "Seconds to wait after starting "
            "the collector."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def stop_process(
    process: subprocess.Popen[Any] | None,
) -> None:
    if process is None:
        return

    if process.poll() is not None:
        return

    process.send_signal(
        signal.SIGINT
    )

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=5)


def main() -> None:
    args = parse_arguments()

    if not args.scenario.exists():
        raise FileNotFoundError(
            f"Scenario not found: "
            f"{args.scenario}"
        )

    scenario_id = load_scenario_id(
        args.scenario
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        args.output_dir
        / f"{scenario_id}.csv"
    )

    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output already exists: "
                f"{output_path}. "
                f"Use --overwrite to replace it."
            )

        output_path.unlink()

    collector_command = [
        sys.executable,
        "scripts/collect_dataset.py",
        "--output",
        str(output_path),
        "--client-id",
        f"collector-{scenario_id}",
    ]

    simulator_command = [
        sys.executable,
        "scripts/run_simulator.py",
        "--scenario",
        str(args.scenario),
    ]

    collector: subprocess.Popen[Any] | None = None
    simulator: subprocess.Popen[Any] | None = None

    print(
        f"Starting collection for "
        f"{scenario_id}"
    )

    try:
        collector = subprocess.Popen(
            collector_command
        )

        time.sleep(
            args.startup_delay
        )

        simulator = subprocess.Popen(
            simulator_command
        )

        simulator_return_code = (
            simulator.wait()
        )

        if simulator_return_code != 0:
            raise RuntimeError(
                "Simulator exited with code "
                f"{simulator_return_code}"
            )

    except KeyboardInterrupt:
        print(
            "\nCollection interrupted."
        )

    finally:
        stop_process(simulator)
        stop_process(collector)

    if not output_path.exists():
        raise RuntimeError(
            f"Output was not created: "
            f"{output_path}"
        )

    print(
        f"Saved dataset to {output_path}"
    )


if __name__ == "__main__":
    main()