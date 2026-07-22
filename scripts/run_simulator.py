from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.common.config import load_yaml_config
from src.simulation.publisher import MqttPublisher
from src.simulation.scenario_manager import (
    ScenarioManager,
    build_scenario_config,
)
from src.simulation.signal_generator import (
    SignalGenerator,
    build_device_configs,
)


running = True


def stop_handler(signum: int, frame: Any) -> None:
    global running
    running = False


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Publish normal or attacked smart-meter data through MQTT."
        )
    )

    parser.add_argument(
        "--scenario",
        type=Path,
        required=True,
        help="Path to the scenario YAML file.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override scenario duration in seconds.",
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Override seconds between measurement cycles.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the normal-signal random seed.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mqtt_config = load_yaml_config("config/mqtt.yaml")
    simulation_config = load_yaml_config(
        "config/simulation.yaml"
    )
    raw_scenario_config = load_yaml_config(
        str(args.scenario)
    )

    broker_config = mqtt_config["broker"]
    topic_config = mqtt_config["topic"]
    publish_config = mqtt_config["publish"]

    devices = build_device_configs(
        simulation_config["devices"]
    )

    scenario = build_scenario_config(
        raw_scenario_config
    )

    duration = (
        args.duration
        if args.duration is not None
        else scenario.duration
    )

    interval = (
        args.interval
        if args.interval is not None
        else scenario.publish_interval
    )

    random_seed = (
        args.seed
        if args.seed is not None
        else scenario.random_seed
    )

    if duration <= 0:
        raise ValueError(
            "duration must be greater than zero"
        )

    if interval <= 0:
        raise ValueError(
            "interval must be greater than zero"
        )

    scenario_manager = ScenarioManager(
        scenario=scenario
    )

    generator = SignalGenerator(
        random_seed=random_seed
    )

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )
    signal.signal(
        signal.SIGTERM,
        stop_handler,
    )

    client_id = (
        f"simulator-{scenario.scenario_id}"
    )

    publisher = MqttPublisher(
        host=str(broker_config["host"]),
        port=int(broker_config["port"]),
        client_id=client_id,
        keepalive=int(
            broker_config["keepalive"]
        ),
    )

    topic_template = str(
        topic_config["measurement_template"]
    )

    qos = int(
        publish_config["qos"]
    )
    retain = bool(
        publish_config["retain"]
    )

    try:
        publisher.connect()
    except OSError as exc:
        print(
            f"Could not connect to MQTT broker: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    start_time = time.monotonic()
    step = 0

    print(
        f"Running scenario={scenario.scenario_id}, "
        f"devices={len(devices)}, "
        f"duration={duration}s, "
        f"interval={interval}s"
    )

    try:
        while (
            running
            and time.monotonic() - start_time < duration
        ):
            cycle_start = time.monotonic()

            for device in devices:
                normal_values = generator.generate(
                    step=step,
                    device=device,
                )

                scenario_result = (
                    scenario_manager.apply(
                        device_id=device.device_id,
                        step=step,
                        measurements=normal_values,
                    )
                )

                values = (
                    scenario_result.measurements
                )

                topic = topic_template.format(
                    device_id=device.device_id
                )

                payload = {
                    "scenario_id": (
                        scenario.scenario_id
                    ),
                    "device_id": (
                        device.device_id
                    ),
                    "client_id": client_id,
                    "timestamp": utc_timestamp(),
                    "sequence_number": step,
                    "voltage": values["voltage"],
                    "current": values["current"],
                    "power": values["power"],
                    "frequency": values["frequency"],
                    "attack_type": (
                        scenario_result.attack_type
                    ),
                    "is_attack": (
                        scenario_result.is_attack
                    ),
                    "attack_step": (
                        scenario_result.attack_step
                    ),
                }

                publisher.publish(
                    topic=topic,
                    payload=payload,
                    qos=qos,
                    retain=retain,
                )

                print(
                    f"Published "
                    f"device={device.device_id}, "
                    f"sequence={step}, "
                    f"attack={scenario_result.attack_type}, "
                    f"topic={topic}"
                )

            step += 1

            elapsed = (
                time.monotonic() - cycle_start
            )

            sleep_time = max(
                0.0,
                interval - elapsed,
            )

            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        publisher.disconnect()
        print("Simulator stopped cleanly.")


if __name__ == "__main__":
    main()