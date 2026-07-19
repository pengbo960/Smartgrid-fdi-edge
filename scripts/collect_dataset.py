from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path
from typing import Any

from src.collection.dataset_writer import (
    CsvDatasetWriter,
)
from src.collection.subscriber import (
    MqttSubscriber,
)
from src.common.config import load_yaml_config


subscriber: MqttSubscriber | None = None


def stop_handler(
    signum: int,
    frame: Any,
) -> None:
    if subscriber is not None:
        print(
            "\nStopping dataset collector..."
        )
        subscriber.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Subscribe to smart-meter MQTT messages "
            "and save them to CSV."
        )
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path.",
    )

    parser.add_argument(
        "--client-id",
        default="dataset-collector",
        help="MQTT subscriber client ID.",
    )

    return parser.parse_args()


def main() -> None:
    global subscriber

    args = parse_args()

    mqtt_config = load_yaml_config(
        "config/mqtt.yaml"
    )

    broker_config = mqtt_config["broker"]
    topic_config = mqtt_config["topic"]
    subscribe_config = mqtt_config["subscribe"]

    signal.signal(
        signal.SIGINT,
        stop_handler,
    )

    signal.signal(
        signal.SIGTERM,
        stop_handler,
    )

    writer = CsvDatasetWriter(
        output_path=args.output
    )

    writer.open()

    subscriber = MqttSubscriber(
        host=str(broker_config["host"]),
        port=int(broker_config["port"]),
        client_id=args.client_id,
        topic=str(
            topic_config[
                "measurement_subscription"
            ]
        ),
        message_handler=writer.write,
        qos=int(subscribe_config["qos"]),
        keepalive=int(
            broker_config["keepalive"]
        ),
    )

    print(
        f"Saving MQTT messages to "
        f"{args.output.resolve()}"
    )

    try:
        subscriber.connect()
        subscriber.run_forever()

    except OSError as exc:
        print(
            f"Could not connect to MQTT broker: "
            f"{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nDataset collector interrupted."
        )

    finally:
        if subscriber is not None:
            subscriber.disconnect()

        writer.close()

        print(
            "Dataset collector stopped cleanly."
        )


if __name__ == "__main__":
    main()