from __future__ import annotations

import argparse
import signal
from pathlib import Path
from typing import Any

from src.collection.subscriber import MqttSubscriber
from src.common.config import load_yaml_config
from src.detection.alert_manager import AlertManager
from src.detection.edge_detector import EdgeDetector
from src.detection.model_loader import OpenSetModelBundle
from src.features.feature_pipeline import StreamingFeaturePipeline
from src.drift.monitor import MultiFeatureDriftMonitor
from src.drift.controller import DriftController


subscriber: MqttSubscriber | None = None


def stop_handler(signum: int, frame: Any) -> None:
    if subscriber is not None:
        print("\nStopping edge detector...")
        subscriber.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-time MQTT open-set edge detector."
    )
    parser.add_argument("--config", default="config/edge.yaml")
    parser.add_argument(
        "--broker-host",
        default=None,
        help="Override the MQTT broker host from config/mqtt.yaml.",
    )
    parser.add_argument(
        "--broker-port",
        type=int,
        default=None,
        help="Override the MQTT broker port from config/mqtt.yaml.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--print-normal", action="store_true",
        help="Print normal decisions as well as alerts.",
    )
    return parser.parse_args()


def main() -> None:
    global subscriber
    args = parse_args()
    edge_config = load_yaml_config(args.config)
    mqtt_config = load_yaml_config("config/mqtt.yaml")
    artifacts = edge_config["artifacts"]
    feature_config = edge_config["features"]

    model = OpenSetModelBundle.load(
        classifier_path=artifacts["classifier"],
        scaler_path=artifacts["scaler"],
        anomaly_detector_path=artifacts["anomaly_detector"],
        metadata_path=artifacts["metadata"],
    )
    pipeline = StreamingFeaturePipeline(
        window_size=int(feature_config["window_size"]),
        minimum_history=int(feature_config["minimum_history"]),
        power_factor=float(feature_config["power_factor"]),
        repeated_value_field=str(feature_config["repeated_value_field"]),
        value_tolerance=float(feature_config["value_tolerance"]),
    )
    output = args.output or Path(edge_config["output"]["detections"])
    alerts = AlertManager(
        output_path=output,
        print_normal=(
            args.print_normal or bool(edge_config["output"]["print_normal"])
        ),
    )
    drift_config = edge_config.get("drift", {})
    drift_monitor = (
        MultiFeatureDriftMonitor.from_config(drift_config["features"])
        if drift_config.get("enabled", False)
        else None
    )
    adaptation_config = drift_config.get("adaptation", {})
    drift_controller = (
        DriftController.from_config(drift_monitor, adaptation_config)
        if (
            drift_monitor is not None
            and adaptation_config.get("enabled", False)
        )
        else None
    )

    detector = EdgeDetector(
        model=model,
        feature_pipeline=pipeline,
        result_handler=alerts.emit,
        drift_monitor=(None if drift_controller is not None else drift_monitor),
        drift_controller=drift_controller,
    )

    broker = mqtt_config["broker"]
    broker_host = (
        args.broker_host if args.broker_host is not None else str(broker["host"])
    )
    broker_port = (
        args.broker_port if args.broker_port is not None else int(broker["port"])
    )
    if not broker_host:
        raise ValueError("MQTT broker host must not be empty")
    if broker_port <= 0:
        raise ValueError("MQTT broker port must be greater than zero")
    subscriber = MqttSubscriber(
        host=broker_host,
        port=broker_port,
        client_id=str(edge_config["mqtt"]["client_id"]),
        topic=str(mqtt_config["topic"]["measurement_subscription"]),
        message_handler=detector.process,
        qos=int(mqtt_config["subscribe"]["qos"]),
        keepalive=int(broker["keepalive"]),
    )

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)
    print(f"Platform: {edge_config['platform_label']}")
    print(f"MQTT broker: {broker_host}:{broker_port}")
    print(f"Detection log: {output.resolve()}")
    try:
        subscriber.connect()
        subscriber.run_forever()
    finally:
        subscriber.disconnect()
        alerts.close()
        print(
            f"Stopped: processed={detector.processed_messages}, "
            f"failed={detector.failed_messages}"
        )


if __name__ == "__main__":
    main()
