from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

import paho.mqtt.client as mqtt


MessageHandler = Callable[[dict[str, Any]], None]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class MqttSubscriber:
    """
    Subscribe to MQTT smart-meter messages and convert them to dataset rows.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: str,
        topic: str,
        message_handler: MessageHandler,
        qos: int = 0,
        keepalive: int = 60,
    ) -> None:
        if not host:
            raise ValueError(
                "MQTT host must not be empty"
            )

        if port <= 0:
            raise ValueError(
                "MQTT port must be greater than zero"
            )

        if not client_id:
            raise ValueError(
                "MQTT client_id must not be empty"
            )

        if not topic:
            raise ValueError(
                "MQTT subscription topic must not be empty"
            )

        self.host = host
        self.port = port
        self.client_id = client_id
        self.topic = topic
        self.message_handler = message_handler
        self.qos = qos
        self.keepalive = keepalive
        self.connected = False

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        if reason_code != 0:
            self.connected = False
            print(
                f"MQTT subscriber connection failed: "
                f"reason_code={reason_code}"
            )
            return

        self.connected = True

        result, message_id = client.subscribe(
            self.topic,
            qos=self.qos,
        )

        if result != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"Subscription failed for {self.topic}: "
                f"rc={result}"
            )

        print(
            f"Connected to MQTT broker "
            f"{self.host}:{self.port} "
            f"as {self.client_id}"
        )

        print(
            f"Subscribed to {self.topic} "
            f"with qos={self.qos}"
        )

    def _on_disconnect(
        self,
        client: mqtt.Client,
        userdata: Any,
        disconnect_flags: mqtt.DisconnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        self.connected = False

        if reason_code != 0:
            print(
                f"Unexpected subscriber disconnection: "
                f"reason_code={reason_code}"
            )

    def _on_message(
        self,
        client: mqtt.Client,
        userdata: Any,
        message: mqtt.MQTTMessage,
    ) -> None:
        receive_timestamp = utc_timestamp()

        try:
            payload_text = message.payload.decode(
                "utf-8"
            )

            payload = json.loads(payload_text)

            if not isinstance(payload, dict):
                raise ValueError(
                    "MQTT payload must be a JSON object"
                )

            row = {
                "receive_timestamp": receive_timestamp,
                "message_timestamp": payload.get(
                    "timestamp",
                    "",
                ),
                "scenario_id": payload.get(
                    "scenario_id",
                    "",
                ),
                "device_id": payload.get(
                    "device_id",
                    "",
                ),
                "client_id": payload.get(
                    "client_id",
                    "",
                ),
                "topic": message.topic,
                "qos": message.qos,
                "retain": int(message.retain),
                "payload_size": len(message.payload),
                "sequence_number": payload.get(
                    "sequence_number",
                    "",
                ),
                "voltage": payload.get(
                    "voltage",
                    "",
                ),
                "current": payload.get(
                    "current",
                    "",
                ),
                "power": payload.get(
                    "power",
                    "",
                ),
                "frequency": payload.get(
                    "frequency",
                    "",
                ),
                "attack_type": payload.get(
                    "attack_type",
                    "",
                ),
                "is_attack": payload.get(
                    "is_attack",
                    "",
                ),
            }

            self.message_handler(row)

        except UnicodeDecodeError as exc:
            print(
                f"Could not decode MQTT payload on "
                f"{message.topic}: {exc}"
            )

        except json.JSONDecodeError as exc:
            print(
                f"Invalid JSON payload on "
                f"{message.topic}: {exc}"
            )

        except (TypeError, ValueError) as exc:
            print(
                f"Invalid MQTT message on "
                f"{message.topic}: {exc}"
            )

    def connect(self) -> None:
        self.client.connect(
            host=self.host,
            port=self.port,
            keepalive=self.keepalive,
        )

    def run_forever(self) -> None:
        self.client.loop_forever()

    def disconnect(self) -> None:
        self.client.disconnect()
        self.connected = False