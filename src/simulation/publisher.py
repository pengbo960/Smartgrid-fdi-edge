from __future__ import annotations

import json
from typing import Any

import paho.mqtt.client as mqtt


class MqttPublisher:
    """
    Lightweight MQTT publisher for simulated smart-meter messages.
    """

    def __init__(
        self,
        host: str,
        port: int,
        client_id: str,
        keepalive: int = 60,
    ) -> None:
        if not host:
            raise ValueError("MQTT host must not be empty")

        if port <= 0:
            raise ValueError("MQTT port must be greater than zero")

        if not client_id:
            raise ValueError("MQTT client_id must not be empty")

        self.host = host
        self.port = port
        self.client_id = client_id
        self.keepalive = keepalive
        self.connected = False

        self.client = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id=client_id,
            protocol=mqtt.MQTTv311,
        )

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: mqtt.ConnectFlags,
        reason_code: mqtt.ReasonCode,
        properties: mqtt.Properties | None,
    ) -> None:
        if reason_code == 0:
            self.connected = True
            print(
                f"Connected to MQTT broker "
                f"{self.host}:{self.port} "
                f"as {self.client_id}"
            )
        else:
            self.connected = False
            print(
                f"MQTT connection failed: "
                f"reason_code={reason_code}"
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
                f"Unexpected MQTT disconnection: "
                f"reason_code={reason_code}"
            )

    def connect(self) -> None:
        """
        Connect to the MQTT broker and start the network loop.
        """
        self.client.connect(
            host=self.host,
            port=self.port,
            keepalive=self.keepalive,
        )
        self.client.loop_start()

    def publish(
        self,
        topic: str,
        payload: dict[str, Any],
        qos: int = 0,
        retain: bool = False,
    ) -> None:
        """
        Publish one JSON message.
        """
        if not topic:
            raise ValueError("MQTT topic must not be empty")

        encoded_payload = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        result = self.client.publish(
            topic=topic,
            payload=encoded_payload,
            qos=qos,
            retain=retain,
        )

        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(
                f"MQTT publish failed for topic {topic}: "
                f"rc={result.rc}"
            )

    def disconnect(self) -> None:
        """
        Stop the network loop and disconnect cleanly.
        """
        self.client.loop_stop()
        self.client.disconnect()
        self.connected = False