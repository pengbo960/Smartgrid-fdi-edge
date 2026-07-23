from __future__ import annotations

import math
from statistics import mean
from typing import Any


def _as_float(
    value: Any,
    field_name: str,
) -> float:
    if value is None:
        raise ValueError(
            f"{field_name} must not be None"
        )

    try:
        numeric_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be numeric"
        ) from exc

    if not math.isfinite(numeric_value):
        raise ValueError(
            f"{field_name} must be finite"
        )

    return numeric_value


def _as_binary_int(
    value: Any,
    field_name: str,
) -> int:
    numeric_value = _as_float(
        value,
        field_name,
    )

    integer_value = int(numeric_value)

    if numeric_value != integer_value:
        raise ValueError(
            f"{field_name} must be an integer"
        )

    if integer_value not in {0, 1}:
        raise ValueError(
            f"{field_name} must be 0 or 1"
        )

    return integer_value


def _as_qos(
    value: Any,
) -> int:
    numeric_value = _as_float(
        value,
        "qos",
    )

    integer_value = int(numeric_value)

    if numeric_value != integer_value:
        raise ValueError(
            "qos must be an integer"
        )

    if integer_value not in {0, 1, 2}:
        raise ValueError(
            "qos must be 0, 1 or 2"
        )

    return integer_value


def _as_non_empty_string(
    value: Any,
    field_name: str,
) -> str:
    if value is None:
        raise ValueError(
            f"{field_name} must not be None"
        )

    text = str(value).strip()

    if not text:
        raise ValueError(
            f"{field_name} must not be empty"
        )

    return text


def _expected_topic_contains_device(
    topic: str,
    device_id: str,
) -> int:
    """
    Return 1 when the device ID appears as a complete topic level.
    """
    topic_levels = [
        level
        for level in topic.split("/")
        if level
    ]

    return int(
        device_id in topic_levels
    )


def _historical_client_topic_pairs(
    history: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()

    for row in history:
        if "client_id" not in row:
            raise KeyError(
                "Historical row missing field: client_id"
            )

        if "topic" not in row:
            raise KeyError(
                "Historical row missing field: topic"
            )

        client_id = _as_non_empty_string(
            row["client_id"],
            "client_id",
        )

        topic = _as_non_empty_string(
            row["topic"],
            "topic",
        )

        pairs.add(
            (
                client_id,
                topic,
            )
        )

    return pairs


def extract_protocol_features(
    current_row: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, float | int]:
    """
    Extract lightweight MQTT and communication-context features.

    The history must contain only previous messages for the same device.
    The current message must not already be included in history.
    """
    required_fields = {
        "device_id",
        "client_id",
        "topic",
        "qos",
        "retain",
        "payload_size",
    }

    missing_fields = (
        required_fields - current_row.keys()
    )

    if missing_fields:
        raise KeyError(
            f"Current row missing fields: "
            f"{sorted(missing_fields)}"
        )

    device_id = _as_non_empty_string(
        current_row["device_id"],
        "device_id",
    )

    client_id = _as_non_empty_string(
        current_row["client_id"],
        "client_id",
    )

    topic = _as_non_empty_string(
        current_row["topic"],
        "topic",
    )

    qos = _as_qos(
        current_row["qos"]
    )

    retain = _as_binary_int(
        current_row["retain"],
        "retain",
    )

    payload_size = _as_float(
        current_row["payload_size"],
        "payload_size",
    )

    if payload_size < 0:
        raise ValueError(
            "payload_size must be zero or greater"
        )

    device_topic_match = (
        _expected_topic_contains_device(
            topic=topic,
            device_id=device_id,
        )
    )

    if not history:
        return {
            "payload_size": round(
                payload_size,
                6,
            ),
            "payload_size_diff": 0.0,
            "payload_size_rolling_mean": round(
                payload_size,
                6,
            ),
            "payload_size_deviation": 0.0,
            "qos": qos,
            "retain": retain,
            "device_topic_match": (
                device_topic_match
            ),
            "client_changed": 0,
            "topic_changed": 0,
            "unexpected_client_topic": 0,
        }

    previous = history[-1]

    required_previous_fields = {
        "client_id",
        "topic",
        "payload_size",
    }

    missing_previous = (
        required_previous_fields - previous.keys()
    )

    if missing_previous:
        raise KeyError(
            f"Previous row missing fields: "
            f"{sorted(missing_previous)}"
        )

    previous_client_id = (
        _as_non_empty_string(
            previous["client_id"],
            "client_id",
        )
    )

    previous_topic = _as_non_empty_string(
        previous["topic"],
        "topic",
    )

    previous_payload_size = _as_float(
        previous["payload_size"],
        "payload_size",
    )

    historical_payload_sizes = [
        _as_float(
            row["payload_size"],
            "payload_size",
        )
        for row in history
    ]

    payload_size_rolling_mean = mean(
        historical_payload_sizes
    )

    payload_size_diff = (
        payload_size
        - previous_payload_size
    )

    payload_size_deviation = (
        payload_size
        - payload_size_rolling_mean
    )

    client_changed = int(
        client_id != previous_client_id
    )

    topic_changed = int(
        topic != previous_topic
    )

    known_pairs = (
        _historical_client_topic_pairs(
            history
        )
    )

    unexpected_client_topic = int(
        (client_id, topic)
        not in known_pairs
    )

    return {
        "payload_size": round(
            payload_size,
            6,
        ),
        "payload_size_diff": round(
            payload_size_diff,
            6,
        ),
        "payload_size_rolling_mean": round(
            payload_size_rolling_mean,
            6,
        ),
        "payload_size_deviation": round(
            payload_size_deviation,
            6,
        ),
        "qos": qos,
        "retain": retain,
        "device_topic_match": (
            device_topic_match
        ),
        "client_changed": client_changed,
        "topic_changed": topic_changed,
        "unexpected_client_topic": (
            unexpected_client_topic
        ),
    }