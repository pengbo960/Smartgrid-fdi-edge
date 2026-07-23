from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _as_timestamp(
    value: Any,
    field_name: str,
) -> pd.Timestamp:
    if value is None:
        raise ValueError(
            f"{field_name} must not be None"
        )

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field_name} must be a valid timestamp"
        ) from exc

    if pd.isna(timestamp):
        raise ValueError(
            f"{field_name} must be a valid timestamp"
        )

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp


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


def _as_int(
    value: Any,
    field_name: str,
) -> int:
    numeric_value = _as_float(
        value,
        field_name,
    )

    if not numeric_value.is_integer():
        raise ValueError(
            f"{field_name} must be an integer"
        )

    return int(numeric_value)


def _same_value(
    first: float,
    second: float,
    tolerance: float,
) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def _count_repeated_values(
    current_value: float,
    history: list[dict[str, Any]],
    field_name: str,
    tolerance: float,
) -> int:
    count = 0

    for row in history:
        if field_name not in row:
            raise KeyError(
                f"Historical row missing field: {field_name}"
            )

        historical_value = _as_float(
            row[field_name],
            field_name,
        )

        if _same_value(
            current_value,
            historical_value,
            tolerance,
        ):
            count += 1

    return count


def _same_value_run_length(
    current_value: float,
    history: list[dict[str, Any]],
    field_name: str,
    tolerance: float,
) -> int:
    run_length = 1

    for row in reversed(history):
        if field_name not in row:
            raise KeyError(
                f"Historical row missing field: {field_name}"
            )

        historical_value = _as_float(
            row[field_name],
            field_name,
        )

        if not _same_value(
            current_value,
            historical_value,
            tolerance,
        ):
            break

        run_length += 1

    return run_length


def extract_temporal_features(
    current_row: dict[str, Any],
    history: list[dict[str, Any]],
    repeated_value_field: str = "voltage",
    value_tolerance: float = 1e-6,
) -> dict[str, float | int]:
    """
    Extract temporal, delay and sequence-behaviour features.

    The history must contain only previous messages from the same device.
    The current message must not already be included in history.
    """
    if value_tolerance < 0:
        raise ValueError(
            "value_tolerance must be zero or greater"
        )

    required_fields = {
        "message_timestamp",
        "receive_timestamp",
        "sequence_number",
        repeated_value_field,
    }

    missing_fields = (
        required_fields - current_row.keys()
    )

    if missing_fields:
        raise KeyError(
            f"Current row missing fields: "
            f"{sorted(missing_fields)}"
        )

    message_timestamp = _as_timestamp(
        current_row["message_timestamp"],
        "message_timestamp",
    )

    receive_timestamp = _as_timestamp(
        current_row["receive_timestamp"],
        "receive_timestamp",
    )

    current_sequence = _as_int(
        current_row["sequence_number"],
        "sequence_number",
    )

    current_value = _as_float(
        current_row[repeated_value_field],
        repeated_value_field,
    )

    transport_delay_estimate = (
        receive_timestamp
        - message_timestamp
    ).total_seconds()

    if not history:
        return {
            "source_publish_interval": 0.0,
            "gateway_inter_arrival_time": 0.0,
            "transport_delay_estimate": round(
                transport_delay_estimate,
                6,
            ),
            "delay_change": 0.0,
            "sequence_gap": 0,
            "is_duplicate_sequence": 0,
            "is_out_of_order": 0,
            "repeated_value_count": 0,
            "same_value_run_length": 1,
        }

    previous = history[-1]

    required_previous_fields = {
        "message_timestamp",
        "receive_timestamp",
        "sequence_number",
    }

    missing_previous = (
        required_previous_fields - previous.keys()
    )

    if missing_previous:
        raise KeyError(
            f"Previous row missing fields: "
            f"{sorted(missing_previous)}"
        )

    previous_message_timestamp = _as_timestamp(
        previous["message_timestamp"],
        "message_timestamp",
    )

    previous_receive_timestamp = _as_timestamp(
        previous["receive_timestamp"],
        "receive_timestamp",
    )

    previous_sequence = _as_int(
        previous["sequence_number"],
        "sequence_number",
    )

    source_publish_interval = (
        message_timestamp
        - previous_message_timestamp
    ).total_seconds()

    gateway_inter_arrival_time = (
        receive_timestamp
        - previous_receive_timestamp
    ).total_seconds()

    previous_transport_delay = (
        previous_receive_timestamp
        - previous_message_timestamp
    ).total_seconds()

    delay_change = (
        transport_delay_estimate
        - previous_transport_delay
    )

    raw_sequence_gap = (
        current_sequence
        - previous_sequence
    )

    is_duplicate_sequence = int(
        raw_sequence_gap == 0
    )

    is_out_of_order = int(
        raw_sequence_gap < 0
    )

    if raw_sequence_gap > 1:
        sequence_gap = (
            raw_sequence_gap - 1
        )
    else:
        sequence_gap = 0

    repeated_value_count = (
        _count_repeated_values(
            current_value=current_value,
            history=history,
            field_name=repeated_value_field,
            tolerance=value_tolerance,
        )
    )

    same_value_run_length = (
        _same_value_run_length(
            current_value=current_value,
            history=history,
            field_name=repeated_value_field,
            tolerance=value_tolerance,
        )
    )

    return {
        "source_publish_interval": round(
            source_publish_interval,
            6,
        ),
        "gateway_inter_arrival_time": round(
            gateway_inter_arrival_time,
            6,
        ),
        "transport_delay_estimate": round(
            transport_delay_estimate,
            6,
        ),
        "delay_change": round(
            delay_change,
            6,
        ),
        "sequence_gap": sequence_gap,
        "is_duplicate_sequence": (
            is_duplicate_sequence
        ),
        "is_out_of_order": is_out_of_order,
        "repeated_value_count": (
            repeated_value_count
        ),
        "same_value_run_length": (
            same_value_run_length
        ),
    }