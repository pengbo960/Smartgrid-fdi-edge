from __future__ import annotations

import math
from statistics import mean, pstdev
from typing import Any


VALUE_FIELDS = (
    "voltage",
    "current",
    "power",
    "frequency",
)


def _as_float(
    value: Any,
    field_name: str,
) -> float:
    """
    Convert a value to float and reject missing or invalid values.
    """
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


def _extract_history_values(
    history: list[dict[str, Any]],
    field_name: str,
) -> list[float]:
    """
    Extract one numeric field from historical messages.
    """
    values: list[float] = []

    for row in history:
        if field_name not in row:
            raise KeyError(
                f"Historical row missing field: "
                f"{field_name}"
            )

        values.append(
            _as_float(
                row[field_name],
                field_name,
            )
        )

    return values


def extract_value_features(
    current_row: dict[str, Any],
    history: list[dict[str, Any]],
    minimum_history: int = 2,
) -> dict[str, float]:
    """
    Extract value-based features for one current message.

    The supplied history must contain only previous messages from the
    same device. The current message must not already be in history.

    Rolling statistics are calculated only from historical values.
    """
    if minimum_history < 1:
        raise ValueError(
            "minimum_history must be at least one"
        )

    features: dict[str, float] = {}

    for field_name in VALUE_FIELDS:
        if field_name not in current_row:
            raise KeyError(
                f"Current row missing field: "
                f"{field_name}"
            )

        current_value = _as_float(
            current_row[field_name],
            field_name,
        )

        history_values = _extract_history_values(
            history,
            field_name,
        )

        if history_values:
            previous_value = history_values[-1]
            difference = (
                current_value - previous_value
            )
        else:
            difference = 0.0

        features[f"{field_name}_diff"] = round(
            difference,
            6,
        )

        if len(history_values) >= minimum_history:
            rolling_mean = mean(
                history_values
            )

            rolling_std = pstdev(
                history_values
            )

            deviation = (
                current_value - rolling_mean
            )

            if rolling_std > 0:
                zscore = (
                    deviation / rolling_std
                )
            else:
                zscore = 0.0

        elif history_values:
            rolling_mean = mean(
                history_values
            )
            rolling_std = 0.0
            deviation = (
                current_value - rolling_mean
            )
            zscore = 0.0

        else:
            rolling_mean = current_value
            rolling_std = 0.0
            deviation = 0.0
            zscore = 0.0

        features[
            f"{field_name}_rolling_mean"
        ] = round(
            rolling_mean,
            6,
        )

        features[
            f"{field_name}_rolling_std"
        ] = round(
            rolling_std,
            6,
        )

        features[
            f"{field_name}_deviation"
        ] = round(
            deviation,
            6,
        )

        features[
            f"{field_name}_zscore"
        ] = round(
            zscore,
            6,
        )

    return features