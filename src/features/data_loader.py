from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_REQUIRED_COLUMNS = {
    "receive_timestamp",
    "message_timestamp",
    "scenario_id",
    "device_id",
    "client_id",
    "topic",
    "qos",
    "retain",
    "payload_size",
    "sequence_number",
    "voltage",
    "current",
    "power",
    "frequency",
    "attack_type",
    "is_attack",
    "attack_step",
}


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
) -> None:
    """
    Validate that the raw dataset contains all required columns.
    """
    required = set(required_columns)
    missing = required - set(dataframe.columns)

    if missing:
        raise ValueError(
            "Missing required dataset columns: "
            f"{sorted(missing)}"
        )


def load_raw_dataset(
    path: str | Path,
    required_columns: Iterable[str] = DEFAULT_REQUIRED_COLUMNS,
) -> pd.DataFrame:
    """
    Load, validate and sort a raw MQTT dataset.

    Sorting is performed by device_id and receive_timestamp so that
    each device's gateway-observed arrival order is preserved.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {dataset_path}"
        )

    validate_required_columns(
        dataframe=dataframe,
        required_columns=required_columns,
    )

    dataframe = dataframe.copy()

    dataframe["receive_timestamp"] = pd.to_datetime(
        dataframe["receive_timestamp"],
        utc=True,
        errors="coerce",
    )

    dataframe["message_timestamp"] = pd.to_datetime(
        dataframe["message_timestamp"],
        utc=True,
        errors="coerce",
    )

    invalid_receive_timestamps = int(
        dataframe["receive_timestamp"].isna().sum()
    )

    invalid_message_timestamps = int(
        dataframe["message_timestamp"].isna().sum()
    )

    if invalid_receive_timestamps > 0:
        raise ValueError(
            "Invalid receive_timestamp values: "
            f"{invalid_receive_timestamps}"
        )

    if invalid_message_timestamps > 0:
        raise ValueError(
            "Invalid message_timestamp values: "
            f"{invalid_message_timestamps}"
        )

    numeric_columns = [
        "qos",
        "retain",
        "payload_size",
        "sequence_number",
        "voltage",
        "current",
        "power",
        "frequency",
        "is_attack",
        "attack_step",
    ]

    for column in numeric_columns:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    required_numeric_columns = [
        "qos",
        "retain",
        "payload_size",
        "sequence_number",
        "voltage",
        "current",
        "power",
        "frequency",
        "is_attack",
    ]

    invalid_numeric = {
        column: int(dataframe[column].isna().sum())
        for column in required_numeric_columns
        if dataframe[column].isna().any()
    }

    if invalid_numeric:
        raise ValueError(
            "Invalid numeric values found: "
            f"{invalid_numeric}"
        )

    dataframe["device_id"] = (
        dataframe["device_id"]
        .astype(str)
        .str.strip()
    )

    if (dataframe["device_id"] == "").any():
        raise ValueError(
            "device_id contains empty values"
        )

    dataframe = dataframe.sort_values(
        by=[
            "device_id",
            "receive_timestamp",
        ],
        kind="stable",
    ).reset_index(drop=True)

    return dataframe