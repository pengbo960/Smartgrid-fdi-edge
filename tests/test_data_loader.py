from pathlib import Path

import pandas as pd
import pytest

from src.features.data_loader import (
    load_raw_dataset,
    validate_required_columns,
)


def build_valid_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "receive_timestamp": (
                    "2026-07-20T10:00:02+00:00"
                ),
                "message_timestamp": (
                    "2026-07-20T10:00:02+00:00"
                ),
                "scenario_id": "normal_01",
                "device_id": "meter_02",
                "client_id": "simulator-normal_01",
                "topic": (
                    "grid/substation_01/"
                    "meter_02/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 200,
                "sequence_number": 1,
                "voltage": 231.0,
                "current": 4.6,
                "power": 1016.0,
                "frequency": 50.0,
                "attack_type": "none",
                "is_attack": 0,
                "attack_step": None,
            },
            {
                "receive_timestamp": (
                    "2026-07-20T10:00:01+00:00"
                ),
                "message_timestamp": (
                    "2026-07-20T10:00:01+00:00"
                ),
                "scenario_id": "normal_01",
                "device_id": "meter_01",
                "client_id": "simulator-normal_01",
                "topic": (
                    "grid/substation_01/"
                    "meter_01/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 200,
                "sequence_number": 0,
                "voltage": 230.0,
                "current": 4.5,
                "power": 983.0,
                "frequency": 50.0,
                "attack_type": "none",
                "is_attack": 0,
                "attack_step": None,
            },
        ]
    )


def test_validate_required_columns_accepts_valid_data() -> None:
    dataframe = build_valid_dataframe()

    validate_required_columns(dataframe)


def test_validate_required_columns_rejects_missing_column() -> None:
    dataframe = build_valid_dataframe().drop(
        columns=["voltage"]
    )

    with pytest.raises(
        ValueError,
        match="Missing required dataset columns",
    ):
        validate_required_columns(dataframe)


def test_load_raw_dataset_sorts_by_device(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.csv"

    build_valid_dataframe().to_csv(
        path,
        index=False,
    )

    dataframe = load_raw_dataset(path)

    assert dataframe.iloc[0]["device_id"] == (
        "meter_01"
    )

    assert dataframe.iloc[1]["device_id"] == (
        "meter_02"
    )


def test_load_raw_dataset_parses_timestamps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "raw.csv"

    build_valid_dataframe().to_csv(
        path,
        index=False,
    )

    dataframe = load_raw_dataset(path)

    assert isinstance(
        dataframe.iloc[0]["message_timestamp"],
        pd.Timestamp,
    )

    assert (
        dataframe.iloc[0]["message_timestamp"].tzinfo
        is not None
    )


def test_load_raw_dataset_rejects_empty_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "empty.csv"

    pd.DataFrame().to_csv(
        path,
        index=False,
    )

    with pytest.raises(ValueError):
        load_raw_dataset(path)