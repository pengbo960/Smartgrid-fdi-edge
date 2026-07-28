from pathlib import Path

import pandas as pd

from scripts.validate_dataset import (
    validate_file,
)


def build_rows(
    attack_types: list[str],
    sequence_numbers: list[int],
) -> pd.DataFrame:
    row_count = len(attack_types)

    return pd.DataFrame(
        {
            "receive_timestamp": [
                f"2026-01-01T00:00:{index:02d}+00:00"
                for index in range(row_count)
            ],
            "message_timestamp": [
                f"2026-01-01T00:00:{index:02d}+00:00"
                for index in range(row_count)
            ],
            "scenario_id": ["test"] * row_count,
            "device_id": ["meter_01"] * row_count,
            "sequence_number": sequence_numbers,
            "voltage": [230.0] * row_count,
            "current": [5.0] * row_count,
            "power": [1092.5] * row_count,
            "frequency": [50.0] * row_count,
            "attack_type": attack_types,
            "is_attack": [
                int(label != "none")
                for label in attack_types
            ],
        }
    )


def add_other_devices(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    extra_rows = []

    for device_id in (
        "meter_02",
        "meter_03",
    ):
        row = dataframe.iloc[0].copy()
        row["device_id"] = device_id
        extra_rows.append(row)

    return pd.concat(
        [
            dataframe,
            pd.DataFrame(extra_rows),
        ],
        ignore_index=True,
    )


def save_frame(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    dataframe.to_csv(
        path,
        index=False,
    )


def test_validator_accepts_replay_sequence_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "replay.csv"
    dataframe = build_rows(
        attack_types=[
            "none",
            "replay",
        ],
        sequence_numbers=[
            1,
            1,
        ],
    )
    save_frame(
        add_other_devices(dataframe),
        path,
    )

    assert validate_file(path) is True


def test_validator_rejects_normal_sequence_duplicates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "normal.csv"
    dataframe = build_rows(
        attack_types=[
            "none",
            "none",
        ],
        sequence_numbers=[
            1,
            1,
        ],
    )
    save_frame(
        add_other_devices(dataframe),
        path,
    )

    assert validate_file(path) is False
