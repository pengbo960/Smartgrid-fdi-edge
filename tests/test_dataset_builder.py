from pathlib import Path

import pandas as pd
import pytest

from src.features.dataset_builder import (
    FeatureDatasetBuilder,
)
from src.features.feature_pipeline import (
    FeaturePipeline,
)


def build_raw_dataframe(
    scenario_id: str,
    attack_type: str,
    is_attack: int,
) -> pd.DataFrame:
    rows = []

    for sequence_number in range(3):
        rows.append(
            {
                "receive_timestamp": (
                    f"2026-07-20T10:00:0"
                    f"{sequence_number}.050000+00:00"
                ),
                "message_timestamp": (
                    f"2026-07-20T10:00:0"
                    f"{sequence_number}+00:00"
                ),
                "scenario_id": scenario_id,
                "device_id": "meter_01",
                "client_id": (
                    f"simulator-{scenario_id}"
                ),
                "topic": (
                    "grid/substation_01/"
                    "meter_01/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 200,
                "sequence_number": (
                    sequence_number
                ),
                "voltage": (
                    242.0
                    if is_attack
                    else 230.0
                    + sequence_number
                ),
                "current": 5.0,
                "power": 1092.5,
                "frequency": 50.0,
                "attack_type": attack_type,
                "is_attack": is_attack,
                "attack_step": (
                    sequence_number
                    if is_attack
                    else None
                ),
            }
        )

    return pd.DataFrame(rows)


def test_discover_csv_files(
    tmp_path: Path,
) -> None:
    first = tmp_path / "normal.csv"
    second = tmp_path / "constant.csv"

    build_raw_dataframe(
        "normal_01",
        "none",
        0,
    ).to_csv(
        first,
        index=False,
    )

    build_raw_dataframe(
        "constant_01",
        "constant",
        1,
    ).to_csv(
        second,
        index=False,
    )

    builder = FeatureDatasetBuilder(
        pipeline=FeaturePipeline()
    )

    files = builder.discover_csv_files(
        tmp_path
    )

    assert files == [
        second,
        first,
    ]


def test_build_from_multiple_files(
    tmp_path: Path,
) -> None:
    normal_path = (
        tmp_path / "normal.csv"
    )

    attack_path = (
        tmp_path / "constant.csv"
    )

    build_raw_dataframe(
        "normal_01",
        "none",
        0,
    ).to_csv(
        normal_path,
        index=False,
    )

    build_raw_dataframe(
        "constant_01",
        "constant",
        1,
    ).to_csv(
        attack_path,
        index=False,
    )

    builder = FeatureDatasetBuilder(
        pipeline=FeaturePipeline()
    )

    result = builder.build_from_files(
        [
            normal_path,
            attack_path,
        ]
    )

    assert len(result) == 6

    assert set(
        result["source_file"]
    ) == {
        "normal.csv",
        "constant.csv",
    }


def test_history_resets_between_files(
    tmp_path: Path,
) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"

    build_raw_dataframe(
        "normal_01",
        "none",
        0,
    ).to_csv(
        first_path,
        index=False,
    )

    build_raw_dataframe(
        "normal_02",
        "none",
        0,
    ).to_csv(
        second_path,
        index=False,
    )

    builder = FeatureDatasetBuilder(
        pipeline=FeaturePipeline()
    )

    result = builder.build_from_files(
        [
            first_path,
            second_path,
        ]
    )

    first_rows = result.groupby(
        "source_file",
        sort=False,
    ).head(1)

    assert (
        first_rows["history_count"]
        == 0
    ).all()


def test_validate_output_creates_report() -> None:
    dataframe = pd.DataFrame(
        {
            "source_file": [
                "normal.csv",
                "attack.csv",
            ],
            "device_id": [
                "meter_01",
                "meter_01",
            ],
            "sequence_number": [
                0,
                0,
            ],
            "attack_type": [
                "none",
                "constant",
            ],
            "is_attack": [
                0,
                1,
            ],
            "feature": [
                1.0,
                2.0,
            ],
        }
    )

    report = (
        FeatureDatasetBuilder
        .validate_output(dataframe)
    )

    assert report["row_count"] == 2
    assert report["column_count"] == 6
    assert report["duplicate_rows"] == 0

    assert report[
        "attack_distribution"
    ] == {
        "none": 1,
        "constant": 1,
    }


def test_validate_output_detects_duplicates() -> None:
    dataframe = pd.DataFrame(
        {
            "source_file": [
                "normal.csv",
                "normal.csv",
            ],
            "device_id": [
                "meter_01",
                "meter_01",
            ],
            "sequence_number": [
                0,
                0,
            ],
            "attack_type": [
                "none",
                "none",
            ],
            "is_attack": [
                0,
                0,
            ],
        }
    )

    report = (
        FeatureDatasetBuilder
        .validate_output(dataframe)
    )

    assert report["duplicate_rows"] == 1


def test_missing_input_directory_is_rejected(
    tmp_path: Path,
) -> None:
    builder = FeatureDatasetBuilder(
        pipeline=FeaturePipeline()
    )

    with pytest.raises(
        FileNotFoundError,
        match="Input directory",
    ):
        builder.discover_csv_files(
            tmp_path / "missing"
        )


def test_empty_input_directory_is_rejected(
    tmp_path: Path,
) -> None:
    builder = FeatureDatasetBuilder(
        pipeline=FeaturePipeline()
    )

    with pytest.raises(
        ValueError,
        match="No CSV files",
    ):
        builder.discover_csv_files(
            tmp_path
        )