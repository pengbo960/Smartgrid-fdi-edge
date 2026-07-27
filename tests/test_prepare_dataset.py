import pandas as pd
import pytest

from src.training.prepare_dataset import (
    prepare_known_attack_dataset,
    select_feature_columns,
)


def build_feature_row(
    source_file: str,
    attack_type: str,
    is_attack: int,
    history_count: int = 5,
) -> dict[str, object]:
    return {
        "source_file": source_file,
        "attack_type": attack_type,
        "is_attack": is_attack,
        "history_count": history_count,
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
        "voltage_diff": 0.0,
        "voltage_percentage_change": 0.0,
        "voltage_rolling_mean": 230.0,
        "voltage_rolling_std": 0.5,
        "voltage_deviation": 0.0,
        "voltage_zscore": 0.0,
        "current_diff": 0.0,
        "current_percentage_change": 0.0,
        "current_rolling_mean": 5.0,
        "current_rolling_std": 0.1,
        "current_deviation": 0.0,
        "current_zscore": 0.0,
        "power_diff": 0.0,
        "power_percentage_change": 0.0,
        "power_rolling_mean": 1092.5,
        "power_rolling_std": 1.0,
        "power_deviation": 0.0,
        "power_zscore": 0.0,
        "frequency_diff": 0.0,
        "frequency_percentage_change": 0.0,
        "frequency_rolling_mean": 50.0,
        "frequency_rolling_std": 0.01,
        "frequency_deviation": 0.0,
        "frequency_zscore": 0.0,
        "power_consistency_error": 0.0,
        "source_publish_interval": 1.0,
        "gateway_inter_arrival_time": 1.0,
        "transport_delay_estimate": 0.05,
        "delay_change": 0.0,
        "sequence_gap": 0,
        "is_duplicate_sequence": 0,
        "is_out_of_order": 0,
        "repeated_value_count": 0,
        "same_value_run_length": 1,
        "payload_size": 200.0,
        "payload_size_diff": 0.0,
        "payload_size_rolling_mean": 200.0,
        "payload_size_deviation": 0.0,
        "qos": 0,
        "retain": 0,
        "device_topic_match": 1,
        "client_changed": 0,
        "topic_changed": 0,
        "unexpected_client_topic": 0,
    }


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            build_feature_row(
                source_file="normal_run_01.csv",
                attack_type="none",
                is_attack=0,
                history_count=0,
            ),
            build_feature_row(
                source_file="constant_run_01.csv",
                attack_type="constant",
                is_attack=1,
            ),
            build_feature_row(
                source_file="random_run_01.csv",
                attack_type="random",
                is_attack=1,
            ),
            build_feature_row(
                source_file="gradual_run_01.csv",
                attack_type="none",
                is_attack=0,
            ),
            build_feature_row(
                source_file="gradual_run_01.csv",
                attack_type="gradual",
                is_attack=1,
            ),
        ]
    )


def test_select_value_and_temporal_features() -> None:
    dataframe = build_dataframe()

    features = select_feature_columns(
        dataframe=dataframe,
        include_groups=[
            "value",
            "temporal",
        ],
    )

    assert "voltage_diff" in features
    assert "same_value_run_length" in features
    assert "payload_size" not in features


def test_prepare_dataset_keeps_known_attacks() -> None:
    prepared = prepare_known_attack_dataset(
        dataframe=build_dataframe(),
        known_attack_types=[
            "none",
            "constant",
            "random",
        ],
        include_groups=[
            "value",
            "temporal",
        ],
    )

    assert set(
        prepared.dataframe["attack_type"]
    ) == {
        "none",
        "constant",
        "random",
    }


def test_excluded_attack_removes_entire_source_file() -> None:
    prepared = prepare_known_attack_dataset(
        dataframe=build_dataframe(),
        known_attack_types=[
            "none",
            "constant",
            "random",
        ],
        excluded_attack_types=[
            "gradual",
        ],
        include_groups=[
            "value",
            "temporal",
        ],
    )

    remaining_files = set(
        prepared.dataframe[
            "source_file"
        ]
    )

    assert (
        "gradual_run_01.csv"
        not in remaining_files
    )

    assert remaining_files == {
        "normal_run_01.csv",
        "constant_run_01.csv",
        "random_run_01.csv",
    }


def test_prepare_dataset_preserves_binary_target() -> None:
    prepared = prepare_known_attack_dataset(
        dataframe=build_dataframe(),
        known_attack_types=[
            "none",
            "constant",
            "random",
        ],
        excluded_attack_types=[
            "gradual",
        ],
    )

    assert set(
        prepared.dataframe["is_attack"]
    ) == {
        0,
        1,
    }


def test_drop_warmup_rows() -> None:
    dataframe = build_dataframe()

    prepared = prepare_known_attack_dataset(
        dataframe=dataframe,
        known_attack_types=[
            "none",
            "constant",
            "random",
        ],
        excluded_attack_types=[
            "gradual",
        ],
        drop_warmup_rows=True,
    )

    assert (
        prepared.dataframe[
            "history_count"
        ] > 0
    ).all()

    assert (
        "normal_run_01.csv"
        not in set(
            prepared.dataframe[
                "source_file"
            ]
        )
    )


def test_unknown_feature_group_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown feature group",
    ):
        select_feature_columns(
            dataframe=build_dataframe(),
            include_groups=[
                "unknown",
            ],
        )


def test_missing_feature_column_is_rejected() -> None:
    dataframe = build_dataframe().drop(
        columns=["voltage_diff"]
    )

    with pytest.raises(
        ValueError,
        match="Feature column missing",
    ):
        prepare_known_attack_dataset(
            dataframe=dataframe,
            known_attack_types=[
                "none",
                "constant",
                "random",
            ],
            excluded_attack_types=[
                "gradual",
            ],
        )


def test_invalid_missing_fraction_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_missing_fraction",
    ):
        prepare_known_attack_dataset(
            dataframe=build_dataframe(),
            known_attack_types=[
                "none",
                "constant",
                "random",
            ],
            maximum_missing_fraction=1.1,
        )