import pandas as pd
import pytest

from src.features.feature_pipeline import (
    FeaturePipeline,
)


def build_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "receive_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:00.050000+00:00"
                ),
                "message_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:00+00:00"
                ),
                "scenario_id": "constant_01",
                "device_id": "meter_01",
                "client_id": "simulator-constant_01",
                "topic": (
                    "grid/substation_01/"
                    "meter_01/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 200,
                "sequence_number": 0,
                "voltage": 230.0,
                "current": 5.0,
                "power": 1092.5,
                "frequency": 50.0,
                "attack_type": "none",
                "is_attack": 0,
                "attack_step": None,
            },
            {
                "receive_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:01.060000+00:00"
                ),
                "message_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:01+00:00"
                ),
                "scenario_id": "constant_01",
                "device_id": "meter_01",
                "client_id": "simulator-constant_01",
                "topic": (
                    "grid/substation_01/"
                    "meter_01/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 200,
                "sequence_number": 1,
                "voltage": 231.0,
                "current": 5.0,
                "power": 1097.25,
                "frequency": 50.0,
                "attack_type": "none",
                "is_attack": 0,
                "attack_step": None,
            },
            {
                "receive_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:02.070000+00:00"
                ),
                "message_timestamp": pd.Timestamp(
                    "2026-07-20T10:00:02+00:00"
                ),
                "scenario_id": "constant_01",
                "device_id": "meter_01",
                "client_id": "simulator-constant_01",
                "topic": (
                    "grid/substation_01/"
                    "meter_01/measurement"
                ),
                "qos": 0,
                "retain": 0,
                "payload_size": 215,
                "sequence_number": 2,
                "voltage": 242.0,
                "current": 5.0,
                "power": 1097.25,
                "frequency": 50.0,
                "attack_type": "constant",
                "is_attack": 1,
                "attack_step": 0,
            },
        ]
    )


def test_pipeline_preserves_row_count() -> None:
    pipeline = FeaturePipeline()

    result = pipeline.transform(
        build_dataframe()
    )

    assert len(result) == 3


def test_pipeline_adds_all_feature_views() -> None:
    pipeline = FeaturePipeline()

    result = pipeline.transform(
        build_dataframe()
    )

    expected_columns = {
        "voltage_diff",
        "voltage_percentage_change",
        "voltage_rolling_mean",
        "voltage_zscore",
        "power_consistency_error",
        "source_publish_interval",
        "gateway_inter_arrival_time",
        "transport_delay_estimate",
        "delay_change",
        "same_value_run_length",
        "payload_size_diff",
        "device_topic_match",
        "client_changed",
        "topic_changed",
        "unexpected_client_topic",
    }

    assert expected_columns.issubset(
        set(result.columns)
    )


def test_pipeline_uses_only_previous_history() -> None:
    pipeline = FeaturePipeline(
        window_size=20,
        minimum_history=2,
    )

    result = pipeline.transform(
        build_dataframe()
    )

    first = result.iloc[0]
    second = result.iloc[1]
    third = result.iloc[2]

    assert first["history_count"] == 0
    assert second["history_count"] == 1
    assert third["history_count"] == 2

    assert first["voltage_diff"] == 0.0
    assert second["voltage_diff"] == 1.0
    assert third["voltage_diff"] == 11.0


def test_pipeline_preserves_labels() -> None:
    pipeline = FeaturePipeline()

    result = pipeline.transform(
        build_dataframe()
    )

    assert result["attack_type"].tolist() == [
        "none",
        "none",
        "constant",
    ]

    assert result["is_attack"].tolist() == [
        0,
        0,
        1,
    ]


def test_pipeline_detects_constant_attack_effects() -> None:
    pipeline = FeaturePipeline()

    result = pipeline.transform(
        build_dataframe()
    )

    attack_row = result.iloc[2]

    assert (
        attack_row["voltage_deviation"]
        > 10.0
    )

    assert (
        attack_row["power_consistency_error"]
        > 0.0
    )

    assert (
        attack_row["payload_size_diff"]
        == 15.0
    )


def test_pipeline_keeps_device_histories_separate() -> None:
    dataframe = build_dataframe()

    second_device = dataframe.iloc[0].copy()
    second_device["device_id"] = "meter_02"
    second_device["topic"] = (
        "grid/substation_01/"
        "meter_02/measurement"
    )
    second_device["sequence_number"] = 0
    second_device["voltage"] = 235.0

    dataframe = pd.concat(
        [
            dataframe,
            pd.DataFrame([second_device]),
        ],
        ignore_index=True,
    )

    pipeline = FeaturePipeline()

    result = pipeline.transform(
        dataframe
    )

    meter_02 = result[
        result["device_id"] == "meter_02"
    ].iloc[0]

    assert meter_02["history_count"] == 0
    assert meter_02["voltage_diff"] == 0.0


def test_pipeline_rejects_empty_dataframe() -> None:
    pipeline = FeaturePipeline()

    with pytest.raises(
        ValueError,
        match="empty dataframe",
    ):
        pipeline.transform(
            pd.DataFrame()
        )


def test_minimum_history_cannot_exceed_window_size() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_history",
    ):
        FeaturePipeline(
            window_size=2,
            minimum_history=3,
        )