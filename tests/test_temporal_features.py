import pytest

from src.features.temporal_features import (
    extract_temporal_features,
)


def build_row(
    sequence_number: int,
    message_timestamp: str,
    receive_timestamp: str | None = None,
    voltage: float = 230.0,
) -> dict[str, object]:
    return {
        "device_id": "meter_01",
        "sequence_number": sequence_number,
        "message_timestamp": message_timestamp,
        "receive_timestamp": (
            receive_timestamp
            if receive_timestamp is not None
            else message_timestamp
        ),
        "voltage": voltage,
    }


def test_temporal_features_without_history() -> None:
    current = build_row(
        sequence_number=0,
        message_timestamp=(
            "2026-07-20T10:00:00+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=[],
    )

    assert (
        features["source_publish_interval"]
        == 0.0
    )

    assert (
        features["gateway_inter_arrival_time"]
        == 0.0
    )

    assert (
        features["transport_delay_estimate"]
        == 0.0
    )

    assert features["delay_change"] == 0.0
    assert features["sequence_gap"] == 0
    assert features["is_duplicate_sequence"] == 0
    assert features["is_out_of_order"] == 0
    assert features["repeated_value_count"] == 0
    assert features["same_value_run_length"] == 1


def test_source_publish_interval() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            receive_timestamp=(
                "2026-07-20T10:00:00.100000+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=1,
        message_timestamp=(
            "2026-07-20T10:00:01.250000+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:01.400000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["source_publish_interval"]
        == 1.25
    )


def test_gateway_inter_arrival_time() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            receive_timestamp=(
                "2026-07-20T10:00:00.100000+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=1,
        message_timestamp=(
            "2026-07-20T10:00:01+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:01.350000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["gateway_inter_arrival_time"]
        == 1.25
    )


def test_transport_delay_estimate() -> None:
    current = build_row(
        sequence_number=0,
        message_timestamp=(
            "2026-07-20T10:00:00+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:00.125000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=[],
    )

    assert (
        features["transport_delay_estimate"]
        == 0.125
    )


def test_delay_change() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            receive_timestamp=(
                "2026-07-20T10:00:00.100000+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=1,
        message_timestamp=(
            "2026-07-20T10:00:01+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:01.250000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert features["delay_change"] == 0.15


def test_sequence_gap_counts_missing_messages() -> None:
    history = [
        build_row(
            sequence_number=4,
            message_timestamp=(
                "2026-07-20T10:00:04+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=7,
        message_timestamp=(
            "2026-07-20T10:00:07+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert features["sequence_gap"] == 2


def test_consecutive_sequence_has_no_gap() -> None:
    history = [
        build_row(
            sequence_number=4,
            message_timestamp=(
                "2026-07-20T10:00:04+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=5,
        message_timestamp=(
            "2026-07-20T10:00:05+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert features["sequence_gap"] == 0
    assert features["is_duplicate_sequence"] == 0
    assert features["is_out_of_order"] == 0


def test_duplicate_sequence_is_detected() -> None:
    history = [
        build_row(
            sequence_number=4,
            message_timestamp=(
                "2026-07-20T10:00:04+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=4,
        message_timestamp=(
            "2026-07-20T10:00:05+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert features["is_duplicate_sequence"] == 1
    assert features["is_out_of_order"] == 0
    assert features["sequence_gap"] == 0


def test_out_of_order_sequence_is_detected() -> None:
    history = [
        build_row(
            sequence_number=5,
            message_timestamp=(
                "2026-07-20T10:00:05+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=3,
        message_timestamp=(
            "2026-07-20T10:00:06+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert features["is_out_of_order"] == 1
    assert features["is_duplicate_sequence"] == 0
    assert features["sequence_gap"] == 0


def test_repeated_value_count() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            voltage=242.0,
        ),
        build_row(
            sequence_number=1,
            message_timestamp=(
                "2026-07-20T10:00:01+00:00"
            ),
            voltage=230.0,
        ),
        build_row(
            sequence_number=2,
            message_timestamp=(
                "2026-07-20T10:00:02+00:00"
            ),
            voltage=242.0,
        ),
    ]

    current = build_row(
        sequence_number=3,
        message_timestamp=(
            "2026-07-20T10:00:03+00:00"
        ),
        voltage=242.0,
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["repeated_value_count"]
        == 2
    )


def test_same_value_run_length_counts_consecutive_values() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            voltage=230.0,
        ),
        build_row(
            sequence_number=1,
            message_timestamp=(
                "2026-07-20T10:00:01+00:00"
            ),
            voltage=242.0,
        ),
        build_row(
            sequence_number=2,
            message_timestamp=(
                "2026-07-20T10:00:02+00:00"
            ),
            voltage=242.0,
        ),
    ]

    current = build_row(
        sequence_number=3,
        message_timestamp=(
            "2026-07-20T10:00:03+00:00"
        ),
        voltage=242.0,
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["same_value_run_length"]
        == 3
    )


def test_same_value_run_length_resets_after_change() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            voltage=242.0,
        ),
        build_row(
            sequence_number=1,
            message_timestamp=(
                "2026-07-20T10:00:01+00:00"
            ),
            voltage=242.0,
        ),
    ]

    current = build_row(
        sequence_number=2,
        message_timestamp=(
            "2026-07-20T10:00:02+00:00"
        ),
        voltage=231.0,
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["same_value_run_length"]
        == 1
    )


def test_value_tolerance_is_used() -> None:
    history = [
        build_row(
            sequence_number=0,
            message_timestamp=(
                "2026-07-20T10:00:00+00:00"
            ),
            voltage=242.0001,
        )
    ]

    current = build_row(
        sequence_number=1,
        message_timestamp=(
            "2026-07-20T10:00:01+00:00"
        ),
        voltage=242.0002,
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
        value_tolerance=0.001,
    )

    assert (
        features["repeated_value_count"]
        == 1
    )

    assert (
        features["same_value_run_length"]
        == 2
    )


def test_negative_source_publish_interval_is_preserved() -> None:
    history = [
        build_row(
            sequence_number=5,
            message_timestamp=(
                "2026-07-20T10:00:05+00:00"
            ),
            receive_timestamp=(
                "2026-07-20T10:00:05.100000+00:00"
            ),
        )
    ]

    current = build_row(
        sequence_number=4,
        message_timestamp=(
            "2026-07-20T10:00:04+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:06.100000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=history,
    )

    assert (
        features["source_publish_interval"]
        == -1.0
    )

    assert (
        features["gateway_inter_arrival_time"]
        == 1.0
    )

    assert features["is_out_of_order"] == 1


def test_negative_transport_delay_is_preserved() -> None:
    current = build_row(
        sequence_number=0,
        message_timestamp=(
            "2026-07-20T10:00:01+00:00"
        ),
        receive_timestamp=(
            "2026-07-20T10:00:00.900000+00:00"
        ),
    )

    features = extract_temporal_features(
        current_row=current,
        history=[],
    )

    assert (
        features["transport_delay_estimate"]
        == -0.1
    )


def test_missing_current_field_is_rejected() -> None:
    current = build_row(
        sequence_number=0,
        message_timestamp=(
            "2026-07-20T10:00:00+00:00"
        ),
    )

    del current["receive_timestamp"]

    with pytest.raises(
        KeyError,
        match="Current row missing fields",
    ):
        extract_temporal_features(
            current_row=current,
            history=[],
        )


def test_missing_previous_receive_timestamp_is_rejected() -> None:
    previous = build_row(
        sequence_number=0,
        message_timestamp=(
            "2026-07-20T10:00:00+00:00"
        ),
    )

    del previous["receive_timestamp"]

    current = build_row(
        sequence_number=1,
        message_timestamp=(
            "2026-07-20T10:00:01+00:00"
        ),
    )

    with pytest.raises(
        KeyError,
        match="Previous row missing fields",
    ):
        extract_temporal_features(
            current_row=current,
            history=[previous],
        )


def test_negative_tolerance_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="value_tolerance",
    ):
        extract_temporal_features(
            current_row=build_row(
                sequence_number=0,
                message_timestamp=(
                    "2026-07-20T10:00:00+00:00"
                ),
            ),
            history=[],
            value_tolerance=-1.0,
        )