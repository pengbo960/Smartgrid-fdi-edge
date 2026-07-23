import pytest

from src.features.protocol_features import (
    extract_protocol_features,
)


def build_row(
    device_id: str = "meter_01",
    client_id: str = "simulator-normal_01",
    topic: str = (
        "grid/substation_01/"
        "meter_01/measurement"
    ),
    qos: int = 0,
    retain: int = 0,
    payload_size: float = 200.0,
) -> dict[str, object]:
    return {
        "device_id": device_id,
        "client_id": client_id,
        "topic": topic,
        "qos": qos,
        "retain": retain,
        "payload_size": payload_size,
    }


def test_protocol_features_without_history() -> None:
    current = build_row()

    features = extract_protocol_features(
        current_row=current,
        history=[],
    )

    assert features["payload_size"] == 200.0
    assert features["payload_size_diff"] == 0.0

    assert (
        features["payload_size_rolling_mean"]
        == 200.0
    )

    assert (
        features["payload_size_deviation"]
        == 0.0
    )

    assert features["qos"] == 0
    assert features["retain"] == 0
    assert features["device_topic_match"] == 1
    assert features["client_changed"] == 0
    assert features["topic_changed"] == 0

    assert (
        features["unexpected_client_topic"]
        == 0
    )


def test_payload_size_difference() -> None:
    history = [
        build_row(
            payload_size=200.0
        )
    ]

    current = build_row(
        payload_size=215.0
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert (
        features["payload_size_diff"]
        == 15.0
    )


def test_payload_size_rolling_mean_uses_history() -> None:
    history = [
        build_row(
            payload_size=190.0
        ),
        build_row(
            payload_size=200.0
        ),
        build_row(
            payload_size=210.0
        ),
    ]

    current = build_row(
        payload_size=230.0
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert (
        features["payload_size_rolling_mean"]
        == 200.0
    )

    assert (
        features["payload_size_deviation"]
        == 30.0
    )


def test_device_topic_match() -> None:
    current = build_row(
        device_id="meter_02",
        topic=(
            "grid/substation_01/"
            "meter_02/measurement"
        ),
    )

    features = extract_protocol_features(
        current_row=current,
        history=[],
    )

    assert (
        features["device_topic_match"]
        == 1
    )


def test_device_topic_mismatch_is_detected() -> None:
    current = build_row(
        device_id="meter_02",
        topic=(
            "grid/substation_01/"
            "meter_01/measurement"
        ),
    )

    features = extract_protocol_features(
        current_row=current,
        history=[],
    )

    assert (
        features["device_topic_match"]
        == 0
    )


def test_client_change_is_detected() -> None:
    history = [
        build_row(
            client_id="simulator-normal_01"
        )
    ]

    current = build_row(
        client_id="rogue-client"
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert features["client_changed"] == 1

    assert (
        features["unexpected_client_topic"]
        == 1
    )


def test_unchanged_client_is_not_flagged() -> None:
    history = [
        build_row(
            client_id="simulator-normal_01"
        )
    ]

    current = build_row(
        client_id="simulator-normal_01"
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert features["client_changed"] == 0

    assert (
        features["unexpected_client_topic"]
        == 0
    )


def test_topic_change_is_detected() -> None:
    history = [
        build_row(
            topic=(
                "grid/substation_01/"
                "meter_01/measurement"
            )
        )
    ]

    current = build_row(
        topic=(
            "grid/substation_02/"
            "meter_01/measurement"
        )
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert features["topic_changed"] == 1

    assert (
        features["unexpected_client_topic"]
        == 1
    )


def test_known_historical_client_topic_pair_is_accepted() -> None:
    history = [
        build_row(
            client_id="client_a",
            topic=(
                "grid/substation_01/"
                "meter_01/measurement"
            ),
        ),
        build_row(
            client_id="client_b",
            topic=(
                "grid/substation_01/"
                "meter_01/backup"
            ),
        ),
    ]

    current = build_row(
        client_id="client_a",
        topic=(
            "grid/substation_01/"
            "meter_01/measurement"
        ),
    )

    features = extract_protocol_features(
        current_row=current,
        history=history,
    )

    assert (
        features["unexpected_client_topic"]
        == 0
    )


def test_qos_values_are_preserved() -> None:
    for qos in [0, 1, 2]:
        features = extract_protocol_features(
            current_row=build_row(
                qos=qos
            ),
            history=[],
        )

        assert features["qos"] == qos


def test_invalid_qos_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="qos",
    ):
        extract_protocol_features(
            current_row=build_row(
                qos=3
            ),
            history=[],
        )


def test_invalid_retain_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="retain",
    ):
        extract_protocol_features(
            current_row=build_row(
                retain=2
            ),
            history=[],
        )


def test_negative_payload_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="payload_size",
    ):
        extract_protocol_features(
            current_row=build_row(
                payload_size=-1
            ),
            history=[],
        )


def test_missing_current_field_is_rejected() -> None:
    current = build_row()
    del current["topic"]

    with pytest.raises(
        KeyError,
        match="Current row missing fields",
    ):
        extract_protocol_features(
            current_row=current,
            history=[],
        )


def test_missing_historical_field_is_rejected() -> None:
    previous = build_row()
    del previous["payload_size"]

    with pytest.raises(
        KeyError,
        match="Previous row missing fields",
    ):
        extract_protocol_features(
            current_row=build_row(),
            history=[previous],
        )