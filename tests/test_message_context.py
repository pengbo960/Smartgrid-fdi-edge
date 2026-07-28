import pytest

from src.simulation.message_context import (
    MessageContext,
    apply_topic_spoof,
)


def test_topic_spoof_replaces_device_topic_level() -> None:
    context = MessageContext(
        topic=(
            "grid/substation_01/"
            "meter_02/measurement"
        ),
        client_id="simulator-topic-spoof",
    )

    spoofed = apply_topic_spoof(
        context=context,
        spoofed_device_id="meter_01",
    )

    assert spoofed.topic == (
        "grid/substation_01/"
        "meter_01/measurement"
    )
    assert spoofed.client_id == context.client_id
    assert spoofed is not context


def test_empty_spoofed_device_is_rejected() -> None:
    context = MessageContext(
        topic=(
            "grid/substation_01/"
            "meter_02/measurement"
        ),
        client_id="simulator-topic-spoof",
    )

    with pytest.raises(
        ValueError,
        match="spoofed_device_id",
    ):
        apply_topic_spoof(
            context=context,
            spoofed_device_id="",
        )


def test_non_measurement_topic_is_rejected() -> None:
    context = MessageContext(
        topic="grid/status",
        client_id="simulator-topic-spoof",
    )

    with pytest.raises(
        ValueError,
        match="device/measurement",
    ):
        apply_topic_spoof(
            context=context,
            spoofed_device_id="meter_01",
        )


def test_empty_context_fields_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="topic",
    ):
        MessageContext(
            topic="",
            client_id="simulator",
        )

    with pytest.raises(
        ValueError,
        match="client_id",
    ):
        MessageContext(
            topic=(
                "grid/substation_01/"
                "meter_02/measurement"
            ),
            client_id="",
        )
