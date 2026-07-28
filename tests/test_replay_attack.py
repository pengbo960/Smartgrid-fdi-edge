import pytest

from src.simulation.attacks.replay import (
    ReplayBuffer,
)


def build_payload(
    sequence_number: int,
) -> dict[str, object]:
    return {
        "scenario_id": "replay_test",
        "device_id": "meter_02",
        "sequence_number": sequence_number,
        "timestamp": (
            f"2026-07-20T10:00:"
            f"{sequence_number:02d}+00:00"
        ),
        "voltage": 230.0 + sequence_number,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
        "attack_type": "none",
        "is_attack": 0,
        "attack_step": None,
    }


def test_replay_buffer_stores_messages() -> None:
    buffer = ReplayBuffer(
        maximum_size=5
    )

    buffer.store(
        build_payload(0)
    )

    assert buffer.size() == 1


def test_replay_returns_previous_payload() -> None:
    buffer = ReplayBuffer(
        maximum_size=5
    )

    for sequence in range(3):
        buffer.store(
            build_payload(sequence)
        )

    replayed = buffer.replay(
        lag_steps=2
    )

    assert (
        replayed["sequence_number"]
        == 1
    )

    assert (
        replayed["voltage"]
        == 231.0
    )


def test_replayed_payload_is_a_copy() -> None:
    buffer = ReplayBuffer()

    original = build_payload(0)

    buffer.store(original)

    replayed = buffer.replay(
        lag_steps=1
    )

    replayed["voltage"] = 999.0

    second_replay = buffer.replay(
        lag_steps=1
    )

    assert (
        second_replay["voltage"]
        == 230.0
    )


def test_stored_payload_is_a_copy() -> None:
    buffer = ReplayBuffer()

    original = build_payload(0)

    buffer.store(original)

    original["voltage"] = 999.0

    replayed = buffer.replay(
        lag_steps=1
    )

    assert (
        replayed["voltage"]
        == 230.0
    )


def test_buffer_respects_maximum_size() -> None:
    buffer = ReplayBuffer(
        maximum_size=3
    )

    for sequence in range(5):
        buffer.store(
            build_payload(sequence)
        )

    assert buffer.size() == 3

    oldest_available = buffer.replay(
        lag_steps=3
    )

    assert (
        oldest_available[
            "sequence_number"
        ]
        == 2
    )


def test_insufficient_history_is_rejected() -> None:
    buffer = ReplayBuffer()

    buffer.store(
        build_payload(0)
    )

    with pytest.raises(
        ValueError,
        match="Not enough messages",
    ):
        buffer.replay(
            lag_steps=2
        )


def test_invalid_lag_is_rejected() -> None:
    buffer = ReplayBuffer()

    with pytest.raises(
        ValueError,
        match="lag_steps",
    ):
        buffer.replay(
            lag_steps=0
        )


def test_invalid_maximum_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="maximum_size",
    ):
        ReplayBuffer(
            maximum_size=0
        )


def test_empty_payload_is_rejected() -> None:
    buffer = ReplayBuffer()

    with pytest.raises(
        ValueError,
        match="payload must not be empty",
    ):
        buffer.store({})