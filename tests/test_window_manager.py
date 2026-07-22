import pytest

from src.features.window_manager import (
    WindowManager,
)


def build_row(
    device_id: str,
    sequence_number: int,
    voltage: float,
) -> dict[str, object]:
    return {
        "device_id": device_id,
        "sequence_number": sequence_number,
        "voltage": voltage,
    }


def test_window_manager_tracks_devices_separately() -> None:
    manager = WindowManager(
        window_size=3
    )

    manager.update(
        build_row(
            "meter_01",
            0,
            230.0,
        )
    )

    manager.update(
        build_row(
            "meter_02",
            0,
            231.0,
        )
    )

    meter_01_history = manager.get_history(
        "meter_01"
    )

    meter_02_history = manager.get_history(
        "meter_02"
    )

    assert len(meter_01_history) == 1
    assert len(meter_02_history) == 1

    assert (
        meter_01_history[0]["voltage"]
        == 230.0
    )

    assert (
        meter_02_history[0]["voltage"]
        == 231.0
    )


def test_window_manager_respects_maximum_size() -> None:
    manager = WindowManager(
        window_size=3
    )

    for sequence in range(5):
        manager.update(
            build_row(
                "meter_01",
                sequence,
                230.0 + sequence,
            )
        )

    history = manager.get_history(
        "meter_01"
    )

    assert len(history) == 3

    assert [
        row["sequence_number"]
        for row in history
    ] == [2, 3, 4]


def test_get_previous_returns_latest_row() -> None:
    manager = WindowManager(
        window_size=5
    )

    manager.update(
        build_row(
            "meter_01",
            0,
            230.0,
        )
    )

    manager.update(
        build_row(
            "meter_01",
            1,
            231.0,
        )
    )

    previous = manager.get_previous(
        "meter_01"
    )

    assert previous is not None
    assert previous["sequence_number"] == 1


def test_get_previous_returns_none_for_unknown_device() -> None:
    manager = WindowManager()

    assert (
        manager.get_previous("meter_99")
        is None
    )


def test_returned_history_does_not_modify_internal_state() -> None:
    manager = WindowManager()

    manager.update(
        build_row(
            "meter_01",
            0,
            230.0,
        )
    )

    history = manager.get_history(
        "meter_01"
    )

    history[0]["voltage"] = 999.0

    stored_history = manager.get_history(
        "meter_01"
    )

    assert (
        stored_history[0]["voltage"]
        == 230.0
    )


def test_update_rejects_missing_device_id() -> None:
    manager = WindowManager()

    with pytest.raises(
        KeyError,
        match="device_id",
    ):
        manager.update(
            {
                "voltage": 230.0,
            }
        )


def test_invalid_window_size_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="window_size",
    ):
        WindowManager(
            window_size=0
        )