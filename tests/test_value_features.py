import pytest

from src.features.value_features import (
    extract_value_features,
)


def build_row(
    voltage: float,
    current: float = 5.0,
    power: float = 1092.5,
    frequency: float = 50.0,
) -> dict[str, float]:
    return {
        "voltage": voltage,
        "current": current,
        "power": power,
        "frequency": frequency,
    }


def test_value_features_without_history() -> None:
    current = build_row(
        voltage=230.0
    )

    features = extract_value_features(
        current_row=current,
        history=[],
        minimum_history=2,
    )

    assert features["voltage_diff"] == 0.0
    assert (
        features["voltage_rolling_mean"]
        == 230.0
    )
    assert (
        features["voltage_rolling_std"]
        == 0.0
    )
    assert (
        features["voltage_deviation"]
        == 0.0
    )
    assert (
        features["voltage_zscore"]
        == 0.0
    )


def test_value_difference_uses_previous_message() -> None:
    history = [
        build_row(voltage=229.0),
        build_row(voltage=230.0),
    ]

    current = build_row(
        voltage=232.5
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    assert (
        features["voltage_diff"]
        == 2.5
    )


def test_rolling_mean_uses_only_history() -> None:
    history = [
        build_row(voltage=228.0),
        build_row(voltage=230.0),
        build_row(voltage=232.0),
    ]

    current = build_row(
        voltage=242.0
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    assert (
        features["voltage_rolling_mean"]
        == 230.0
    )

    assert (
        features["voltage_deviation"]
        == 12.0
    )


def test_rolling_standard_deviation() -> None:
    history = [
        build_row(voltage=228.0),
        build_row(voltage=230.0),
        build_row(voltage=232.0),
    ]

    current = build_row(
        voltage=234.0
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    assert features[
        "voltage_rolling_std"
    ] == pytest.approx(
        1.632993,
        abs=1e-6,
    )


def test_zscore_is_calculated() -> None:
    history = [
        build_row(voltage=228.0),
        build_row(voltage=230.0),
        build_row(voltage=232.0),
    ]

    current = build_row(
        voltage=234.0
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    assert features[
        "voltage_zscore"
    ] == pytest.approx(
        2.44949,
        abs=1e-6,
    )


def test_zero_standard_deviation_returns_zero_zscore() -> None:
    history = [
        build_row(voltage=230.0),
        build_row(voltage=230.0),
        build_row(voltage=230.0),
    ]

    current = build_row(
        voltage=242.0
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    assert (
        features["voltage_rolling_std"]
        == 0.0
    )

    assert (
        features["voltage_zscore"]
        == 0.0
    )

    assert (
        features["voltage_deviation"]
        == 12.0
    )


def test_features_are_created_for_all_value_fields() -> None:
    history = [
        build_row(
            voltage=230.0,
            current=5.0,
            power=1092.5,
            frequency=50.0,
        ),
        build_row(
            voltage=231.0,
            current=5.1,
            power=1118.0,
            frequency=50.01,
        ),
    ]

    current = build_row(
        voltage=232.0,
        current=5.2,
        power=1140.0,
        frequency=50.02,
    )

    features = extract_value_features(
        current_row=current,
        history=history,
    )

    expected_fields = {
        "voltage_diff",
        "voltage_rolling_mean",
        "voltage_rolling_std",
        "voltage_deviation",
        "voltage_zscore",
        "current_diff",
        "current_rolling_mean",
        "current_rolling_std",
        "current_deviation",
        "current_zscore",
        "power_diff",
        "power_rolling_mean",
        "power_rolling_std",
        "power_deviation",
        "power_zscore",
        "frequency_diff",
        "frequency_rolling_mean",
        "frequency_rolling_std",
        "frequency_deviation",
        "frequency_zscore",
    }

    assert set(features.keys()) == (
        expected_fields
    )


def test_current_row_is_not_modified() -> None:
    current = build_row(
        voltage=230.0
    )

    original = current.copy()

    extract_value_features(
        current_row=current,
        history=[],
    )

    assert current == original


def test_missing_value_field_is_rejected() -> None:
    current = {
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
    }

    with pytest.raises(
        KeyError,
        match="frequency",
    ):
        extract_value_features(
            current_row=current,
            history=[],
        )


def test_invalid_minimum_history_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="minimum_history",
    ):
        extract_value_features(
            current_row=build_row(
                voltage=230.0
            ),
            history=[],
            minimum_history=0,
        )