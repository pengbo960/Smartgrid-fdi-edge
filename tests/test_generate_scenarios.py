import pytest

from scripts.generate_scenarios import (
    ATTACK_SETTINGS,
    build_scenario,
)


def test_all_required_scenario_types_are_supported() -> None:
    assert set(ATTACK_SETTINGS) == {
        "normal",
        "constant",
        "random",
        "gradual",
        "replay",
        "topic_spoof",
    }


@pytest.mark.parametrize(
    "scenario_type",
    tuple(ATTACK_SETTINGS),
)
def test_generated_scenario_has_reproducible_identity(
    scenario_type: str,
) -> None:
    config = build_scenario(
        scenario_type=scenario_type,
        run_number=3,
        duration=300,
        interval=0.5,
    )

    assert config["scenario"]["scenario_id"] == (
        f"{scenario_type}_run_03"
    )
    assert config["scenario"]["random_seed"] > 0


def test_normal_scenario_has_no_attack() -> None:
    config = build_scenario(
        scenario_type="normal",
        run_number=1,
        duration=300,
        interval=0.5,
    )

    assert config["attacks"] == []


def test_replay_scenario_contains_lag_configuration() -> None:
    config = build_scenario(
        scenario_type="replay",
        run_number=1,
        duration=300,
        interval=0.5,
    )

    attack = config["attacks"][0]

    assert attack["device_id"] == "meter_02"
    assert attack["attack_type"] == "replay"
    assert attack["replay_lag_steps"] == 10
    assert (
        attack["start_step"]
        >= attack["replay_lag_steps"]
    )


def test_topic_spoof_scenario_contains_target() -> None:
    config = build_scenario(
        scenario_type="topic_spoof",
        run_number=1,
        duration=300,
        interval=0.5,
    )

    attack = config["attacks"][0]

    assert attack["device_id"] == "meter_02"
    assert attack["attack_type"] == "topic_spoof"
    assert attack["spoofed_device_id"] == "meter_01"


def test_scenario_seeds_are_distinct() -> None:
    seeds = {
        build_scenario(
            scenario_type=scenario_type,
            run_number=1,
            duration=300,
            interval=0.5,
        )["scenario"]["random_seed"]
        for scenario_type in ATTACK_SETTINGS
    }

    assert len(seeds) == len(ATTACK_SETTINGS)


def test_replay_rejects_insufficient_history() -> None:
    with pytest.raises(
        ValueError,
        match="enough history",
    ):
        build_scenario(
            scenario_type="replay",
            run_number=1,
            duration=4,
            interval=0.2,
        )


def test_unknown_scenario_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Unsupported scenario type",
    ):
        build_scenario(
            scenario_type="unknown",
            run_number=1,
            duration=300,
            interval=0.5,
        )
