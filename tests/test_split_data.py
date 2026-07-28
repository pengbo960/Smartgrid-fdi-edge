import pandas as pd
import pytest

from src.training.prepare_dataset import (
    PreparedDataset,
)
from src.training.split_data import (
    infer_scenario_type,
    split_stratified_grouped_dataset,
)


def build_prepared_dataset(
    runs_per_scenario: int = 5,
) -> PreparedDataset:
    rows: list[dict[str, object]] = []

    for scenario_type in [
        "normal",
        "constant",
        "random",
    ]:
        for run_number in range(
            1,
            runs_per_scenario + 1,
        ):
            source_file = (
                f"{scenario_type}_run_"
                f"{run_number:02d}.csv"
            )

            for row_number in range(4):
                is_attack = int(
                    scenario_type != "normal"
                    and row_number >= 2
                )

                attack_type = (
                    scenario_type
                    if is_attack
                    else "none"
                )

                rows.append(
                    {
                        "source_file": source_file,
                        "feature_a": float(
                            row_number
                        ),
                        "feature_b": float(
                            run_number
                        ),
                        "is_attack": is_attack,
                        "attack_type": attack_type,
                    }
                )

    dataframe = pd.DataFrame(
        rows
    )

    return PreparedDataset(
        dataframe=dataframe,
        feature_columns=(
            "feature_a",
            "feature_b",
        ),
        target_column="is_attack",
        group_column="source_file",
    )


def test_infer_scenario_type() -> None:
    assert (
        infer_scenario_type(
            "normal_run_01.csv"
        )
        == "normal"
    )

    assert (
        infer_scenario_type(
            "constant_run_02.csv"
        )
        == "constant"
    )

    assert (
        infer_scenario_type(
            "random_run_03.csv"
        )
        == "random"
    )

    assert (
        infer_scenario_type(
            "replay_run_04.csv"
        )
        == "replay"
    )

    assert (
        infer_scenario_type(
            "topic_spoof_run_05.csv"
        )
        == "topic_spoof"
    )


def test_unknown_scenario_filename_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="Cannot infer scenario type",
    ):
        infer_scenario_type(
            "unknown_run_01.csv"
        )


def test_grouped_split_has_no_group_overlap() -> None:
    split = split_stratified_grouped_dataset(
        prepared=build_prepared_dataset(),
        random_seed=42,
    )

    train_groups = set(
        split.train_groups
    )

    validation_groups = set(
        split.validation_groups
    )

    test_groups = set(
        split.test_groups
    )

    assert not (
        train_groups
        & validation_groups
    )

    assert not (
        train_groups
        & test_groups
    )

    assert not (
        validation_groups
        & test_groups
    )


def test_all_rows_are_preserved() -> None:
    prepared = build_prepared_dataset()

    split = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=42,
    )

    total_rows = (
        len(split.train)
        + len(split.validation)
        + len(split.test)
    )

    assert total_rows == len(
        prepared.dataframe
    )


def test_split_is_reproducible() -> None:
    prepared = build_prepared_dataset()

    first = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=42,
    )

    second = split_stratified_grouped_dataset(
        prepared=prepared,
        random_seed=42,
    )

    assert (
        first.train_groups
        == second.train_groups
    )

    assert (
        first.validation_groups
        == second.validation_groups
    )

    assert (
        first.test_groups
        == second.test_groups
    )


def test_five_runs_produce_three_one_one_split() -> None:
    split = split_stratified_grouped_dataset(
        prepared=build_prepared_dataset(
            runs_per_scenario=5
        ),
        random_seed=42,
    )

    assert len(
        split.train_groups
    ) == 9

    assert len(
        split.validation_groups
    ) == 3

    assert len(
        split.test_groups
    ) == 3


def test_each_split_contains_every_scenario_type() -> None:
    split = split_stratified_grouped_dataset(
        prepared=build_prepared_dataset(),
        random_seed=42,
    )

    for groups in [
        split.train_groups,
        split.validation_groups,
        split.test_groups,
    ]:
        scenario_types = {
            infer_scenario_type(
                group
            )
            for group in groups
        }

        assert scenario_types == {
            "normal",
            "constant",
            "random",
        }


def test_each_split_contains_normal_and_attack_labels() -> None:
    split = split_stratified_grouped_dataset(
        prepared=build_prepared_dataset(),
        random_seed=42,
    )

    for frame in [
        split.train,
        split.validation,
        split.test,
    ]:
        assert set(
            frame["is_attack"]
        ) == {
            0,
            1,
        }

        assert set(
            frame["attack_type"]
        ) == {
            "none",
            "constant",
            "random",
        }


def test_too_few_runs_per_scenario_are_rejected() -> None:
    prepared = build_prepared_dataset(
        runs_per_scenario=2
    )

    with pytest.raises(
        ValueError,
        match="requires at least three runs",
    ):
        split_stratified_grouped_dataset(
            prepared=prepared,
            random_seed=42,
        )
