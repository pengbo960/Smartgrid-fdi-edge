from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from src.training.prepare_dataset import (
    PreparedDataset,
)


KNOWN_SCENARIO_TYPES = (
    "normal",
    "constant",
    "random",
    "gradual",
)


@dataclass(frozen=True)
class DatasetSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def infer_scenario_type(
    source_file: str,
) -> str:
    """
    Infer a scenario type from its source filename.

    Expected examples:
        normal_run_01.csv
        constant_run_02.csv
        random_run_03.csv
        gradual_run_04.csv
    """
    filename = str(source_file).strip().lower()

    for scenario_type in KNOWN_SCENARIO_TYPES:
        if filename.startswith(
            f"{scenario_type}_"
        ):
            return scenario_type

    raise ValueError(
        "Cannot infer scenario type from source file: "
        f"{source_file}"
    )


def _build_group_table(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    group_table = (
        dataframe[[group_column]]
        .drop_duplicates()
        .copy()
    )

    group_table["scenario_type"] = (
        group_table[group_column]
        .astype(str)
        .map(infer_scenario_type)
    )

    return group_table


def _split_scenario_groups(
    groups: list[str],
    scenario_type: str,
    random_seed: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Split one scenario's source files.

    Three runs:
        1 train, 1 validation, 1 test

    Four runs:
        2 train, 1 validation, 1 test

    Five runs:
        3 train, 1 validation, 1 test
    """
    if len(groups) < 3:
        raise ValueError(
            f"Scenario type {scenario_type} "
            "requires at least three runs; "
            f"found {len(groups)}"
        )

    shuffled = sorted(groups)

    scenario_seed = (
        random_seed
        + sum(
            ord(character)
            for character in scenario_type
        )
    )

    rng = random.Random(
        scenario_seed
    )

    rng.shuffle(
        shuffled
    )

    test_group = shuffled.pop()
    validation_group = shuffled.pop()

    train_groups = shuffled

    return (
        train_groups,
        [validation_group],
        [test_group],
    )


def _validate_group_separation(
    train_groups: set[str],
    validation_groups: set[str],
    test_groups: set[str],
) -> None:
    if train_groups & validation_groups:
        raise RuntimeError(
            "Train and validation groups overlap"
        )

    if train_groups & test_groups:
        raise RuntimeError(
            "Train and test groups overlap"
        )

    if validation_groups & test_groups:
        raise RuntimeError(
            "Validation and test groups overlap"
        )


def _validate_split_labels(
    frame: pd.DataFrame,
    split_name: str,
    required_attack_types: set[str],
) -> None:
    actual_attack_types = set(
        frame["attack_type"]
        .astype(str)
        .unique()
    )

    missing_attack_types = (
        required_attack_types
        - actual_attack_types
    )

    if missing_attack_types:
        raise ValueError(
            f"{split_name} split is missing attack types: "
            f"{sorted(missing_attack_types)}"
        )

    binary_labels = set(
        pd.to_numeric(
            frame["is_attack"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )

    if binary_labels != {0, 1}:
        raise ValueError(
            f"{split_name} split must contain "
            "both normal and attack labels"
        )


def split_stratified_grouped_dataset(
    prepared: PreparedDataset,
    random_seed: int = 42,
) -> DatasetSplit:
    """
    Split data by source file while preserving each scenario type.

    For every scenario type:
        all but two runs -> training
        one run -> validation
        one run -> testing
    """
    dataframe = prepared.dataframe
    group_column = prepared.group_column

    required_columns = {
        group_column,
        "attack_type",
        "is_attack",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset missing split columns: "
            f"{sorted(missing_columns)}"
        )

    group_table = _build_group_table(
        dataframe=dataframe,
        group_column=group_column,
    )

    train_groups: list[str] = []
    validation_groups: list[str] = []
    test_groups: list[str] = []

    for scenario_type, scenario_frame in (
        group_table.groupby(
            "scenario_type",
            sort=True,
        )
    ):
        scenario_groups = (
            scenario_frame[group_column]
            .astype(str)
            .tolist()
        )

        (
            scenario_train,
            scenario_validation,
            scenario_test,
        ) = _split_scenario_groups(
            groups=scenario_groups,
            scenario_type=str(
                scenario_type
            ),
            random_seed=random_seed,
        )

        train_groups.extend(
            scenario_train
        )

        validation_groups.extend(
            scenario_validation
        )

        test_groups.extend(
            scenario_test
        )

    train_group_set = set(
        train_groups
    )

    validation_group_set = set(
        validation_groups
    )

    test_group_set = set(
        test_groups
    )

    _validate_group_separation(
        train_groups=train_group_set,
        validation_groups=validation_group_set,
        test_groups=test_group_set,
    )

    train = dataframe[
        dataframe[group_column]
        .astype(str)
        .isin(train_group_set)
    ].copy()

    validation = dataframe[
        dataframe[group_column]
        .astype(str)
        .isin(validation_group_set)
    ].copy()

    test = dataframe[
        dataframe[group_column]
        .astype(str)
        .isin(test_group_set)
    ].copy()

    if train.empty:
        raise ValueError(
            "Training split is empty"
        )

    if validation.empty:
        raise ValueError(
            "Validation split is empty"
        )

    if test.empty:
        raise ValueError(
            "Test split is empty"
        )

    required_attack_types = {
        attack_type
        for attack_type in dataframe[
            "attack_type"
        ].astype(str).unique()
    }

    _validate_split_labels(
        frame=train,
        split_name="Training",
        required_attack_types=required_attack_types,
    )

    _validate_split_labels(
        frame=validation,
        split_name="Validation",
        required_attack_types=required_attack_types,
    )

    _validate_split_labels(
        frame=test,
        split_name="Test",
        required_attack_types=required_attack_types,
    )

    return DatasetSplit(
        train=train.reset_index(
            drop=True
        ),
        validation=validation.reset_index(
            drop=True
        ),
        test=test.reset_index(
            drop=True
        ),
        train_groups=tuple(
            sorted(train_group_set)
        ),
        validation_groups=tuple(
            sorted(validation_group_set)
        ),
        test_groups=tuple(
            sorted(test_group_set)
        ),
    )