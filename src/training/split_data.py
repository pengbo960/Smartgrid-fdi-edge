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
    "replay",
    "topic_spoof",
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
        replay_run_05.csv
        topic_spoof_run_05.csv
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


def _assemble_dataset_split(
    prepared: PreparedDataset,
    train_groups: list[str],
    validation_groups: list[str],
    test_groups: list[str],
) -> DatasetSplit:
    """Build and validate dataframes from three disjoint group lists."""
    dataframe = prepared.dataframe
    group_column = prepared.group_column
    train_group_set = set(train_groups)
    validation_group_set = set(validation_groups)
    test_group_set = set(test_groups)

    _validate_group_separation(
        train_groups=train_group_set,
        validation_groups=validation_group_set,
        test_groups=test_group_set,
    )

    train = dataframe[
        dataframe[group_column].astype(str).isin(train_group_set)
    ].copy()
    validation = dataframe[
        dataframe[group_column].astype(str).isin(validation_group_set)
    ].copy()
    test = dataframe[
        dataframe[group_column].astype(str).isin(test_group_set)
    ].copy()

    for split_name, frame in (
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ):
        if frame.empty:
            raise ValueError(f"{split_name} split is empty")

    required_attack_types = set(dataframe["attack_type"].astype(str).unique())
    for split_name, frame in (
        ("Training", train),
        ("Validation", validation),
        ("Test", test),
    ):
        _validate_split_labels(
            frame=frame,
            split_name=split_name,
            required_attack_types=required_attack_types,
        )

    return DatasetSplit(
        train=train.reset_index(drop=True),
        validation=validation.reset_index(drop=True),
        test=test.reset_index(drop=True),
        train_groups=tuple(sorted(train_group_set)),
        validation_groups=tuple(sorted(validation_group_set)),
        test_groups=tuple(sorted(test_group_set)),
    )


def _validate_required_columns(prepared: PreparedDataset) -> None:
    required_columns = {
        prepared.group_column,
        "attack_type",
        "is_attack",
    }
    missing_columns = required_columns - set(prepared.dataframe.columns)
    if missing_columns:
        raise ValueError(
            "Dataset missing split columns: "
            f"{sorted(missing_columns)}"
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
    _validate_required_columns(prepared)
    dataframe = prepared.dataframe
    group_column = prepared.group_column

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

    return _assemble_dataset_split(
        prepared=prepared,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )


def split_stratified_grouped_fold_dataset(
    prepared: PreparedDataset,
    fold_index: int,
    validation_offset: int = 1,
) -> DatasetSplit:
    """Create one deterministic grouped fold with a rotating validation run.

    Source files are sorted inside each scenario family. ``fold_index`` selects
    one test file per family, while ``validation_offset`` selects a different
    file relative to the test file. Across all folds, every source file appears
    exactly once in the test split and exactly once in the validation split.
    """
    _validate_required_columns(prepared)
    if isinstance(fold_index, bool) or not isinstance(fold_index, int):
        raise TypeError("fold_index must be an integer")
    if isinstance(validation_offset, bool) or not isinstance(
        validation_offset, int
    ):
        raise TypeError("validation_offset must be an integer")

    group_table = _build_group_table(
        dataframe=prepared.dataframe,
        group_column=prepared.group_column,
    )
    group_counts = group_table.groupby("scenario_type").size()
    if group_counts.empty:
        raise ValueError("No source-file groups were found")
    if group_counts.nunique() != 1:
        raise ValueError(
            "Grouped folds require the same number of runs for every "
            f"scenario type; found {group_counts.to_dict()}"
        )

    fold_count = int(group_counts.iloc[0])
    if fold_count < 3:
        raise ValueError(
            "Grouped folds require at least three runs per scenario type"
        )
    if not 0 <= fold_index < fold_count:
        raise ValueError(
            f"fold_index must be between 0 and {fold_count - 1}; "
            f"received {fold_index}"
        )
    if validation_offset % fold_count == 0:
        raise ValueError(
            "validation_offset must select a different run from the test run"
        )

    train_groups: list[str] = []
    validation_groups: list[str] = []
    test_groups: list[str] = []
    validation_index = (fold_index + validation_offset) % fold_count

    for _, scenario_frame in group_table.groupby("scenario_type", sort=True):
        scenario_groups = sorted(
            scenario_frame[prepared.group_column].astype(str).tolist()
        )
        test_groups.append(scenario_groups[fold_index])
        validation_groups.append(scenario_groups[validation_index])
        train_groups.extend(
            group
            for index, group in enumerate(scenario_groups)
            if index not in {fold_index, validation_index}
        )

    return _assemble_dataset_split(
        prepared=prepared,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )


def split_grouped_dataset(
    prepared: PreparedDataset,
    strategy: str = "grouped_holdout",
    random_seed: int = 42,
    fold_index: int | None = None,
    validation_offset: int = 1,
) -> DatasetSplit:
    """Dispatch to the configured grouped splitting strategy."""
    normalised_strategy = str(strategy).strip().lower()
    if normalised_strategy in {"grouped_holdout", "repeated_holdout"}:
        return split_stratified_grouped_dataset(
            prepared=prepared,
            random_seed=random_seed,
        )
    if normalised_strategy in {"grouped_kfold", "grouped_fold"}:
        if fold_index is None:
            raise ValueError("fold_index is required for grouped_kfold")
        return split_stratified_grouped_fold_dataset(
            prepared=prepared,
            fold_index=fold_index,
            validation_offset=validation_offset,
        )
    raise ValueError(f"Unsupported grouped split strategy: {strategy}")
