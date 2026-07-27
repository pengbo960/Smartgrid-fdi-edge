from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


VALUE_FEATURES = (
    "voltage",
    "current",
    "power",
    "frequency",
    "voltage_diff",
    "voltage_percentage_change",
    "voltage_rolling_mean",
    "voltage_rolling_std",
    "voltage_deviation",
    "voltage_zscore",
    "current_diff",
    "current_percentage_change",
    "current_rolling_mean",
    "current_rolling_std",
    "current_deviation",
    "current_zscore",
    "power_diff",
    "power_percentage_change",
    "power_rolling_mean",
    "power_rolling_std",
    "power_deviation",
    "power_zscore",
    "frequency_diff",
    "frequency_percentage_change",
    "frequency_rolling_mean",
    "frequency_rolling_std",
    "frequency_deviation",
    "frequency_zscore",
    "power_consistency_error",
)

TEMPORAL_FEATURES = (
    "source_publish_interval",
    "gateway_inter_arrival_time",
    "transport_delay_estimate",
    "delay_change",
    "sequence_gap",
    "is_duplicate_sequence",
    "is_out_of_order",
    "repeated_value_count",
    "same_value_run_length",
)

PROTOCOL_FEATURES = (
    "payload_size",
    "payload_size_diff",
    "payload_size_rolling_mean",
    "payload_size_deviation",
    "qos",
    "retain",
    "device_topic_match",
    "client_changed",
    "topic_changed",
    "unexpected_client_topic",
)

FEATURE_GROUPS = {
    "value": VALUE_FEATURES,
    "temporal": TEMPORAL_FEATURES,
    "protocol": PROTOCOL_FEATURES,
}


@dataclass(frozen=True)
class PreparedDataset:
    dataframe: pd.DataFrame
    feature_columns: tuple[str, ...]
    target_column: str
    group_column: str


def load_feature_dataset(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a processed multi-view feature dataset.
    """
    dataset_path = Path(path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Feature dataset not found: {dataset_path}"
        )

    if not dataset_path.is_file():
        raise ValueError(
            f"Feature dataset path is not a file: {dataset_path}"
        )

    dataframe = pd.read_csv(dataset_path)

    if dataframe.empty:
        raise ValueError(
            f"Feature dataset is empty: {dataset_path}"
        )

    return dataframe


def select_feature_columns(
    dataframe: pd.DataFrame,
    include_groups: Iterable[str],
) -> tuple[str, ...]:
    """
    Select ordered feature columns from one or more feature groups.
    """
    selected: list[str] = []

    for group_name in include_groups:
        if group_name not in FEATURE_GROUPS:
            raise ValueError(
                f"Unknown feature group: {group_name}"
            )

        for feature_name in FEATURE_GROUPS[group_name]:
            if feature_name not in dataframe.columns:
                raise ValueError(
                    "Feature column missing from dataset: "
                    f"{feature_name}"
                )

            if feature_name not in selected:
                selected.append(feature_name)

    if not selected:
        raise ValueError(
            "At least one feature group must be selected"
        )

    return tuple(selected)


def _find_excluded_source_files(
    dataframe: pd.DataFrame,
    excluded_attack_types: set[str],
    group_column: str,
) -> set[str]:
    """
    Find complete source files containing an excluded attack type.

    For example, when gradual is excluded, every row originating from a
    gradual scenario file is removed, including the normal periods and
    normal messages from other devices in that same file.
    """
    if not excluded_attack_types:
        return set()

    excluded_rows = dataframe[
        dataframe["attack_type"].astype(str).isin(
            excluded_attack_types
        )
    ]

    return set(
        excluded_rows[group_column]
        .astype(str)
        .unique()
    )


def prepare_known_attack_dataset(
    dataframe: pd.DataFrame,
    known_attack_types: Iterable[str],
    excluded_attack_types: Iterable[str] = (),
    target_column: str = "is_attack",
    group_column: str = "source_file",
    include_groups: Iterable[str] = (
        "value",
        "temporal",
    ),
    drop_warmup_rows: bool = False,
    maximum_missing_fraction: float = 0.0,
) -> PreparedDataset:
    """
    Prepare data for the known-attack baseline.

    Entire source files containing an excluded attack type are removed
    before rows are filtered to known attack labels. This prevents
    unseen-attack scenario data from leaking into known-attack training.
    """
    known_types = {
        str(attack_type)
        for attack_type in known_attack_types
    }

    excluded_types = {
        str(attack_type)
        for attack_type in excluded_attack_types
    }

    if not known_types:
        raise ValueError(
            "known_attack_types must not be empty"
        )

    required_columns = {
        "attack_type",
        target_column,
        group_column,
    }

    missing_required = (
        required_columns - set(dataframe.columns)
    )

    if missing_required:
        raise ValueError(
            "Dataset missing required columns: "
            f"{sorted(missing_required)}"
        )

    if not 0 <= maximum_missing_fraction <= 1:
        raise ValueError(
            "maximum_missing_fraction must be between 0 and 1"
        )

    working = dataframe.copy()

    excluded_source_files = _find_excluded_source_files(
        dataframe=working,
        excluded_attack_types=excluded_types,
        group_column=group_column,
    )

    if excluded_source_files:
        working = working[
            ~working[group_column]
            .astype(str)
            .isin(excluded_source_files)
        ].copy()

    filtered = working[
        working["attack_type"]
        .astype(str)
        .isin(known_types)
    ].copy()

    if filtered.empty:
        raise ValueError(
            "No rows remain after filtering known attack types"
        )

    if drop_warmup_rows:
        if "history_count" not in filtered.columns:
            raise ValueError(
                "history_count is required when "
                "drop_warmup_rows is enabled"
            )

        filtered = filtered[
            filtered["history_count"] > 0
        ].copy()

    if filtered.empty:
        raise ValueError(
            "No rows remain after warm-up filtering"
        )

    feature_columns = select_feature_columns(
        dataframe=filtered,
        include_groups=include_groups,
    )

    numeric_features = filtered[
        list(feature_columns)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    missing_fraction = (
        numeric_features.isna().mean()
    )

    excessive_missing = missing_fraction[
        missing_fraction
        > maximum_missing_fraction
    ]

    if not excessive_missing.empty:
        raise ValueError(
            "Feature columns exceed the allowed missing fraction: "
            f"{excessive_missing.to_dict()}"
        )

    numeric_values = numeric_features.to_numpy(
        dtype=float
    )

    if np.isinf(numeric_values).any():
        raise ValueError(
            "Feature dataset contains infinite values"
        )

    target = pd.to_numeric(
        filtered[target_column],
        errors="coerce",
    )

    if target.isna().any():
        raise ValueError(
            "Target column contains invalid values: "
            f"{target_column}"
        )

    target_as_integer = target.astype(int)

    target_values = set(
        target_as_integer.unique()
    )

    if not target_values.issubset({0, 1}):
        raise ValueError(
            "Baseline target must contain only binary labels 0 and 1"
        )

    filtered.loc[:, list(feature_columns)] = (
        numeric_features
    )

    filtered.loc[:, target_column] = (
        target_as_integer
    )

    filtered = filtered.reset_index(
        drop=True
    )

    return PreparedDataset(
        dataframe=filtered,
        feature_columns=feature_columns,
        target_column=target_column,
        group_column=group_column,
    )