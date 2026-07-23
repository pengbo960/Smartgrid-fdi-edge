from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.features.data_loader import load_raw_dataset
from src.features.protocol_features import (
    extract_protocol_features,
)
from src.features.temporal_features import (
    extract_temporal_features,
)
from src.features.value_features import (
    extract_value_features,
)
from src.features.window_manager import WindowManager


METADATA_COLUMNS = (
    "receive_timestamp",
    "message_timestamp",
    "scenario_id",
    "device_id",
    "client_id",
    "topic",
    "sequence_number",
)

RAW_VALUE_COLUMNS = (
    "voltage",
    "current",
    "power",
    "frequency",
)

LABEL_COLUMNS = (
    "attack_type",
    "is_attack",
    "attack_step",
)


class FeaturePipeline:
    """
    Build multi-view features from raw MQTT message records.

    Feature extraction uses only previous messages from the same device.
    The current row is added to the history window after feature
    extraction has completed.
    """

    def __init__(
        self,
        window_size: int = 20,
        minimum_history: int = 2,
        power_factor: float = 0.95,
        repeated_value_field: str = "voltage",
        value_tolerance: float = 1e-6,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero"
            )

        if minimum_history < 1:
            raise ValueError(
                "minimum_history must be at least one"
            )

        if minimum_history > window_size:
            raise ValueError(
                "minimum_history must not exceed window_size"
            )

        self.window_size = window_size
        self.minimum_history = minimum_history
        self.power_factor = power_factor
        self.repeated_value_field = (
            repeated_value_field
        )
        self.value_tolerance = value_tolerance

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform one validated raw dataframe into feature rows.
        """
        if dataframe.empty:
            raise ValueError(
                "Cannot extract features from an empty dataframe"
            )

        manager = WindowManager(
            window_size=self.window_size
        )

        output_rows: list[dict[str, Any]] = []

        for _, row in dataframe.iterrows():
            current_row = row.to_dict()

            device_id = str(
                current_row["device_id"]
            )

            history = manager.get_history(
                device_id
            )

            value_features = extract_value_features(
                current_row=current_row,
                history=history,
                minimum_history=self.minimum_history,
                power_factor=self.power_factor,
            )

            temporal_features = (
                extract_temporal_features(
                    current_row=current_row,
                    history=history,
                    repeated_value_field=(
                        self.repeated_value_field
                    ),
                    value_tolerance=(
                        self.value_tolerance
                    ),
                )
            )

            protocol_features = (
                extract_protocol_features(
                    current_row=current_row,
                    history=history,
                )
            )

            output_row: dict[str, Any] = {}

            for column in METADATA_COLUMNS:
                output_row[column] = (
                    current_row[column]
                )

            for column in RAW_VALUE_COLUMNS:
                output_row[column] = (
                    current_row[column]
                )

            output_row["history_count"] = len(
                history
            )

            output_row.update(
                value_features
            )

            output_row.update(
                temporal_features
            )

            output_row.update(
                protocol_features
            )

            for column in LABEL_COLUMNS:
                output_row[column] = (
                    current_row[column]
                )

            output_rows.append(
                output_row
            )

            manager.update(
                current_row
            )

        return pd.DataFrame(
            output_rows
        )

    def transform_file(
        self,
        input_path: str | Path,
        output_path: str | Path | None = None,
    ) -> pd.DataFrame:
        """
        Load a raw CSV, generate features and optionally save the result.
        """
        dataframe = load_raw_dataset(
            input_path
        )

        feature_dataframe = self.transform(
            dataframe
        )

        if output_path is not None:
            output_file = Path(
                output_path
            )

            output_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            feature_dataframe.to_csv(
                output_file,
                index=False,
            )

        return feature_dataframe