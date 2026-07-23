from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.features.feature_pipeline import FeaturePipeline


class FeatureDatasetBuilder:
    """
    Build and combine feature datasets from multiple raw scenario CSVs.
    """

    def __init__(
        self,
        pipeline: FeaturePipeline,
    ) -> None:
        self.pipeline = pipeline

    def discover_csv_files(
        self,
        input_directory: str | Path,
        pattern: str = "*.csv",
    ) -> list[Path]:
        directory = Path(input_directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Input directory not found: {directory}"
            )

        if not directory.is_dir():
            raise ValueError(
                f"Input path is not a directory: {directory}"
            )

        files = sorted(
            path
            for path in directory.glob(pattern)
            if path.is_file()
        )

        if not files:
            raise ValueError(
                f"No CSV files found in: {directory}"
            )

        return files

    def build_from_files(
        self,
        input_files: Iterable[str | Path],
    ) -> pd.DataFrame:
        feature_frames: list[pd.DataFrame] = []

        for input_file in input_files:
            path = Path(input_file)

            feature_frame = (
                self.pipeline.transform_file(path)
            )

            feature_frame = feature_frame.copy()

            feature_frame["source_file"] = path.name

            feature_frames.append(
                feature_frame
            )

        if not feature_frames:
            raise ValueError(
                "No input files were provided"
            )

        combined = pd.concat(
            feature_frames,
            ignore_index=True,
        )

        return combined

    def build_from_directory(
        self,
        input_directory: str | Path,
        pattern: str = "*.csv",
    ) -> pd.DataFrame:
        files = self.discover_csv_files(
            input_directory=input_directory,
            pattern=pattern,
        )

        return self.build_from_files(files)

    @staticmethod
    def validate_output(
        dataframe: pd.DataFrame,
    ) -> dict[str, object]:
        if dataframe.empty:
            raise ValueError(
                "Feature dataset is empty"
            )

        if "is_attack" not in dataframe.columns:
            raise KeyError(
                "Feature dataset missing is_attack"
            )

        if "attack_type" not in dataframe.columns:
            raise KeyError(
                "Feature dataset missing attack_type"
            )

        duplicate_rows = int(
            dataframe.duplicated(
                subset=[
                    "source_file",
                    "device_id",
                    "sequence_number",
                ]
            ).sum()
        )

        missing_by_column = {
            column: int(count)
            for column, count in (
                dataframe.isna().sum().items()
            )
            if count > 0
        }

        infinite_by_column: dict[str, int] = {}

        numeric_dataframe = dataframe.select_dtypes(
            include="number"
        )

        for column in numeric_dataframe.columns:
            values = numeric_dataframe[column]

            infinite_count = int(
                (
                    values == float("inf")
                ).sum()
                + (
                    values == float("-inf")
                ).sum()
            )

            if infinite_count > 0:
                infinite_by_column[column] = (
                    infinite_count
                )

        report = {
            "row_count": len(dataframe),
            "column_count": len(
                dataframe.columns
            ),
            "duplicate_rows": duplicate_rows,
            "missing_by_column": (
                missing_by_column
            ),
            "infinite_by_column": (
                infinite_by_column
            ),
            "attack_distribution": (
                dataframe[
                    "attack_type"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            ),
            "binary_label_distribution": (
                dataframe[
                    "is_attack"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            ),
            "device_distribution": (
                dataframe[
                    "device_id"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            ),
            "source_file_distribution": (
                dataframe[
                    "source_file"
                ]
                .value_counts(
                    dropna=False
                )
                .to_dict()
            ),
        }

        return report

    @staticmethod
    def save_dataset(
        dataframe: pd.DataFrame,
        output_path: str | Path,
    ) -> None:
        path = Path(output_path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe.to_csv(
            path,
            index=False,
        )