from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.features.dataset_builder import (
    FeatureDatasetBuilder,
)
from src.features.feature_pipeline import (
    FeaturePipeline,
)


def load_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Config must contain a YAML mapping"
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one combined multi-view feature dataset "
            "from multiple raw MQTT CSV files."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing raw scenario CSV files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output path for the combined feature CSV.",
    )

    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Input filename pattern.",
    )

    parser.add_argument(
        "--config",
        default="config/features.yaml",
        help="Feature configuration file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = load_config(args.config)

    window_config = config.get(
        "window",
        {},
    )

    value_config = config.get(
        "value",
        {},
    )

    temporal_config = config.get(
        "temporal",
        {},
    )

    pipeline = FeaturePipeline(
        window_size=int(
            window_config.get(
                "size",
                20,
            )
        ),
        minimum_history=int(
            window_config.get(
                "minimum_history",
                2,
            )
        ),
        power_factor=float(
            value_config.get(
                "power_factor",
                0.95,
            )
        ),
        repeated_value_field=str(
            temporal_config.get(
                "repeated_value_field",
                "voltage",
            )
        ),
        value_tolerance=float(
            temporal_config.get(
                "value_tolerance",
                1e-6,
            )
        ),
    )

    builder = FeatureDatasetBuilder(
        pipeline=pipeline
    )

    dataframe = builder.build_from_directory(
        input_directory=args.input_dir,
        pattern=args.pattern,
    )

    report = builder.validate_output(
        dataframe
    )

    builder.save_dataset(
        dataframe=dataframe,
        output_path=args.output,
    )

    print(
        f"Rows: {report['row_count']}"
    )

    print(
        f"Columns: {report['column_count']}"
    )

    print(
        f"Duplicate rows: "
        f"{report['duplicate_rows']}"
    )

    print(
        f"Missing values: "
        f"{report['missing_by_column']}"
    )

    print(
        f"Infinite values: "
        f"{report['infinite_by_column']}"
    )

    print("\nAttack distribution:")

    for label, count in (
        report[
            "attack_distribution"
        ].items()
    ):
        print(
            f"  {label}: {count}"
        )

    print("\nSource files:")

    for filename, count in (
        report[
            "source_file_distribution"
        ].items()
    ):
        print(
            f"  {filename}: {count}"
        )

    print(
        f"\nSaved to: {args.output}"
    )


if __name__ == "__main__":
    main()