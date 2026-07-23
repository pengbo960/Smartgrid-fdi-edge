from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.features.feature_pipeline import (
    FeaturePipeline,
)


def load_feature_config(
    path: str | Path,
) -> dict[str, Any]:
    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Feature config not found: {config_path}"
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Feature config must contain a YAML mapping"
        )

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build multi-view features from a raw MQTT CSV."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the raw CSV dataset.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for the processed feature CSV.",
    )

    parser.add_argument(
        "--config",
        default="config/features.yaml",
        help="Path to the feature configuration YAML.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    config = load_feature_config(
        args.config
    )

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

    feature_dataframe = pipeline.transform_file(
        input_path=args.input,
        output_path=args.output,
    )

    print(
        f"Processed rows: {len(feature_dataframe)}"
    )

    print(
        f"Processed columns: "
        f"{len(feature_dataframe.columns)}"
    )

    print(
        f"Output saved to: {args.output}"
    )

    print("\nAttack distribution:")

    print(
        feature_dataframe[
            "attack_type"
        ].value_counts(
            dropna=False
        )
    )


if __name__ == "__main__":
    main()