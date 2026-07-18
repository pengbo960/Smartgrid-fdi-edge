from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(relative_path: str) -> dict[str, Any]:
    """
    Load a YAML configuration file relative to the project root.
    """
    config_path = PROJECT_ROOT / relative_path

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(
            f"Configuration file is empty: {config_path}"
        )

    if not isinstance(config, dict):
        raise ValueError(
            f"Configuration file must contain a YAML mapping: {config_path}"
        )

    return config