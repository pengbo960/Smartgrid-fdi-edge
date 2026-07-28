from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "receive_timestamp",
    "message_timestamp",
    "scenario_id",
    "device_id",
    "sequence_number",
    "voltage",
    "current",
    "power",
    "frequency",
    "attack_type",
    "is_attack",
}


def validate_file(path: Path) -> bool:
    dataframe = pd.read_csv(path)

    problems: list[str] = []

    missing_columns = (
        REQUIRED_COLUMNS
        - set(dataframe.columns)
    )

    if missing_columns:
        problems.append(
            f"missing columns: "
            f"{sorted(missing_columns)}"
        )

    if dataframe.empty:
        problems.append("empty dataset")

    if not missing_columns:
        duplicate_mask = dataframe.duplicated(
            subset=[
                "scenario_id",
                "device_id",
                "sequence_number",
            ]
        )

        replay_duplicate_mask = (
            duplicate_mask
            & dataframe["attack_type"]
            .astype(str)
            .eq("replay")
        )

        duplicate_count = int(
            (
                duplicate_mask
                & ~replay_duplicate_mask
            ).sum()
        )

        if duplicate_count:
            problems.append(
                f"{duplicate_count} unexpected "
                "duplicate rows"
            )

        device_counts = (
            dataframe["device_id"]
            .value_counts()
        )

        if len(device_counts) != 3:
            problems.append(
                "expected three devices, found "
                f"{len(device_counts)}"
            )

        if dataframe[
            "message_timestamp"
        ].isna().any():
            problems.append(
                "missing message timestamps"
            )

    if problems:
        print(
            f"FAIL {path.name}: "
            + "; ".join(problems)
        )
        return False

    print(
        f"PASS {path.name}: "
        f"{len(dataframe)} rows"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    files = sorted(
        args.input_dir.glob("*.csv")
    )

    if not files:
        raise ValueError(
            f"No CSV files found in "
            f"{args.input_dir}"
        )

    results = [
        validate_file(path)
        for path in files
    ]

    failed = len(results) - sum(results)

    print(
        f"\nValidated {len(results)} files; "
        f"failed: {failed}"
    )

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
