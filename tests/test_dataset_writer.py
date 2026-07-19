import csv

from src.collection.dataset_writer import (
    RAW_DATA_FIELDS,
    CsvDatasetWriter,
)


def test_writer_creates_csv_with_header(
    tmp_path,
) -> None:
    output_path = tmp_path / "dataset.csv"

    writer = CsvDatasetWriter(output_path)
    writer.open()

    writer.write(
        {
            "device_id": "meter_01",
            "voltage": 230.0,
            "attack_type": "none",
            "is_attack": 0,
        }
    )

    writer.close()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1

    assert list(rows[0].keys()) == (
        RAW_DATA_FIELDS
    )

    assert rows[0]["device_id"] == (
        "meter_01"
    )

    assert rows[0]["voltage"] == "230.0"
    assert rows[0]["attack_type"] == "none"
    assert rows[0]["is_attack"] == "0"


def test_writer_appends_without_duplicate_header(
    tmp_path,
) -> None:
    output_path = tmp_path / "dataset.csv"

    first_writer = CsvDatasetWriter(output_path)
    first_writer.open()
    first_writer.write(
        {
            "device_id": "meter_01",
        }
    )
    first_writer.close()

    second_writer = CsvDatasetWriter(output_path)
    second_writer.open()
    second_writer.write(
        {
            "device_id": "meter_02",
        }
    )
    second_writer.close()

    lines = output_path.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 3