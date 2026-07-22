from __future__ import annotations

import csv
import threading
from pathlib import Path
from typing import Any


RAW_DATA_FIELDS = [
    "receive_timestamp",
    "message_timestamp",
    "scenario_id",
    "device_id",
    "client_id",
    "topic",
    "qos",
    "retain",
    "payload_size",
    "sequence_number",
    "voltage",
    "current",
    "power",
    "frequency",
    "attack_type",
    "is_attack",
    "attack_step",
]


class CsvDatasetWriter:
    """
    Append MQTT measurements to a CSV file using a fixed schema.
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = threading.Lock()
        self._file = None
        self._writer = None

    def open(self) -> None:
        """
        Open the CSV file and write the header if the file is empty.
        """
        file_exists = self.output_path.exists()
        file_has_content = (
            file_exists
            and self.output_path.stat().st_size > 0
        )

        self._file = self.output_path.open(
            mode="a",
            newline="",
            encoding="utf-8",
        )

        self._writer = csv.DictWriter(
            self._file,
            fieldnames=RAW_DATA_FIELDS,
            extrasaction="ignore",
        )

        if not file_has_content:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        """
        Write one row to the CSV file.
        """
        if self._file is None or self._writer is None:
            raise RuntimeError(
                "CSV writer is not open"
            )

        completed_row = {
            field: row.get(field, "")
            for field in RAW_DATA_FIELDS
        }

        with self._lock:
            self._writer.writerow(completed_row)
            self._file.flush()

    def close(self) -> None:
        """
        Flush and close the CSV file.
        """
        if self._file is not None:
            with self._lock:
                self._file.flush()
                self._file.close()

        self._file = None
        self._writer = None

    def __enter__(self) -> "CsvDatasetWriter":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()