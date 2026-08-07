from __future__ import annotations

import csv
from pathlib import Path
from threading import Lock
from typing import Any


RESULT_FIELDS = (
    "receive_timestamp",
    "scenario_id",
    "device_id",
    "sequence_number",
    "true_attack_type",
    "known_prediction",
    "decision",
    "confidence",
    "anomaly_score",
    "drift_detected",
    "drift_features",
    "feature_extraction_ms",
    "model_inference_ms",
    "total_detection_ms",
)


class AlertManager:
    """Print decisions and optionally append them to a CSV log."""

    def __init__(
        self,
        output_path: str | Path | None = None,
        print_normal: bool = False,
    ) -> None:
        self.output_path = Path(output_path) if output_path else None
        self.print_normal = print_normal
        self._lock = Lock()
        self._file: Any = None
        self._writer: csv.DictWriter | None = None

        if self.output_path is not None:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            has_content = (
                self.output_path.exists()
                and self.output_path.stat().st_size > 0
            )
            self._file = self.output_path.open(
                "a", newline="", encoding="utf-8"
            )
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=RESULT_FIELDS,
                extrasaction="ignore",
            )
            if not has_content:
                self._writer.writeheader()
                self._file.flush()

    def emit(self, result: dict[str, Any]) -> None:
        if self._writer is not None:
            with self._lock:
                self._writer.writerow(
                    {field: result.get(field, "") for field in RESULT_FIELDS}
                )
                self._file.flush()

        if (
            self.print_normal
            or result["decision"] != "none"
            or result.get("drift_detected", 0)
        ):
            print(
                f"{result['receive_timestamp']} | "
                f"{result['device_id']} | {result['decision'].upper()} | "
                f"confidence={result['confidence']:.4f} | "
                f"anomaly={result['anomaly_score']:.4f} | "
                f"latency={result['total_detection_ms']:.3f} ms | "
                f"drift={result.get('drift_features', '') or 'none'}"
            )

    def close(self) -> None:
        if self._file is not None:
            with self._lock:
                self._file.flush()
                self._file.close()
            self._file = None
            self._writer = None
