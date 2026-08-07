from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from src.detection.model_loader import OpenSetModelBundle
from src.drift.monitor import MultiFeatureDriftMonitor
from src.drift.controller import DriftController
from src.features.feature_pipeline import StreamingFeaturePipeline


class EdgeDetector:
    """Stateful online feature extraction and open-set inference."""

    def __init__(
        self,
        model: OpenSetModelBundle,
        feature_pipeline: StreamingFeaturePipeline,
        result_handler: Callable[[dict[str, Any]], None] | None = None,
        drift_monitor: MultiFeatureDriftMonitor | None = None,
        drift_controller: DriftController | None = None,
    ) -> None:
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.result_handler = result_handler
        self.drift_monitor = drift_monitor
        self.drift_controller = drift_controller
        self.processed_messages = 0
        self.failed_messages = 0

    def process(self, row: dict[str, Any]) -> dict[str, Any]:
        started = perf_counter()
        try:
            feature_started = perf_counter()
            features = self.feature_pipeline.transform_one(row)
            feature_ms = (perf_counter() - feature_started) * 1000.0

            inference_started = perf_counter()
            prediction = self.model.predict(features)
            inference_ms = (perf_counter() - inference_started) * 1000.0

            control_result = (
                self.drift_controller.update(
                    device_id=str(row["device_id"]),
                    features=features,
                    prediction=prediction,
                )
                if self.drift_controller is not None
                else None
            )
            drift_events = (
                list(control_result.drift_events)
                if control_result is not None
                else (
                    self.drift_monitor.update(
                        device_id=str(row["device_id"]),
                        features=features,
                    )
                    if self.drift_monitor is not None
                    else []
                )
            )

            result = {
                "receive_timestamp": row["receive_timestamp"],
                "scenario_id": row["scenario_id"],
                "device_id": row["device_id"],
                "sequence_number": row["sequence_number"],
                "true_attack_type": row.get("attack_type", ""),
                "true_drift_type": row.get("drift_type", "none"),
                "known_prediction": prediction.known_prediction,
                "decision": prediction.decision,
                "confidence": prediction.confidence,
                "anomaly_score": prediction.anomaly_score,
                "drift_detected": int(bool(drift_events)),
                "drift_features": ";".join(
                    str(event["feature"])
                    for event in drift_events
                ),
                "adaptation_allowed": int(
                    control_result.adaptation_allowed
                    if control_result is not None
                    else False
                ),
                "adaptation_updated": int(
                    control_result.adaptation_updated
                    if control_result is not None
                    else False
                ),
                "adaptation_reason": (
                    control_result.adaptation_reason
                    if control_result is not None
                    else "disabled"
                ),
                "adaptation_features": (
                    ";".join(control_result.adaptation_features)
                    if control_result is not None
                    else ""
                ),
                "adaptation_references": (
                    ";".join(control_result.reference_values)
                    if control_result is not None
                    else ""
                ),
                "feature_extraction_ms": feature_ms,
                "model_inference_ms": inference_ms,
                "total_detection_ms": (perf_counter() - started) * 1000.0,
            }
            self.processed_messages += 1
            if self.result_handler is not None:
                self.result_handler(result)
            return result
        except Exception:
            self.failed_messages += 1
            raise
