from pathlib import Path

import numpy as np
import pandas as pd

from src.detection.edge_detector import EdgeDetector
from src.detection.model_loader import EdgePrediction
from src.features.feature_pipeline import StreamingFeaturePipeline
from src.drift.monitor import MultiFeatureDriftMonitor
from src.drift.controller import DriftControlResult


def build_row(sequence: int = 0) -> dict[str, object]:
    return {
        "receive_timestamp": pd.Timestamp("2026-07-20T10:00:00.05+00:00"),
        "message_timestamp": pd.Timestamp("2026-07-20T10:00:00+00:00"),
        "scenario_id": "normal_test",
        "device_id": "meter_01",
        "client_id": "simulator-normal_test",
        "topic": "grid/substation_01/meter_01/measurement",
        "qos": 0,
        "retain": 0,
        "payload_size": 200,
        "sequence_number": sequence,
        "voltage": 230.0,
        "current": 5.0,
        "power": 1092.5,
        "frequency": 50.0,
        "attack_type": "none",
        "is_attack": 0,
        "attack_step": None,
    }


class FakeModel:
    def predict(self, features: dict[str, object]) -> EdgePrediction:
        assert "voltage_diff" in features
        return EdgePrediction("none", "none", 0.99, 0.2)


class FakeAttackModel:
    def predict(self, features: dict[str, object]) -> EdgePrediction:
        return EdgePrediction("random", "random", 0.99, 0.2)


class FakeApprovedDriftController:
    def update(
        self,
        device_id: str,
        features: dict[str, object],
        prediction: EdgePrediction,
    ) -> DriftControlResult:
        return DriftControlResult(
            drift_events=(),
            adaptation_allowed=True,
            adaptation_updated=False,
            adaptation_reason="collecting_candidates",
            adaptation_features=("source_publish_interval",),
            reference_updates=(),
            reference_values=("source_publish_interval=0.500000",),
            approved_features=("source_publish_interval",),
        )


def test_streaming_pipeline_matches_batch_pipeline() -> None:
    rows = [build_row(0), build_row(1)]
    streaming = StreamingFeaturePipeline()
    online = pd.DataFrame([streaming.transform_one(row) for row in rows])

    from src.features.feature_pipeline import FeaturePipeline
    batch = FeaturePipeline().transform(pd.DataFrame(rows))
    assert online["history_count"].tolist() == batch["history_count"].tolist()
    assert online["voltage_diff"].tolist() == batch["voltage_diff"].tolist()


def test_edge_detector_returns_latency_and_decision() -> None:
    detector = EdgeDetector(
        model=FakeModel(),  # type: ignore[arg-type]
        feature_pipeline=StreamingFeaturePipeline(),
    )
    result = detector.process(build_row())
    assert result["decision"] == "none"
    assert result["raw_decision"] == "none"
    assert result["drift_aware_decision"] == "none"
    assert result["total_detection_ms"] >= 0
    assert detector.processed_messages == 1
    assert detector.failed_messages == 0


def test_edge_detector_reports_drift_events() -> None:
    monitor = MultiFeatureDriftMonitor.from_config(
        [
            {
                "name": "voltage",
                "delta": 0.0,
                "threshold": 1.0,
                "minimum_instances": 2,
            }
        ]
    )
    detector = EdgeDetector(
        model=FakeModel(),  # type: ignore[arg-type]
        feature_pipeline=StreamingFeaturePipeline(),
        drift_monitor=monitor,
    )
    for sequence in range(5):
        detector.process(build_row(sequence))
    shifted = build_row(5)
    shifted["voltage"] = 240.0
    result = detector.process(shifted)
    assert result["drift_detected"] == 1
    assert result["drift_features"] == "voltage"


def test_edge_detector_preserves_raw_decision_when_drift_is_approved() -> None:
    detector = EdgeDetector(
        model=FakeAttackModel(),  # type: ignore[arg-type]
        feature_pipeline=StreamingFeaturePipeline(),
        drift_controller=FakeApprovedDriftController(),  # type: ignore[arg-type]
    )
    result = detector.process(build_row())
    assert result["decision"] == "random"
    assert result["raw_decision"] == "random"
    assert result["drift_aware_decision"] == "normal_drift"
    assert result["approved_drift_features"] == "source_publish_interval"
