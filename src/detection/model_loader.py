from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.detection.open_set import apply_open_set_decision


@dataclass(frozen=True)
class EdgePrediction:
    known_prediction: str
    decision: str
    confidence: float
    anomaly_score: float


class OpenSetModelBundle:
    """Load and execute the persisted stage-6 model artifacts."""

    def __init__(
        self,
        classifier: Any,
        scaler: Any,
        anomaly_detector: Any,
        metadata: dict[str, Any],
    ) -> None:
        self.classifier = classifier
        self.scaler = scaler
        self.anomaly_detector = anomaly_detector
        self.feature_columns = tuple(metadata["feature_columns"])
        self.anomaly_feature_columns = tuple(
            metadata["anomaly_feature_columns"]
        )
        self.anomaly_indices = tuple(
            self.feature_columns.index(column)
            for column in self.anomaly_feature_columns
        )
        self.confidence_threshold = float(
            metadata["confidence_threshold"]
        )
        self.anomaly_threshold = float(
            metadata["anomaly_threshold"]
        )
        self.normal_label = str(metadata.get("normal_label", "none"))
        self.unknown_label = str(metadata.get("unknown_label", "unknown"))

    @classmethod
    def load(
        cls,
        classifier_path: str | Path,
        scaler_path: str | Path,
        anomaly_detector_path: str | Path,
        metadata_path: str | Path,
    ) -> "OpenSetModelBundle":
        paths = [
            Path(classifier_path),
            Path(scaler_path),
            Path(anomaly_detector_path),
            Path(metadata_path),
        ]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing open-set model artifacts: " + ", ".join(missing)
            )

        with paths[3].open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        return cls(
            classifier=joblib.load(paths[0]),
            scaler=joblib.load(paths[1]),
            anomaly_detector=joblib.load(paths[2]),
            metadata=metadata,
        )

    def predict(self, features: dict[str, Any]) -> EdgePrediction:
        missing = set(self.feature_columns) - features.keys()
        if missing:
            raise ValueError(
                "Online feature row is missing model features: "
                f"{sorted(missing)}"
            )

        values = np.asarray(
            [[float(features[column]) for column in self.feature_columns]],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("Online model features must be finite")

        scaled = self.scaler.transform(values)
        probabilities = self.classifier.predict_proba(scaled)
        class_index = int(probabilities[0].argmax())
        known_prediction = str(self.classifier.classes_[class_index])
        confidence = float(probabilities[0, class_index])
        anomaly_score = float(
            -self.anomaly_detector.score_samples(
                scaled[:, self.anomaly_indices]
            )[0]
        )
        decision = str(
            apply_open_set_decision(
                predicted_labels=[known_prediction],
                confidence_scores=[confidence],
                anomaly_scores=[anomaly_score],
                confidence_threshold=self.confidence_threshold,
                anomaly_threshold=self.anomaly_threshold,
                normal_label=self.normal_label,
                unknown_label=self.unknown_label,
            )[0]
        )
        return EdgePrediction(
            known_prediction=known_prediction,
            decision=decision,
            confidence=confidence,
            anomaly_score=anomaly_score,
        )
