from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.detection.model_loader import EdgePrediction


class ComparisonModelBundle:
    """Adapter for the persisted binary LR/RF comparison artifacts."""

    def __init__(
        self,
        model: Any,
        scaler: Any,
        feature_columns: tuple[str, ...],
        threshold: float,
        model_name: str,
        artifact_path: str | Path,
    ) -> None:
        if not feature_columns:
            raise ValueError("Comparison feature columns must not be empty")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Comparison threshold must be between zero and one")
        self.model = model
        self.scaler = scaler
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.model_name = model_name
        self.artifact_path = str(artifact_path)

    @classmethod
    def load(
        cls, artifact_path: str | Path, model_name: str,
    ) -> "ComparisonModelBundle":
        path = Path(artifact_path)
        if not path.is_file():
            raise FileNotFoundError(f"Comparison model artifact not found: {path}")
        artifact = joblib.load(path)
        if not isinstance(artifact, dict):
            raise ValueError("Comparison model artifact must contain a dictionary")
        required = {"model", "scaler", "feature_columns", "threshold"}
        missing = required - artifact.keys()
        if missing:
            raise ValueError(f"Comparison artifact missing fields: {sorted(missing)}")
        return cls(
            model=artifact["model"],
            scaler=artifact["scaler"],
            feature_columns=tuple(artifact["feature_columns"]),
            threshold=float(artifact["threshold"]),
            model_name=model_name,
            artifact_path=path,
        )

    def predict(self, features: dict[str, Any]) -> EdgePrediction:
        missing = set(self.feature_columns) - features.keys()
        if missing:
            raise ValueError(
                "Online feature row is missing comparison features: "
                f"{sorted(missing)}"
            )
        values = np.asarray(
            [[float(features[column]) for column in self.feature_columns]],
            dtype=float,
        )
        if not np.isfinite(values).all():
            raise ValueError("Online comparison features must be finite")
        transformed = self.scaler.transform(values) if self.scaler is not None else values
        probabilities = self.model.predict_proba(transformed)
        classes = np.asarray(self.model.classes_)
        attack_matches = np.flatnonzero(classes == 1)
        if len(attack_matches) != 1:
            raise ValueError("Comparison classifier must contain binary attack class 1")
        attack_probability = float(probabilities[0, int(attack_matches[0])])
        decision = "known_attack" if attack_probability >= self.threshold else "none"
        confidence = max(attack_probability, 1.0 - attack_probability)
        return EdgePrediction(
            known_prediction=decision,
            decision=decision,
            confidence=confidence,
            anomaly_score=0.0,
        )
