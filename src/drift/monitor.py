from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.drift.page_hinkley import PageHinkley


@dataclass(frozen=True)
class DriftFeatureConfig:
    name: str
    delta: float
    threshold: float
    minimum_instances: int


class MultiFeatureDriftMonitor:
    """Maintain independent Page-Hinkley detectors per device and feature."""

    def __init__(self, feature_configs: list[DriftFeatureConfig]) -> None:
        if not feature_configs:
            raise ValueError("At least one drift feature is required")
        names = [config.name for config in feature_configs]
        if len(names) != len(set(names)):
            raise ValueError("Drift feature names must be unique")
        self.feature_configs = tuple(feature_configs)
        self._detectors: dict[tuple[str, str], PageHinkley] = {}

    @classmethod
    def from_config(
        cls,
        raw_features: list[dict[str, Any]],
    ) -> "MultiFeatureDriftMonitor":
        if not isinstance(raw_features, list):
            raise TypeError("drift features must be a list")
        return cls(
            [
                DriftFeatureConfig(
                    name=str(item["name"]),
                    delta=float(item["delta"]),
                    threshold=float(item["threshold"]),
                    minimum_instances=int(item["minimum_instances"]),
                )
                for item in raw_features
            ]
        )

    def update(
        self,
        device_id: str,
        features: dict[str, Any],
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for config in self.feature_configs:
            if config.name not in features:
                raise ValueError(
                    f"Drift feature missing from online row: {config.name}"
                )
            key = (str(device_id), config.name)
            detector = self._detectors.get(key)
            if detector is None:
                detector = PageHinkley(
                    delta=config.delta,
                    threshold=config.threshold,
                    minimum_instances=config.minimum_instances,
                )
                self._detectors[key] = detector
            result = detector.update(float(features[config.name]))
            if result.drift_detected:
                events.append(
                    {
                        "device_id": str(device_id),
                        "feature": config.name,
                        "direction": result.direction,
                        "statistic": result.statistic,
                    }
                )
        return events
