from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

from src.detection.model_loader import EdgePrediction
from src.drift.guarded_adaptation import ReferenceAdapter
from src.drift.monitor import MultiFeatureDriftMonitor


@dataclass(frozen=True)
class DriftControlResult:
    drift_events: tuple[dict[str, Any], ...]
    adaptation_allowed: bool
    adaptation_updated: bool
    adaptation_reason: str
    adaptation_features: tuple[str, ...]
    reference_updates: tuple[str, ...]
    reference_values: tuple[str, ...]
    approved_features: tuple[str, ...]


class DriftController:
    """Coordinate drift alerts, approval, trust checks and adaptation."""

    def __init__(
        self,
        monitor: MultiFeatureDriftMonitor,
        adaptive_features: tuple[str, ...],
        calibration_samples: int = 50,
        trusted_confidence: float = 0.98,
        maximum_trusted_anomaly: float = 0.75,
        minimum_history: int = 20,
        auto_approve: bool = False,
        approval_ttl_messages: int = 300,
        allow_approved_prediction_override: bool = False,
        adapter_config: dict[str, Any] | None = None,
    ) -> None:
        if not adaptive_features:
            raise ValueError("adaptive_features must not be empty")
        if calibration_samples <= 0:
            raise ValueError("calibration_samples must be greater than zero")
        if not 0 <= trusted_confidence <= 1:
            raise ValueError("trusted_confidence must be between 0 and 1")
        if minimum_history < 0:
            raise ValueError("minimum_history must be non-negative")
        if approval_ttl_messages <= 0:
            raise ValueError("approval_ttl_messages must be greater than zero")

        self.monitor = monitor
        self.adaptive_features = adaptive_features
        self.calibration_samples = calibration_samples
        self.trusted_confidence = trusted_confidence
        self.maximum_trusted_anomaly = maximum_trusted_anomaly
        self.minimum_history = minimum_history
        self.auto_approve = auto_approve
        self.approval_ttl_messages = approval_ttl_messages
        self.allow_approved_prediction_override = (
            allow_approved_prediction_override
        )
        self.adapter_config = adapter_config or {}
        self._calibration: dict[tuple[str, str], deque[float]] = defaultdict(
            lambda: deque(maxlen=self.calibration_samples)
        )
        self._adapters: dict[tuple[str, str], ReferenceAdapter] = {}
        self._approval_remaining: dict[tuple[str, str], int] = {}

    @classmethod
    def from_config(
        cls,
        monitor: MultiFeatureDriftMonitor,
        config: dict[str, Any],
    ) -> "DriftController":
        return cls(
            monitor=monitor,
            adaptive_features=tuple(config["adaptive_features"]),
            calibration_samples=int(config.get("calibration_samples", 50)),
            trusted_confidence=float(config.get("trusted_confidence", 0.98)),
            maximum_trusted_anomaly=float(
                config.get("maximum_trusted_anomaly", 0.75)
            ),
            minimum_history=int(config.get("minimum_history", 20)),
            auto_approve=bool(config.get("auto_approve", False)),
            approval_ttl_messages=int(
                config.get("approval_ttl_messages", 300)
            ),
            allow_approved_prediction_override=bool(
                config.get("allow_approved_prediction_override", False)
            ),
            adapter_config=config,
        )

    def approve_drift(self, device_id: str, feature: str) -> None:
        if feature not in self.adaptive_features:
            raise ValueError(f"Feature is not adaptive: {feature}")
        self._approval_remaining[(str(device_id), feature)] = (
            self.approval_ttl_messages
        )

    def reference_mean(
        self,
        device_id: str,
        feature: str,
    ) -> float | None:
        adapter = self._adapters.get((str(device_id), feature))
        return None if adapter is None else adapter.reference_mean

    def update(
        self,
        device_id: str,
        features: dict[str, Any],
        prediction: EdgePrediction,
    ) -> DriftControlResult:
        device = str(device_id)
        events = self.monitor.update(device, features)
        if self.auto_approve:
            for event in events:
                feature = str(event["feature"])
                if feature in self.adaptive_features:
                    self.approve_drift(device, feature)

        updated_features: list[str] = []
        allowed_features: list[str] = []
        approved_features: list[str] = []
        reasons: list[str] = []

        for feature in self.adaptive_features:
            if feature not in features:
                raise ValueError(f"Adaptive feature missing: {feature}")
            key = (device, feature)
            value = float(features[feature])
            adapter = self._adapters.get(key)
            approved = self._approval_remaining.get(key, 0) > 0
            trusted = self._is_trusted(
                features,
                prediction,
                allow_known_attack=(
                    approved and self.allow_approved_prediction_override
                ),
            )

            if adapter is None:
                if trusted:
                    calibration = self._calibration[key]
                    calibration.append(value)
                    if len(calibration) >= self.calibration_samples:
                        self._adapters[key] = self._build_adapter(
                            sum(calibration) / len(calibration)
                        )
                        reasons.append("reference_calibrated")
                else:
                    reasons.append("calibration_sample_not_trusted")
                continue

            if approved:
                approved_features.append(feature)
            result = adapter.update(
                value=value,
                trusted_sample=trusted,
                drift_confirmed=approved,
            )
            reasons.append(result.reason)
            if result.accepted:
                allowed_features.append(feature)
            if result.updated:
                updated_features.append(feature)

            if approved:
                remaining = self._approval_remaining[key] - 1
                if remaining > 0:
                    self._approval_remaining[key] = remaining
                else:
                    del self._approval_remaining[key]

        return DriftControlResult(
            drift_events=tuple(events),
            adaptation_allowed=bool(allowed_features),
            adaptation_updated=bool(updated_features),
            adaptation_reason=";".join(sorted(set(reasons))) or "none",
            adaptation_features=tuple(allowed_features),
            reference_updates=tuple(updated_features),
            reference_values=tuple(
                f"{feature}={self._adapters[(device, feature)].reference_mean:.6f}"
                for feature in self.adaptive_features
                if (device, feature) in self._adapters
            ),
            approved_features=tuple(approved_features),
        )

    def _is_trusted(
        self,
        features: dict[str, Any],
        prediction: EdgePrediction,
        allow_known_attack: bool = False,
    ) -> bool:
        protocol_safe = all(
            int(features.get(name, 0)) == 0
            for name in (
                "client_changed",
                "topic_changed",
                "unexpected_client_topic",
                "is_duplicate_sequence",
                "is_out_of_order",
            )
        ) and int(features.get("device_topic_match", 1)) == 1

        return (
            (prediction.known_prediction == "none" or allow_known_attack)
            and prediction.confidence >= self.trusted_confidence
            and prediction.anomaly_score <= self.maximum_trusted_anomaly
            and int(features.get("history_count", 0)) >= self.minimum_history
            and protocol_safe
        )

    def _build_adapter(self, initial_mean: float) -> ReferenceAdapter:
        return ReferenceAdapter(
            initial_mean=initial_mean,
            window_size=int(self.adapter_config.get("window_size", 30)),
            minimum_samples=int(
                self.adapter_config.get("minimum_samples", 20)
            ),
            blend_rate=float(self.adapter_config.get("blend_rate", 0.5)),
            maximum_update_step=float(
                self.adapter_config.get("maximum_update_step", 1.0)
            ),
            guarded=True,
        )
