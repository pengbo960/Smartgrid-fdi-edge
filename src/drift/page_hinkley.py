from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PageHinkleyResult:
    drift_detected: bool
    direction: str | None
    sample_count: int
    mean: float
    statistic: float


class PageHinkley:
    """Two-sided Page-Hinkley mean-shift detector for online streams."""

    def __init__(
        self,
        delta: float = 0.01,
        threshold: float = 5.0,
        minimum_instances: int = 30,
        reset_after_drift: bool = True,
    ) -> None:
        if delta < 0:
            raise ValueError("delta must be zero or greater")
        if threshold <= 0:
            raise ValueError("threshold must be greater than zero")
        if minimum_instances < 2:
            raise ValueError("minimum_instances must be at least two")
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.minimum_instances = int(minimum_instances)
        self.reset_after_drift = bool(reset_after_drift)
        self.reset()

    def reset(self) -> None:
        self.sample_count = 0
        self.mean = 0.0
        self._positive_sum = 0.0
        self._negative_sum = 0.0

    def update(self, value: float) -> PageHinkleyResult:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Page-Hinkley value must be finite")

        self.sample_count += 1
        self.mean += (numeric - self.mean) / self.sample_count
        difference = numeric - self.mean
        self._positive_sum = max(
            0.0,
            self._positive_sum + difference - self.delta,
        )
        self._negative_sum = min(
            0.0,
            self._negative_sum + difference + self.delta,
        )

        direction: str | None = None
        statistic = max(self._positive_sum, -self._negative_sum)
        if self.sample_count >= self.minimum_instances:
            if self._positive_sum > self.threshold:
                direction = "increase"
            elif -self._negative_sum > self.threshold:
                direction = "decrease"

        result = PageHinkleyResult(
            drift_detected=direction is not None,
            direction=direction,
            sample_count=self.sample_count,
            mean=self.mean,
            statistic=statistic,
        )
        if direction is not None and self.reset_after_drift:
            self.reset()
        return result
