from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AdaptationResult:
    accepted: bool
    updated: bool
    reference_mean: float
    candidate_count: int
    reason: str


class ReferenceAdapter:
    """Update a scalar normal reference through a bounded candidate window."""

    def __init__(
        self,
        initial_mean: float,
        window_size: int = 30,
        minimum_samples: int = 20,
        blend_rate: float = 0.25,
        maximum_update_step: float = 1.0,
        guarded: bool = True,
    ) -> None:
        if not math.isfinite(initial_mean):
            raise ValueError("initial_mean must be finite")
        if window_size <= 0:
            raise ValueError("window_size must be greater than zero")
        if not 1 <= minimum_samples <= window_size:
            raise ValueError("minimum_samples must be within the window")
        if not 0 < blend_rate <= 1:
            raise ValueError("blend_rate must be in (0, 1]")
        if maximum_update_step <= 0:
            raise ValueError("maximum_update_step must be greater than zero")

        self.reference_mean = float(initial_mean)
        self.minimum_samples = int(minimum_samples)
        self.blend_rate = float(blend_rate)
        self.maximum_update_step = float(maximum_update_step)
        self.guarded = bool(guarded)
        self._candidates: deque[float] = deque(maxlen=window_size)
        self.update_count = 0

    def update(
        self,
        value: float,
        trusted_sample: bool,
        drift_confirmed: bool,
    ) -> AdaptationResult:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Adaptation value must be finite")

        if self.guarded and not drift_confirmed:
            return self._result(False, False, "drift_not_confirmed")
        if self.guarded and not trusted_sample:
            return self._result(False, False, "sample_not_trusted")

        self._candidates.append(numeric)
        if len(self._candidates) < self.minimum_samples:
            return self._result(True, False, "collecting_candidates")

        candidate_mean = sum(self._candidates) / len(self._candidates)
        proposed_change = (
            candidate_mean - self.reference_mean
        ) * self.blend_rate
        bounded_change = max(
            -self.maximum_update_step,
            min(self.maximum_update_step, proposed_change),
        )
        self.reference_mean += bounded_change
        self.update_count += 1
        self._candidates.clear()
        return self._result(True, True, "reference_updated")

    def _result(
        self,
        accepted: bool,
        updated: bool,
        reason: str,
    ) -> AdaptationResult:
        return AdaptationResult(
            accepted=accepted,
            updated=updated,
            reference_mean=self.reference_mean,
            candidate_count=len(self._candidates),
            reason=reason,
        )
