from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DriftSchedule:
    drift_type: str
    start_step: int
    end_step: int
    device_id: str = "*"
    voltage_offset: float | None = None
    target_interval: float | None = None
    transition_steps: int = 1

    def __post_init__(self) -> None:
        if self.drift_type not in {"measurement_shift", "publish_interval"}:
            raise ValueError(f"Unsupported drift type: {self.drift_type}")
        if self.start_step < 0 or self.end_step <= self.start_step:
            raise ValueError("Invalid drift step range")
        if self.transition_steps <= 0:
            raise ValueError("transition_steps must be greater than zero")
        if self.drift_type == "measurement_shift" and self.voltage_offset is None:
            raise ValueError("measurement_shift requires voltage_offset")
        if (
            self.drift_type == "publish_interval"
            and (self.target_interval is None or self.target_interval <= 0)
        ):
            raise ValueError("publish_interval requires positive target_interval")


class NormalDriftManager:
    """Apply labelled, legitimate measurement or communication drift."""

    def __init__(self, schedules: tuple[DriftSchedule, ...] = ()) -> None:
        self.schedules = schedules

    @classmethod
    def from_config(cls, raw_drifts: list[dict[str, Any]] | None) -> "NormalDriftManager":
        if raw_drifts is None:
            return cls()
        if not isinstance(raw_drifts, list):
            raise TypeError("drifts section must be a list")
        return cls(
            tuple(
                DriftSchedule(
                    drift_type=str(item["drift_type"]),
                    start_step=int(item["start_step"]),
                    end_step=int(item["end_step"]),
                    device_id=str(item.get("device_id", "*")),
                    voltage_offset=(
                        float(item["voltage_offset"])
                        if item.get("voltage_offset") is not None
                        else None
                    ),
                    target_interval=(
                        float(item["target_interval"])
                        if item.get("target_interval") is not None
                        else None
                    ),
                    transition_steps=int(item.get("transition_steps", 1)),
                )
                for item in raw_drifts
            )
        )

    def apply_measurements(
        self,
        device_id: str,
        step: int,
        measurements: dict[str, float],
        power_factor: float = 0.95,
    ) -> tuple[dict[str, float], str, int | None]:
        output = measurements.copy()
        for schedule in self.schedules:
            if schedule.drift_type != "measurement_shift":
                continue
            if schedule.device_id not in {"*", device_id}:
                continue
            if not schedule.start_step <= step < schedule.end_step:
                continue
            drift_step = step - schedule.start_step
            progress = min(1.0, (drift_step + 1) / schedule.transition_steps)
            offset = float(schedule.voltage_offset) * progress
            output["voltage"] = round(output["voltage"] + offset, 4)
            output["power"] = round(
                output["voltage"] * output["current"] * power_factor,
                4,
            )
            return output, schedule.drift_type, drift_step
        return output, "none", None

    def publish_interval(
        self,
        step: int,
        base_interval: float,
    ) -> tuple[float, str, int | None]:
        for schedule in self.schedules:
            if schedule.drift_type != "publish_interval":
                continue
            if not schedule.start_step <= step < schedule.end_step:
                continue
            drift_step = step - schedule.start_step
            progress = min(1.0, (drift_step + 1) / schedule.transition_steps)
            interval = base_interval + (
                float(schedule.target_interval) - base_interval
            ) * progress
            return interval, schedule.drift_type, drift_step
        return base_interval, "none", None
