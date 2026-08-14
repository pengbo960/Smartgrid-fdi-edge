from __future__ import annotations

import math
import re
from typing import Iterable


def validate_message_rates(values: Iterable[float]) -> tuple[float, ...]:
    rates = tuple(float(value) for value in values)
    if not rates:
        raise ValueError("At least one message rate is required")
    if any(not math.isfinite(rate) or rate <= 0.0 for rate in rates):
        raise ValueError("Message rates must be finite and greater than zero")
    if len(rates) != len(set(rates)):
        raise ValueError("Message rates must be unique")
    return rates


def calculate_target_messages(message_rate: float, duration_seconds: float) -> int:
    if not math.isfinite(message_rate) or message_rate <= 0.0:
        raise ValueError("message_rate must be finite and greater than zero")
    if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and greater than zero")
    return max(1, int(round(message_rate * duration_seconds)))


def calculate_cpu_metrics(
    cpu_seconds: float,
    elapsed_seconds: float,
    measured_messages: int,
    logical_cpu_count: int,
) -> dict[str, float]:
    if cpu_seconds < 0.0:
        raise ValueError("cpu_seconds must not be negative")
    if elapsed_seconds <= 0.0:
        raise ValueError("elapsed_seconds must be greater than zero")
    if measured_messages <= 0:
        raise ValueError("measured_messages must be greater than zero")
    if logical_cpu_count <= 0:
        raise ValueError("logical_cpu_count must be greater than zero")
    single_core = cpu_seconds / elapsed_seconds * 100.0
    return {
        "process_cpu_seconds": cpu_seconds,
        "cpu_time_per_message_ms": cpu_seconds / measured_messages * 1000.0,
        "cpu_percent_single_core_equivalent": single_core,
        "cpu_percent_total_machine_capacity": single_core / logical_cpu_count,
    }


def parse_temperature(output: str) -> float | None:
    match = re.search(r"temp=([0-9]+(?:\.[0-9]+)?)", output)
    return float(match.group(1)) if match else None


def parse_throttled(output: str) -> str | None:
    match = re.search(r"throttled=(0x[0-9a-fA-F]+)", output)
    return match.group(1).lower() if match else None
