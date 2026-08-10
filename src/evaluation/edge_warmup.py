from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def partition_device_warmup(
    rows: Iterable[dict[str, Any]], messages_per_device: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Select the first N rows of every device as stateful warm-up."""
    if messages_per_device < 0:
        raise ValueError("messages_per_device must be zero or greater")
    counts: dict[str, int] = defaultdict(int)
    warmup: list[dict[str, Any]] = []
    measured: list[dict[str, Any]] = []
    for row in rows:
        if "device_id" not in row:
            raise ValueError("Warm-up row is missing device_id")
        device_id = str(row["device_id"])
        if counts[device_id] < messages_per_device:
            warmup.append(row)
            counts[device_id] += 1
        else:
            measured.append(row)
    if not measured:
        raise ValueError("Per-device warm-up must leave measured messages")
    return warmup, measured, dict(counts)
