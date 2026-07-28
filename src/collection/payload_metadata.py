from __future__ import annotations

import json
from typing import Any


OPERATIONAL_PAYLOAD_FIELDS = (
    "device_id",
    "timestamp",
    "sequence_number",
    "voltage",
    "current",
    "power",
    "frequency",
)


def calculate_operational_payload_size(
    payload: dict[str, Any],
) -> int:
    """
    Calculate payload size without experiment-only labels.

    Ground-truth fields such as attack_type, is_attack, attack_step and
    scenario_id must not influence protocol features.
    """
    missing_fields = (
        set(OPERATIONAL_PAYLOAD_FIELDS)
        - set(payload)
    )

    if missing_fields:
        raise ValueError(
            "Operational payload is missing fields: "
            f"{sorted(missing_fields)}"
        )

    operational_payload = {
        field: payload[field]
        for field in OPERATIONAL_PAYLOAD_FIELDS
    }

    encoded = json.dumps(
        operational_payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")

    return len(encoded)