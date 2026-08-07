from __future__ import annotations

from typing import Any

import pandas as pd


NON_ALERT_DECISIONS = {"none", "normal_drift"}


def evaluate_live_drift(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarise one labelled live MQTT drift experiment."""
    required = {
        "device_id",
        "sequence_number",
        "true_drift_type",
        "decision",
        "drift_aware_decision",
        "drift_detected",
        "adaptation_updated",
        "total_detection_ms",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Live drift results missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Live drift results must not be empty")

    data = frame.copy()
    data["evaluation_phase"] = "baseline"
    affected_devices: list[str] = []
    detection_delays: dict[str, int | None] = {}

    for device_id, indices in data.groupby("device_id", sort=True).groups.items():
        device = data.loc[indices]
        active = device[device["true_drift_type"] != "none"]
        if active.empty:
            continue
        affected_devices.append(str(device_id))
        start = int(active["sequence_number"].min())
        end = int(active["sequence_number"].max())
        sequences = data.loc[indices, "sequence_number"].astype(int)
        data.loc[indices[sequences.between(start, end)], "evaluation_phase"] = (
            "active_drift"
        )
        data.loc[indices[sequences > end], "evaluation_phase"] = "recovery"

        detections = device[
            (device["drift_detected"] == 1)
            & (device["sequence_number"].astype(int).between(start, end))
        ]
        detection_delays[str(device_id)] = (
            None
            if detections.empty
            else int(detections["sequence_number"].min()) - start
        )

    active = data[data["evaluation_phase"] == "active_drift"]
    baseline = data[data["evaluation_phase"] == "baseline"]
    recovery = data[data["evaluation_phase"] == "recovery"]
    raw_active_alerts = int((active["decision"] != "none").sum())
    aware_active_alerts = int(
        (~active["drift_aware_decision"].isin(NON_ALERT_DECISIONS)).sum()
    )
    reduction = (
        0.0
        if raw_active_alerts == 0
        else 100.0 * (raw_active_alerts - aware_active_alerts) / raw_active_alerts
    )
    observed_delays = [value for value in detection_delays.values() if value is not None]

    return {
        "rows": int(len(data)),
        "affected_devices": affected_devices,
        "active_drift_rows": int(len(active)),
        "recovery_rows": int(len(recovery)),
        "detection_delay_messages": detection_delays,
        "mean_detection_delay_messages": (
            None if not observed_delays else sum(observed_delays) / len(observed_delays)
        ),
        "raw_active_alerts": raw_active_alerts,
        "drift_aware_active_alerts": aware_active_alerts,
        "active_alert_reduction_percent": reduction,
        "baseline_raw_false_alerts": int((baseline["decision"] != "none").sum()),
        "baseline_drift_aware_false_alerts": int(
            (~baseline["drift_aware_decision"].isin(NON_ALERT_DECISIONS)).sum()
        ),
        "recovery_normal_drift_rows": int(
            (recovery["drift_aware_decision"] == "normal_drift").sum()
        ),
        "adaptation_updates": int(data["adaptation_updated"].sum()),
        "latency_mean_ms": float(data["total_detection_ms"].mean()),
        "latency_p95_ms": float(data["total_detection_ms"].quantile(0.95)),
        "latency_p99_ms": float(data["total_detection_ms"].quantile(0.99)),
    }
