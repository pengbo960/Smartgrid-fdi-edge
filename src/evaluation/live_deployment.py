from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "scenario_id",
    "device_id",
    "sequence_number",
    "true_attack_type",
    "decision",
    "total_detection_ms",
}


def summarize_live_deployment_logs(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Summarise real MQTT logs without treating them as a new accuracy test."""
    rows: list[dict[str, object]] = []
    path_list = [Path(path) for path in paths]
    if not path_list:
        raise ValueError("At least one live deployment log is required")
    for path in path_list:
        frame = pd.read_csv(path)
        missing = REQUIRED_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError(f"Live deployment log is empty: {path}")
        decisions = frame["decision"].astype(str)
        true_attacks = frame["true_attack_type"].astype(str)
        attack_mask = true_attacks.ne("none")
        normal_mask = ~attack_mask
        alerted = decisions.ne("none")
        unknown = decisions.eq("unknown")
        exact_attack_class = decisions.eq(true_attacks)
        latencies = pd.to_numeric(frame["total_detection_ms"], errors="raise")
        scenario_values = frame["scenario_id"].dropna().astype(str).unique()
        rows.append({
            "source_file": path.name,
            "scenario_id": (
                scenario_values[0] if len(scenario_values) == 1 else "mixed"
            ),
            "messages": len(frame),
            "devices": frame["device_id"].astype(str).nunique(),
            "attack_messages": int(attack_mask.sum()),
            "normal_messages": int(normal_mask.sum()),
            "alert_messages": int(alerted.sum()),
            "unknown_messages": int(unknown.sum()),
            "attack_alert_rate": float(alerted[attack_mask].mean())
            if attack_mask.any() else None,
            "unknown_rate_on_attack": float(unknown[attack_mask].mean())
            if attack_mask.any() else None,
            "exact_attack_class_rate": float(
                exact_attack_class[attack_mask].mean()
            )
            if attack_mask.any() else None,
            "normal_alert_rate": float(alerted[normal_mask].mean())
            if normal_mask.any() else None,
            "mean_latency_ms": float(latencies.mean()),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "maximum_latency_ms": float(latencies.max()),
        })
    return pd.DataFrame(rows).sort_values("source_file").reset_index(drop=True)
