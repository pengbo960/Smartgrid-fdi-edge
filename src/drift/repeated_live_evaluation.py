from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.drift.live_evaluation import (
    build_live_drift_phase_metrics,
    evaluate_live_drift,
)
from src.evaluation.repeated_experiments import aggregate_repeated_runs


RUN_PATTERN = re.compile(
    r"^(measurement|communication)_drift_run_(\d+)$"
)


def evaluate_repeated_live_drift(
    paths: Iterable[str | Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate independent live MQTT trials and aggregate mean/sample SD."""
    run_rows: list[dict[str, object]] = []
    phase_frames: list[pd.DataFrame] = []
    seen: set[tuple[str, int]] = set()
    for raw_path in paths:
        path = Path(raw_path)
        match = RUN_PATTERN.match(path.stem)
        if match is None:
            raise ValueError(f"Unexpected repeated live drift filename: {path.name}")
        scenario_type = match.group(1)
        run_id = int(match.group(2))
        key = (scenario_type, run_id)
        if key in seen:
            raise ValueError(f"Duplicate repeated live drift run: {key}")
        seen.add(key)
        frame = pd.read_csv(path)
        overall = evaluate_live_drift(frame)
        run_rows.append({
            "scenario_type": scenario_type,
            "run_id": run_id,
            "source_file": path.name,
            "rows": overall["rows"],
            "active_drift_rows": overall["active_drift_rows"],
            "recovery_rows": overall["recovery_rows"],
            "mean_detection_delay_messages": overall[
                "mean_detection_delay_messages"
            ],
            "active_alert_reduction_percent": overall[
                "active_alert_reduction_percent"
            ],
            "adaptation_updates": overall["adaptation_updates"],
            "latency_mean_ms": overall["latency_mean_ms"],
            "latency_p95_ms": overall["latency_p95_ms"],
            "latency_p99_ms": overall["latency_p99_ms"],
        })
        phases = build_live_drift_phase_metrics(frame)
        phases.insert(0, "run_id", run_id)
        phases.insert(0, "scenario_type", scenario_type)
        phases.insert(2, "source_file", path.name)
        phase_frames.append(phases)

    runs = pd.DataFrame(run_rows).sort_values(["scenario_type", "run_id"])
    phases = pd.concat(phase_frames, ignore_index=True).sort_values(
        ["scenario_type", "run_id", "evaluation_phase"]
    )
    run_summary = aggregate_repeated_runs(
        runs,
        ("scenario_type",),
        (
            "rows", "active_drift_rows", "recovery_rows",
            "mean_detection_delay_messages", "active_alert_reduction_percent",
            "adaptation_updates", "latency_mean_ms", "latency_p95_ms",
            "latency_p99_ms",
        ),
    )
    phase_summary = aggregate_repeated_runs(
        phases,
        ("scenario_type", "evaluation_phase"),
        (
            "rows", "raw_false_alert_rate", "drift_aware_false_alert_rate",
            "false_alert_reduction_percent", "normal_drift_rows",
            "drift_detections", "adaptation_updates", "latency_mean_ms",
            "latency_p95_ms",
        ),
    )
    return runs, run_summary, phases, phase_summary
