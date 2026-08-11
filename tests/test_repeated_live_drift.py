from pathlib import Path

import pandas as pd
import pytest

from src.drift.repeated_live_evaluation import evaluate_repeated_live_drift


def write_run(path: Path, alert_after_detection: bool) -> None:
    frame = pd.DataFrame({
        "device_id": ["meter_01"] * 6,
        "sequence_number": range(6),
        "true_drift_type": ["none", "none", "shift", "shift", "none", "none"],
        "decision": ["none", "none", "random", "random", "none", "none"],
        "drift_aware_decision": [
            "none", "none", "random",
            "random" if alert_after_detection else "normal_drift",
            "none", "none",
        ],
        "drift_detected": [0, 0, 0, 1, 0, 0],
        "adaptation_updated": [0, 0, 0, 1, 0, 0],
        "total_detection_ms": [1.0] * 6,
    })
    frame.to_csv(path, index=False)


def test_repeated_live_drift_aggregates_runs(tmp_path: Path) -> None:
    first = tmp_path / "measurement_drift_run_01.csv"
    second = tmp_path / "measurement_drift_run_02.csv"
    write_run(first, True)
    write_run(second, False)
    runs, summary, phases, phase_summary = evaluate_repeated_live_drift(
        [first, second]
    )
    assert len(runs) == 2
    assert summary.iloc[0]["runs"] == 2
    assert len(phases) == 8
    post = phase_summary[
        phase_summary["evaluation_phase"] == "drift_post_adaptation"
    ].iloc[0]
    assert post["drift_aware_false_alert_rate_mean"] == pytest.approx(0.5)


def test_repeated_live_drift_rejects_unexpected_filename(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    write_run(path, False)
    with pytest.raises(ValueError, match="filename"):
        evaluate_repeated_live_drift([path])
