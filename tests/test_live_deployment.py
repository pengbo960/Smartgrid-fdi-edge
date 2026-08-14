from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.live_deployment import summarize_live_deployment_logs


def test_summarize_live_deployment_log(tmp_path: Path) -> None:
    path = tmp_path / "gradual.csv"
    pd.DataFrame({
        "scenario_id": ["gradual_01"] * 4,
        "device_id": ["meter_01", "meter_02", "meter_02", "meter_03"],
        "sequence_number": [1, 1, 2, 1],
        "true_attack_type": ["none", "gradual", "gradual", "none"],
        "decision": ["none", "gradual", "none", "known_attack"],
        "total_detection_ms": [1.0, 2.0, 3.0, 4.0],
    }).to_csv(path, index=False)
    summary = summarize_live_deployment_logs([path]).iloc[0]
    assert summary["messages"] == 4
    assert summary["devices"] == 3
    assert summary["attack_alert_rate"] == pytest.approx(0.5)
    assert summary["unknown_rate_on_attack"] == pytest.approx(0.0)
    assert summary["exact_attack_class_rate"] == pytest.approx(0.5)
    assert summary["normal_alert_rate"] == pytest.approx(0.5)
    assert summary["mean_latency_ms"] == pytest.approx(2.5)
    assert summary["p95_latency_ms"] == pytest.approx(3.85)


def test_empty_live_log_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    pd.DataFrame(columns=sorted(REQUIRED_COLUMNS)).to_csv(path, index=False)
    with pytest.raises(ValueError, match="empty"):
        summarize_live_deployment_logs([path])


REQUIRED_COLUMNS = {
    "scenario_id",
    "device_id",
    "sequence_number",
    "true_attack_type",
    "decision",
    "total_detection_ms",
}
