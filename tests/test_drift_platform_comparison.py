import pandas as pd
import pytest

from src.evaluation.drift_platform_comparison import (
    build_drift_platform_comparison,
)


def summary(scale: float) -> pd.DataFrame:
    rows = []
    for scenario_type in ("measurement", "communication"):
        row: dict[str, object] = {
            "scenario_type": scenario_type,
            "runs": 5,
        }
        for metric in (
            "rows",
            "mean_detection_delay_messages",
            "active_alert_reduction_percent",
            "adaptation_updates",
            "latency_mean_ms",
            "latency_p95_ms",
            "latency_p99_ms",
        ):
            row[f"{metric}_mean"] = scale
            row[f"{metric}_std"] = scale / 10.0
        rows.append(row)
    return pd.DataFrame(rows)


def test_build_drift_platform_comparison_calculates_ratios() -> None:
    comparison = build_drift_platform_comparison(
        mac_summary=summary(10.0),
        pi_summary=summary(20.0),
    )
    assert len(comparison) == 4
    assert set(comparison["platform"]) == {"MacBook", "Raspberry Pi 5"}
    assert set(comparison["runs"]) == {5}
    assert comparison["pi_to_mac_latency_ratio"].tolist() == [
        pytest.approx(2.0),
        pytest.approx(2.0),
        pytest.approx(2.0),
        pytest.approx(2.0),
    ]


def test_missing_scenario_is_rejected() -> None:
    incomplete = summary(10.0).query("scenario_type == 'measurement'")
    with pytest.raises(ValueError, match="communication"):
        build_drift_platform_comparison(
            mac_summary=incomplete,
            pi_summary=summary(20.0),
        )
