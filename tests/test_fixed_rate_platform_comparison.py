import pandas as pd
import pytest

from src.evaluation.fixed_rate_platform_comparison import (
    build_fixed_rate_platform_comparison,
)


def make_summary(cpu: float) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "pipeline": pipeline,
            "configured_message_rate": rate,
            "achieved_message_rate_mean": rate,
            "cpu_percent_single_core_equivalent_mean": cpu,
            "cpu_percent_single_core_equivalent_std": 0.1,
            "cpu_time_per_message_ms_mean": cpu,
            "total_latency_mean_ms_mean": cpu,
            "deadline_misses_mean": 0.0,
            "process_peak_memory_after_mb_mean": 150.0,
        }
        for pipeline in ("logistic_regression", "random_forest", "open_set")
        for rate in (3.0, 10.0, 25.0)
    ])


def test_build_fixed_rate_platform_comparison() -> None:
    comparison = build_fixed_rate_platform_comparison(
        make_summary(2.0),
        make_summary(4.0),
    )
    assert len(comparison) == 9
    assert comparison["pi_to_mac_cpu_ratio"].tolist() == [2.0] * 9
    assert comparison["pipeline"].iloc[0] == "logistic_regression"
    assert comparison["configured_message_rate"].iloc[:3].tolist() == [
        3.0,
        10.0,
        25.0,
    ]


def test_unmatched_fixed_rate_groups_are_rejected() -> None:
    mac = make_summary(2.0)
    pi = make_summary(4.0).iloc[:-1]
    with pytest.raises(ValueError, match="do not match"):
        build_fixed_rate_platform_comparison(mac, pi)
