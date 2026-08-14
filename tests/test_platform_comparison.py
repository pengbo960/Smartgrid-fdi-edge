import pandas as pd
import pytest

from src.evaluation.platform_comparison import build_platform_comparison


def summary_row(model_name: str | None, scale: float) -> dict[str, object]:
    row: dict[str, object] = {}
    if model_name is not None:
        row["model_name"] = model_name
    for metric in (
        "throughput_messages_per_second",
        "total_latency_mean_ms",
        "total_latency_p95_ms",
        "model_inference_mean_ms",
        "cpu_percent_single_core_equivalent",
        "process_peak_memory_after_mb",
    ):
        row[f"{metric}_mean"] = scale
        row[f"{metric}_std"] = scale / 10.0
    return row


def test_build_platform_comparison_calculates_ratios() -> None:
    mac_models = pd.DataFrame([
        summary_row("logistic_regression", 2.0),
        summary_row("random_forest", 4.0),
    ])
    pi_models = pd.DataFrame([
        summary_row("logistic_regression", 4.0),
        summary_row("random_forest", 8.0),
    ])
    comparison = build_platform_comparison(
        mac_model_summary=mac_models,
        pi_model_summary=pi_models,
        mac_open_set_summary=pd.DataFrame([summary_row(None, 5.0)]),
        pi_open_set_summary=pd.DataFrame([summary_row(None, 10.0)]),
    )
    assert comparison["pipeline"].tolist() == [
        "logistic_regression",
        "random_forest",
        "open_set",
    ]
    assert comparison["pi_to_mac_latency_ratio"].tolist() == [2.0, 2.0, 2.0]
    assert comparison["mac_to_pi_throughput_ratio"].tolist() == [0.5, 0.5, 0.5]
    assert comparison["pi_memory_change_percent"].tolist() == [
        pytest.approx(100.0),
        pytest.approx(100.0),
        pytest.approx(100.0),
    ]


def test_missing_model_is_rejected() -> None:
    only_logistic = pd.DataFrame([summary_row("logistic_regression", 2.0)])
    open_set = pd.DataFrame([summary_row(None, 5.0)])
    with pytest.raises(ValueError, match="random_forest"):
        build_platform_comparison(
            only_logistic,
            only_logistic,
            open_set,
            open_set,
        )
