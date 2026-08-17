from scripts.summarize_results import build_summary


def test_final_summary_contains_all_research_components() -> None:
    summary, table = build_summary()
    assert set(summary) == {
        "multi_view",
        "open_set",
        "model_comparison",
        "edge_gateway",
        "raspberry_pi_gateway",
        "cross_platform",
        "live_mqtt_deployment",
        "drift",
        "raspberry_pi_live_drift",
        "normal_mqtt_cpu",
        "limitations",
    }
    assert summary["open_set"]["unknown_recall"] > 0.9
    assert summary["limitations"]["raspberry_pi_measured"] is True
    assert summary["raspberry_pi_gateway"]["runs"] == 5
    assert summary["live_mqtt_deployment"]["known_attack_alert_rate"] == 1.0
    assert summary["live_mqtt_deployment"]["unseen_unknown_recall"] > 0.9
    assert summary["limitations"]["raspberry_pi_drift_measured"] is True
    assert summary["limitations"]["normal_load_cpu_measured"] is True
    assert summary["raspberry_pi_live_drift"]["runs_per_scenario"] == 5
    assert (
        summary["raspberry_pi_live_drift"]
        ["communication_detection_delay_messages"]
        == 5.0
    )
    assert (
        summary["raspberry_pi_live_drift"]
        ["maximum_observed_temperature_c"]
        == 60.9
    )
    assert (
        summary["raspberry_pi_live_drift"]
        ["thermal_throttling_observed"]
        is False
    )
    assert summary["normal_mqtt_cpu"]["message_rate"] == 6.0
    assert summary["normal_mqtt_cpu"]["deadline_misses_pi"] == 0.0
    assert not table.empty
