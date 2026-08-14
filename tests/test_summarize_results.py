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
        "limitations",
    }
    assert summary["open_set"]["unknown_recall"] > 0.9
    assert summary["limitations"]["raspberry_pi_measured"] is True
    assert summary["raspberry_pi_gateway"]["runs"] == 5
    assert summary["live_mqtt_deployment"]["known_attack_alert_rate"] == 1.0
    assert summary["live_mqtt_deployment"]["unseen_unknown_recall"] > 0.9
    assert not table.empty
