from scripts.summarize_results import build_summary


def test_final_summary_contains_all_research_components() -> None:
    summary, table = build_summary()
    assert set(summary) == {
        "multi_view",
        "open_set",
        "model_comparison",
        "edge_gateway",
        "drift",
        "limitations",
    }
    assert summary["open_set"]["unknown_recall"] > 0.9
    assert summary["limitations"]["raspberry_pi_measured"] is False
    assert not table.empty
