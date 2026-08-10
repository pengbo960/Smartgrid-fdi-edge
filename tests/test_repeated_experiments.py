from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.repeated_experiments import (
    aggregate_repeated_runs,
    configure_ablation_run,
    extract_open_set_row,
    save_table,
    validate_seeds,
)


def test_validate_seeds() -> None:
    assert validate_seeds([42, 43]) == (42, 43)
    with pytest.raises(ValueError, match="unique"):
        validate_seeds([42, 42])
    with pytest.raises(ValueError, match="At least one"):
        validate_seeds([])
    with pytest.raises(TypeError, match="integer"):
        validate_seeds([42, "43"])


def test_configure_ablation_run_does_not_mutate_base(tmp_path: Path) -> None:
    base = {
        "split": {"random_seed": 1},
        "model": {"random_seed": 1},
        "output": {"summary_csv": "old.csv"},
    }
    configured = configure_ablation_run(base, 44, tmp_path)
    assert base["split"]["random_seed"] == 1
    assert configured["split"]["random_seed"] == 44
    assert configured["model"]["random_seed"] == 44
    assert configured["output"]["summary_csv"] == str(tmp_path / "summary.csv")


def test_aggregate_repeated_runs_uses_sample_standard_deviation() -> None:
    runs = pd.DataFrame({
        "seed": [42, 43, 42, 43],
        "experiment": ["a", "a", "b", "b"],
        "macro_f1": [0.8, 1.0, 0.5, 0.7],
    })
    summary = aggregate_repeated_runs(runs, ("experiment",))
    first = summary[summary["experiment"] == "a"].iloc[0]
    assert first["runs"] == 2
    assert first["macro_f1_mean"] == pytest.approx(0.9)
    assert first["macro_f1_std"] == pytest.approx(0.1414213562)
    assert first["macro_f1_min"] == pytest.approx(0.8)
    assert first["macro_f1_max"] == pytest.approx(1.0)


def test_save_table_creates_parent(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "table.csv"
    save_table(pd.DataFrame({"value": [1]}), output)
    assert pd.read_csv(output)["value"].tolist() == [1]


def test_extract_open_set_row() -> None:
    report = {
        "thresholds": {"confidence_threshold": 0.8, "anomaly_threshold": 0.6},
        "known_closed_set": {"accuracy": 0.99, "macro_f1": 0.98},
        "known_open_set": {
            "false_unknown_rate": 0.02, "acceptance_rate": 0.98,
            "overall_correct_rate": 0.97, "per_class_recall": {"none": 0.96},
        },
        "unseen": {
            "unknown_recall": 0.9, "unknown_precision": 0.85,
            "confidence_only_recall": 0.1, "normal_anomaly_recall": 0.88,
            "mean_first_unknown_step": 4.0,
        },
    }
    row = extract_open_set_row(report, 42)
    assert row["seed"] == 42
    assert row["unknown_recall"] == 0.9
    assert row["known_none_recall"] == 0.96
