from pathlib import Path

import pandas as pd
import pytest

from src.evaluation.repeated_experiments import (
    aggregate_binary_confusion_matrix,
    aggregate_repeated_runs,
    configure_ablation_run,
    extract_open_set_row,
    save_table,
    validate_folds,
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


def test_validate_folds() -> None:
    assert validate_folds([1, 3, 5], 5) == (1, 3, 5)
    with pytest.raises(ValueError, match="unique"):
        validate_folds([1, 1], 5)
    with pytest.raises(ValueError, match="between 1 and 5"):
        validate_folds([0], 5)
    with pytest.raises(TypeError, match="integer"):
        validate_folds(["1"], 5)


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


def test_configure_ablation_fold(tmp_path: Path) -> None:
    base = {
        "split": {"random_seed": 1},
        "model": {"random_seed": 1},
        "output": {"summary_csv": "old.csv"},
    }
    configured = configure_ablation_run(
        base,
        44,
        tmp_path,
        fold_index=2,
        validation_offset=1,
    )
    assert configured["split"] == {
        "random_seed": 44,
        "strategy": "grouped_kfold",
        "fold_index": 2,
        "validation_offset": 1,
    }
    assert configured["model"]["random_seed"] == 44


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


def test_aggregate_binary_confusion_matrix() -> None:
    runs = pd.DataFrame(
        {
            "experiment_name": ["all_views", "all_views", "value_only"],
            "true_negative": [90, 91, 80],
            "false_positive": [10, 9, 20],
            "false_negative": [2, 3, 5],
            "true_positive": [18, 17, 15],
        }
    )
    matrix = aggregate_binary_confusion_matrix(runs, "all_views")
    assert matrix.tolist() == [[181, 19], [5, 35]]


def test_aggregate_binary_confusion_matrix_requires_experiment() -> None:
    runs = pd.DataFrame(
        {
            "experiment_name": ["value_only"],
            "true_negative": [90],
            "false_positive": [10],
            "false_negative": [2],
            "true_positive": [18],
        }
    )
    with pytest.raises(ValueError, match="No runs found"):
        aggregate_binary_confusion_matrix(runs, "all_views")


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
            "unknown_true_positive": 270,
            "unknown_false_positive": 48,
        },
        "dataset": {
            "unseen_test_rows": 300,
            "unseen_source_files": ["gradual_run_01.csv"],
        },
    }
    row = extract_open_set_row(report, 42, fold=1)
    assert row["seed"] == 42
    assert row["fold"] == 1
    assert row["unknown_recall"] == 0.9
    assert row["unknown_true_positive"] == 270
    assert row["unseen_test_rows"] == 300
    assert row["unseen_source_files"] == "gradual_run_01.csv"
    assert row["known_none_recall"] == 0.96
