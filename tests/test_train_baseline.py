import numpy as np
import pandas as pd
import pytest

from src.training.prepare_dataset import (
    PreparedDataset,
)
from src.training.split_data import (
    DatasetSplit,
)
from src.training.train_baseline import (
    train_logistic_baseline,
)


def build_training_inputs() -> tuple[
    PreparedDataset,
    DatasetSplit,
]:
    train = pd.DataFrame(
        {
            "feature_a": [
                0.0,
                0.2,
                0.4,
                2.0,
                2.2,
                2.4,
            ],
            "feature_b": [
                0.1,
                0.2,
                0.3,
                2.1,
                2.3,
                2.5,
            ],
            "is_attack": [
                0,
                0,
                0,
                1,
                1,
                1,
            ],
            "source_file": [
                "normal_run_01.csv",
                "normal_run_01.csv",
                "normal_run_01.csv",
                "constant_run_01.csv",
                "constant_run_01.csv",
                "constant_run_01.csv",
            ],
            "attack_type": [
                "none",
                "none",
                "none",
                "constant",
                "constant",
                "constant",
            ],
        }
    )

    validation = pd.DataFrame(
        {
            "feature_a": [
                0.1,
                2.1,
            ],
            "feature_b": [
                0.2,
                2.2,
            ],
            "is_attack": [
                0,
                1,
            ],
            "source_file": [
                "normal_run_02.csv",
                "constant_run_02.csv",
            ],
            "attack_type": [
                "none",
                "constant",
            ],
        }
    )

    test = pd.DataFrame(
        {
            "feature_a": [
                0.3,
                2.3,
            ],
            "feature_b": [
                0.4,
                2.4,
            ],
            "is_attack": [
                0,
                1,
            ],
            "source_file": [
                "normal_run_03.csv",
                "constant_run_03.csv",
            ],
            "attack_type": [
                "none",
                "constant",
            ],
        }
    )

    prepared = PreparedDataset(
        dataframe=pd.concat(
            [
                train,
                validation,
                test,
            ],
            ignore_index=True,
        ),
        feature_columns=(
            "feature_a",
            "feature_b",
        ),
        target_column="is_attack",
        group_column="source_file",
    )

    split = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        train_groups=(
            "normal_run_01.csv",
            "constant_run_01.csv",
        ),
        validation_groups=(
            "normal_run_02.csv",
            "constant_run_02.csv",
        ),
        test_groups=(
            "normal_run_03.csv",
            "constant_run_03.csv",
        ),
    )

    return prepared, split


def test_logistic_baseline_trains() -> None:
    prepared, split = (
        build_training_inputs()
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
        random_seed=42,
    )

    assert result.model is not None
    assert result.scaler is not None

    assert result.feature_columns == (
        "feature_a",
        "feature_b",
    )


def test_training_result_has_expected_shapes() -> None:
    prepared, split = (
        build_training_inputs()
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
    )

    assert result.x_train.shape == (
        6,
        2,
    )

    assert result.x_validation.shape == (
        2,
        2,
    )

    assert result.x_test.shape == (
        2,
        2,
    )

    assert result.y_train.shape == (
        6,
    )


def test_scaler_is_fitted_only_on_training_data() -> None:
    prepared, split = (
        build_training_inputs()
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
    )

    expected_mean = (
        split.train[
            [
                "feature_a",
                "feature_b",
            ]
        ]
        .mean()
        .to_numpy()
    )

    assert np.allclose(
        result.scaler.mean_,
        expected_mean,
    )


def test_probabilities_are_between_zero_and_one() -> None:
    prepared, split = (
        build_training_inputs()
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
    )

    for probabilities in [
        result.train_probabilities,
        result.validation_probabilities,
        result.test_probabilities,
    ]:
        assert (
            probabilities >= 0
        ).all()

        assert (
            probabilities <= 1
        ).all()


def test_predictions_are_binary() -> None:
    prepared, split = (
        build_training_inputs()
    )

    result = train_logistic_baseline(
        prepared=prepared,
        split=split,
    )

    for predictions in [
        result.train_predictions,
        result.validation_predictions,
        result.test_predictions,
    ]:
        assert set(
            np.unique(predictions)
        ).issubset(
            {0, 1}
        )


def test_training_rejects_single_class_train_split() -> None:
    prepared, split = (
        build_training_inputs()
    )

    invalid_train = split.train.copy()

    invalid_train["is_attack"] = 0

    invalid_split = DatasetSplit(
        train=invalid_train,
        validation=split.validation,
        test=split.test,
        train_groups=split.train_groups,
        validation_groups=(
            split.validation_groups
        ),
        test_groups=split.test_groups,
    )

    with pytest.raises(
        ValueError,
        match="both classes",
    ):
        train_logistic_baseline(
            prepared=prepared,
            split=invalid_split,
        )


def test_invalid_max_iter_is_rejected() -> None:
    prepared, split = (
        build_training_inputs()
    )

    with pytest.raises(
        ValueError,
        match="max_iter",
    ):
        train_logistic_baseline(
            prepared=prepared,
            split=split,
            max_iter=0,
        )


def test_missing_feature_is_rejected() -> None:
    prepared, split = (
        build_training_inputs()
    )

    invalid_validation = (
        split.validation.drop(
            columns=["feature_a"]
        )
    )

    invalid_split = DatasetSplit(
        train=split.train,
        validation=invalid_validation,
        test=split.test,
        train_groups=split.train_groups,
        validation_groups=(
            split.validation_groups
        ),
        test_groups=split.test_groups,
    )

    with pytest.raises(
        ValueError,
        match="missing columns",
    ):
        train_logistic_baseline(
            prepared=prepared,
            split=invalid_split,
        )