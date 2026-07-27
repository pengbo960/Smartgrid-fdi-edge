from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.training.prepare_dataset import (
    PreparedDataset,
)
from src.training.split_data import (
    DatasetSplit,
)


@dataclass(frozen=True)
class BaselineTrainingResult:
    """
    Trained Logistic Regression baseline and its transformed datasets.
    """

    model: LogisticRegression
    scaler: StandardScaler
    feature_columns: tuple[str, ...]

    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray

    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray

    train_predictions: np.ndarray
    validation_predictions: np.ndarray
    test_predictions: np.ndarray

    train_probabilities: np.ndarray
    validation_probabilities: np.ndarray
    test_probabilities: np.ndarray


def _extract_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract numeric feature and target arrays from one split.
    """
    missing_columns = (
        set(feature_columns)
        | {target_column}
    ) - set(dataframe.columns)

    if missing_columns:
        raise ValueError(
            "Dataset split is missing columns: "
            f"{sorted(missing_columns)}"
        )

    features = dataframe[
        list(feature_columns)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    target = pd.to_numeric(
        dataframe[target_column],
        errors="coerce",
    )

    if features.isna().any().any():
        raise ValueError(
            "Feature matrix contains missing or invalid values"
        )

    if target.isna().any():
        raise ValueError(
            "Target contains missing or invalid values"
        )

    x = features.to_numpy(
        dtype=float
    )

    y = target.to_numpy(
        dtype=int
    )

    if not np.isfinite(x).all():
        raise ValueError(
            "Feature matrix contains non-finite values"
        )

    if set(np.unique(y)) - {0, 1}:
        raise ValueError(
            "Baseline target must contain only 0 and 1"
        )

    return x, y


def train_logistic_baseline(
    prepared: PreparedDataset,
    split: DatasetSplit,
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 1000,
    random_seed: int = 42,
) -> BaselineTrainingResult:
    """
    Train a Logistic Regression baseline.

    The scaler is fitted only on the training split. Validation and test
    data are transformed using the training scaler to avoid leakage.
    """
    if max_iter <= 0:
        raise ValueError(
            "max_iter must be greater than zero"
        )

    feature_columns = (
        prepared.feature_columns
    )

    target_column = (
        prepared.target_column
    )

    x_train_raw, y_train = (
        _extract_features_and_target(
            dataframe=split.train,
            feature_columns=feature_columns,
            target_column=target_column,
        )
    )

    x_validation_raw, y_validation = (
        _extract_features_and_target(
            dataframe=split.validation,
            feature_columns=feature_columns,
            target_column=target_column,
        )
    )

    x_test_raw, y_test = (
        _extract_features_and_target(
            dataframe=split.test,
            feature_columns=feature_columns,
            target_column=target_column,
        )
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Training split must contain both classes"
        )

    scaler = StandardScaler()

    x_train = scaler.fit_transform(
        x_train_raw
    )

    x_validation = scaler.transform(
        x_validation_raw
    )

    x_test = scaler.transform(
        x_test_raw
    )

    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_seed,
    )

    model.fit(
        x_train,
        y_train,
    )

    train_predictions = model.predict(
        x_train
    )

    validation_predictions = model.predict(
        x_validation
    )

    test_predictions = model.predict(
        x_test
    )

    train_probabilities = model.predict_proba(
        x_train
    )[:, 1]

    validation_probabilities = model.predict_proba(
        x_validation
    )[:, 1]

    test_probabilities = model.predict_proba(
        x_test
    )[:, 1]

    return BaselineTrainingResult(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,

        x_train=x_train,
        x_validation=x_validation,
        x_test=x_test,

        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,

        train_predictions=train_predictions,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,

        train_probabilities=(
            train_probabilities
        ),
        validation_probabilities=(
            validation_probabilities
        ),
        test_probabilities=(
            test_probabilities
        ),
    )