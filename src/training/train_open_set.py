from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.detection.open_set import (
    quantile_threshold,
)
from src.training.prepare_dataset import (
    PreparedDataset,
)
from src.training.split_data import (
    DatasetSplit,
)


@dataclass(frozen=True)
class OpenSetTrainingResult:
    classifier: LogisticRegression
    scaler: StandardScaler
    anomaly_detector: IsolationForest
    feature_columns: tuple[str, ...]
    anomaly_feature_columns: tuple[str, ...]
    anomaly_feature_indices: tuple[int, ...]
    classes: tuple[str, ...]
    confidence_threshold: float
    anomaly_threshold: float
    y_validation: np.ndarray
    y_test: np.ndarray
    validation_predictions: np.ndarray
    test_predictions: np.ndarray
    validation_confidence: np.ndarray
    test_confidence: np.ndarray
    validation_anomaly_scores: np.ndarray
    test_anomaly_scores: np.ndarray


def _extract_features(
    dataframe: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    missing = (
        set(feature_columns)
        - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Dataset split is missing features: "
            f"{sorted(missing)}"
        )

    features = dataframe[
        list(feature_columns)
    ].apply(
        pd.to_numeric,
        errors="coerce",
    )

    values = features.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Open-set feature matrix contains "
            "missing or non-finite values"
        )

    return values


def _extract_labels(
    dataframe: pd.DataFrame,
) -> np.ndarray:
    if "attack_type" not in dataframe.columns:
        raise ValueError(
            "Dataset split is missing attack_type"
        )

    labels = (
        dataframe["attack_type"]
        .astype(str)
        .to_numpy()
    )

    if len(labels) == 0:
        raise ValueError(
            "Open-set labels must not be empty"
        )

    return labels


def train_open_set_detector(
    prepared: PreparedDataset,
    split: DatasetSplit,
    anomaly_feature_columns: tuple[str, ...],
    confidence_lower_quantile: float = 0.01,
    anomaly_upper_quantile: float = 0.99,
    class_weight: str | dict[str, float] | None = "balanced",
    max_iter: int = 2000,
    isolation_estimators: int = 300,
    random_seed: int = 42,
) -> OpenSetTrainingResult:
    """
    Train a known-class classifier and normal-only anomaly detector.

    Thresholds are selected exclusively from the validation split.
    """
    if max_iter <= 0:
        raise ValueError(
            "max_iter must be greater than zero"
        )

    if isolation_estimators <= 0:
        raise ValueError(
            "isolation_estimators must be greater than zero"
        )

    if not anomaly_feature_columns:
        raise ValueError(
            "anomaly_feature_columns must not be empty"
        )

    feature_columns = prepared.feature_columns

    missing_anomaly_features = (
        set(anomaly_feature_columns)
        - set(feature_columns)
    )

    if missing_anomaly_features:
        raise ValueError(
            "Anomaly features are not present in the "
            "classifier feature set: "
            f"{sorted(missing_anomaly_features)}"
        )

    anomaly_indices = tuple(
        feature_columns.index(feature)
        for feature in anomaly_feature_columns
    )

    x_train_raw = _extract_features(
        split.train,
        feature_columns,
    )
    x_validation_raw = _extract_features(
        split.validation,
        feature_columns,
    )
    x_test_raw = _extract_features(
        split.test,
        feature_columns,
    )

    y_train = _extract_labels(
        split.train
    )
    y_validation = _extract_labels(
        split.validation
    )
    y_test = _extract_labels(
        split.test
    )

    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "Known-class training requires at least two classes"
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

    classifier = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_seed,
    )
    classifier.fit(
        x_train,
        y_train,
    )

    validation_probabilities = (
        classifier.predict_proba(
            x_validation
        )
    )
    test_probabilities = (
        classifier.predict_proba(
            x_test
        )
    )

    validation_predictions = (
        classifier.classes_[
            validation_probabilities.argmax(
                axis=1
            )
        ]
    )
    test_predictions = (
        classifier.classes_[
            test_probabilities.argmax(
                axis=1
            )
        ]
    )

    validation_confidence = (
        validation_probabilities.max(
            axis=1
        )
    )
    test_confidence = (
        test_probabilities.max(
            axis=1
        )
    )

    correct_validation = (
        validation_predictions
        == y_validation
    )

    if not correct_validation.any():
        raise ValueError(
            "No correctly classified validation rows "
            "are available for confidence calibration"
        )

    confidence_threshold = (
        quantile_threshold(
            validation_confidence[
                correct_validation
            ],
            confidence_lower_quantile,
        )
    )

    normal_train = (
        y_train == "none"
    )
    normal_validation = (
        y_validation == "none"
    )

    if not normal_train.any():
        raise ValueError(
            "Training split contains no normal samples"
        )

    if not normal_validation.any():
        raise ValueError(
            "Validation split contains no normal samples"
        )

    anomaly_detector = IsolationForest(
        n_estimators=isolation_estimators,
        contamination="auto",
        random_state=random_seed,
        n_jobs=-1,
    )
    anomaly_detector.fit(
        x_train[
            normal_train
        ][:, anomaly_indices]
    )

    validation_anomaly_scores = -(
        anomaly_detector.score_samples(
            x_validation[
                :,
                anomaly_indices,
            ]
        )
    )
    test_anomaly_scores = -(
        anomaly_detector.score_samples(
            x_test[
                :,
                anomaly_indices,
            ]
        )
    )

    anomaly_threshold = quantile_threshold(
        validation_anomaly_scores[
            normal_validation
        ],
        anomaly_upper_quantile,
    )

    return OpenSetTrainingResult(
        classifier=classifier,
        scaler=scaler,
        anomaly_detector=anomaly_detector,
        feature_columns=feature_columns,
        anomaly_feature_columns=(
            anomaly_feature_columns
        ),
        anomaly_feature_indices=(
            anomaly_indices
        ),
        classes=tuple(
            str(label)
            for label in classifier.classes_
        ),
        confidence_threshold=(
            confidence_threshold
        ),
        anomaly_threshold=(
            anomaly_threshold
        ),
        y_validation=y_validation,
        y_test=y_test,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,
        validation_confidence=(
            validation_confidence
        ),
        test_confidence=test_confidence,
        validation_anomaly_scores=(
            validation_anomaly_scores
        ),
        test_anomaly_scores=(
            test_anomaly_scores
        ),
    )


def score_open_set_rows(
    result: OpenSetTrainingResult,
    dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return predicted known labels, confidence and anomaly scores.
    """
    raw_features = _extract_features(
        dataframe,
        result.feature_columns,
    )
    features = result.scaler.transform(
        raw_features
    )

    probabilities = (
        result.classifier.predict_proba(
            features
        )
    )
    predictions = (
        result.classifier.classes_[
            probabilities.argmax(
                axis=1
            )
        ]
    )
    confidence = probabilities.max(
        axis=1
    )
    anomaly_scores = -(
        result.anomaly_detector.score_samples(
            features[
                :,
                result.anomaly_feature_indices,
            ]
        )
    )

    return (
        predictions,
        confidence,
        anomaly_scores,
    )


def save_open_set_artifacts(
    result: OpenSetTrainingResult,
    classifier_path: str | Path,
    scaler_path: str | Path,
    anomaly_detector_path: str | Path,
    metadata_path: str | Path,
) -> None:
    paths = [
        Path(classifier_path),
        Path(scaler_path),
        Path(anomaly_detector_path),
        Path(metadata_path),
    ]

    for path in paths:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    joblib.dump(
        result.classifier,
        paths[0],
    )
    joblib.dump(
        result.scaler,
        paths[1],
    )
    joblib.dump(
        result.anomaly_detector,
        paths[2],
    )

    metadata = {
        "feature_columns": list(
            result.feature_columns
        ),
        "anomaly_feature_columns": list(
            result.anomaly_feature_columns
        ),
        "classes": list(
            result.classes
        ),
        "confidence_threshold": (
            result.confidence_threshold
        ),
        "anomaly_threshold": (
            result.anomaly_threshold
        ),
        "normal_label": "none",
        "unknown_label": "unknown",
    }

    with paths[3].open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )
