import numpy as np
import pandas as pd

from src.training.prepare_dataset import (
    PreparedDataset,
)
from src.training.split_data import (
    DatasetSplit,
)
from src.training.train_open_set import (
    score_open_set_rows,
    train_open_set_detector,
)


FEATURES = (
    "voltage",
    "voltage_deviation",
)


def build_frame(
    offset: float,
) -> pd.DataFrame:
    rows = []

    for label, centre in (
        ("none", 0.0),
        ("constant", 5.0),
        ("random", -5.0),
    ):
        for index in range(12):
            rows.append(
                {
                    "source_file": (
                        f"{label}_{offset}.csv"
                    ),
                    "device_id": "meter_02",
                    "sequence_number": index,
                    "attack_type": label,
                    "is_attack": int(
                        label != "none"
                    ),
                    "voltage": (
                        centre
                        + offset
                        + index * 0.01
                    ),
                    "voltage_deviation": (
                        abs(centre)
                        + index * 0.01
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_inputs() -> tuple[
    PreparedDataset,
    DatasetSplit,
]:
    train = build_frame(0.0)
    validation = build_frame(0.1)
    test = build_frame(-0.1)
    combined = pd.concat(
        [
            train,
            validation,
            test,
        ],
        ignore_index=True,
    )

    prepared = PreparedDataset(
        dataframe=combined,
        feature_columns=FEATURES,
        target_column="is_attack",
        group_column="source_file",
    )
    split = DatasetSplit(
        train=train,
        validation=validation,
        test=test,
        train_groups=(
            "train",
        ),
        validation_groups=(
            "validation",
        ),
        test_groups=(
            "test",
        ),
    )

    return prepared, split


def test_train_open_set_detector() -> None:
    prepared, split = build_inputs()

    result = train_open_set_detector(
        prepared=prepared,
        split=split,
        anomaly_feature_columns=(
            "voltage",
        ),
        isolation_estimators=20,
        random_seed=42,
    )

    assert set(result.classes) == {
        "none",
        "constant",
        "random",
    }
    assert (
        0
        <= result.confidence_threshold
        <= 1
    )
    assert np.isfinite(
        result.anomaly_threshold
    )
    assert len(
        result.test_predictions
    ) == len(
        split.test
    )


def test_score_open_set_rows() -> None:
    prepared, split = build_inputs()
    result = train_open_set_detector(
        prepared=prepared,
        split=split,
        anomaly_feature_columns=(
            "voltage",
        ),
        isolation_estimators=20,
    )

    predictions, confidence, anomaly = (
        score_open_set_rows(
            result,
            split.test,
        )
    )

    assert len(predictions) == len(
        split.test
    )
    assert len(confidence) == len(
        split.test
    )
    assert len(anomaly) == len(
        split.test
    )
