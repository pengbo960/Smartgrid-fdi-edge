import pytest

from src.evaluation.edge_warmup import partition_device_warmup


def test_partition_device_warmup_balances_devices() -> None:
    rows = [
        {"device_id": device, "sequence": sequence}
        for device in ("meter_01", "meter_02")
        for sequence in range(4)
    ]
    warmup, measured, counts = partition_device_warmup(rows, 2)
    assert len(warmup) == 4
    assert len(measured) == 4
    assert counts == {"meter_01": 2, "meter_02": 2}
    assert [row["sequence"] for row in measured] == [2, 3, 2, 3]


def test_partition_device_warmup_validates_inputs() -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        partition_device_warmup([], -1)
    with pytest.raises(ValueError, match="device_id"):
        partition_device_warmup([{"value": 1}], 1)
