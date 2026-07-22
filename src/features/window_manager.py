from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any


class WindowManager:
    """
    Maintain an independent fixed-size history window for each device.
    """

    def __init__(self, window_size: int = 20) -> None:
        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than zero"
            )

        self.window_size = window_size

        self._windows: dict[
            str,
            deque[dict[str, Any]],
        ] = {}

    def update(
        self,
        row: dict[str, Any],
    ) -> None:
        """
        Add one message row to its device-specific history.
        """
        if "device_id" not in row:
            raise KeyError(
                "row must contain device_id"
            )

        device_id = str(
            row["device_id"]
        ).strip()

        if not device_id:
            raise ValueError(
                "device_id must not be empty"
            )

        if device_id not in self._windows:
            self._windows[device_id] = deque(
                maxlen=self.window_size
            )

        self._windows[device_id].append(
            deepcopy(row)
        )

    def get_history(
        self,
        device_id: str,
    ) -> list[dict[str, Any]]:
        """
        Return a copy of the stored history for one device.
        """
        device_id = str(device_id).strip()

        if not device_id:
            raise ValueError(
                "device_id must not be empty"
            )

        history = self._windows.get(
            device_id
        )

        if history is None:
            return []

        return [
            deepcopy(row)
            for row in history
        ]

    def get_previous(
        self,
        device_id: str,
    ) -> dict[str, Any] | None:
        """
        Return the most recent stored message for one device.
        """
        history = self._windows.get(
            str(device_id).strip()
        )

        if not history:
            return None

        return deepcopy(history[-1])

    def get_previous_n(
        self,
        device_id: str,
        count: int,
    ) -> list[dict[str, Any]]:
        """
        Return up to the most recent count messages for one device.
        """
        if count <= 0:
            raise ValueError(
                "count must be greater than zero"
            )

        history = self.get_history(device_id)

        return history[-count:]

    def history_length(
        self,
        device_id: str,
    ) -> int:
        """
        Return the number of stored messages for one device.
        """
        history = self._windows.get(
            str(device_id).strip()
        )

        if history is None:
            return 0

        return len(history)

    def device_ids(self) -> list[str]:
        """
        Return all device IDs currently tracked.
        """
        return sorted(
            self._windows.keys()
        )

    def clear_device(
        self,
        device_id: str,
    ) -> None:
        """
        Remove the stored history for one device.
        """
        self._windows.pop(
            str(device_id).strip(),
            None,
        )

    def clear_all(self) -> None:
        """
        Remove all stored device histories.
        """
        self._windows.clear()