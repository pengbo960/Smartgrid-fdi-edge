from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any


class ReplayBuffer:
    """
    Maintain a bounded history of complete message payloads.

    Replay attacks return a copied historical payload so that the
    original sequence number, timestamp and measurements are preserved.
    """

    def __init__(
        self,
        maximum_size: int = 100,
    ) -> None:
        if maximum_size <= 0:
            raise ValueError(
                "maximum_size must be greater than zero"
            )

        self.maximum_size = maximum_size

        self._messages: deque[
            dict[str, Any]
        ] = deque(
            maxlen=maximum_size
        )

    def store(
        self,
        payload: dict[str, Any],
    ) -> None:
        """
        Store an independent copy of one legitimate payload.
        """
        if not isinstance(payload, dict):
            raise TypeError(
                "payload must be a dictionary"
            )

        if not payload:
            raise ValueError(
                "payload must not be empty"
            )

        self._messages.append(
            deepcopy(payload)
        )

    def replay(
        self,
        lag_steps: int,
    ) -> dict[str, Any]:
        """
        Return a copy of the payload stored lag_steps messages ago.

        lag_steps=1 returns the most recently stored payload.
        lag_steps=10 returns the tenth most recent stored payload.
        """
        if lag_steps <= 0:
            raise ValueError(
                "lag_steps must be greater than zero"
            )

        if len(self._messages) < lag_steps:
            raise ValueError(
                "Not enough messages in replay buffer: "
                f"required {lag_steps}, "
                f"available {len(self._messages)}"
            )

        return deepcopy(
            self._messages[-lag_steps]
        )

    def size(self) -> int:
        return len(
            self._messages
        )

    def clear(self) -> None:
        self._messages.clear()