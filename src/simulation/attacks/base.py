from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias


Measurements: TypeAlias = dict[str, float]


class Attack(ABC):
    """
    Base interface for all measurement manipulation attacks.
    """

    attack_name: str

    @abstractmethod
    def apply(
        self,
        measurements: Measurements,
        attack_step: int,
    ) -> Measurements:
        """
        Return a modified copy of the input measurements.

        The original measurement dictionary must not be changed.
        """
        raise NotImplementedError