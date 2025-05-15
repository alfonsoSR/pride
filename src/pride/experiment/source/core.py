from typing import TYPE_CHECKING, Any
from astropy import time
from ...logger import log
import numpy as np
from abc import abstractmethod, ABCMeta


if TYPE_CHECKING:
    from ..experiment import Experiment
    from ..observation import Observation


class Source(metaclass=ABCMeta):
    """Base class for radio sources"""

    __slots__ = (
        "name",
        "observed_ra",
        "observed_dec",
        "observed_ks",
        "spice_id",
        "three_way_ramping",
        "one_way_ramping",
        "default_frequency",
        "is_nearfield",
        "is_farfield",
        "has_three_way_ramping",
        "has_one_way_ramping",
    )

    def __init__(self, name: str) -> None:

        self.name = name
        self.observed_ra: float = NotImplemented
        self.observed_dec: float = NotImplemented
        self.observed_ks: np.ndarray = NotImplemented
        self.spice_id: str = NotImplemented
        self.three_way_ramping: dict[str, Any] = NotImplemented
        self.one_way_ramping: dict[str, Any] = NotImplemented
        self.has_one_way_ramping: bool = False
        self.has_three_way_ramping: bool = False
        self.default_frequency: float = NotImplemented
        self.is_nearfield: bool = False
        self.is_farfield: bool = False

        return None

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            log.error(f"Attribute {name} not set for source {self.name}")
            exit(1)
        return val

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"

    @staticmethod
    @abstractmethod
    def from_experiment(exp: "Experiment", name: str) -> "Source": ...

    @abstractmethod
    def spherical_coordinates(
        self, obs: "Observation"
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, time.Time | None
    ]: ...
