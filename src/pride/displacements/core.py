from typing import Any, TYPE_CHECKING
from ..logger import log
from abc import abstractmethod, ABCMeta
import numpy as np

if TYPE_CHECKING:
    from astropy import time


class Displacement(metaclass=ABCMeta):
    """Base class for implementation of geophysical displacements

    Used to correct the position of stations for geophysical phenomena
    """

    name: str = NotImplemented
    requires_spice: bool = NotImplemented

    def __init__(self) -> None:

        # self.config: dict[str, Any] = self.exp.setup.displacements[self.name]
        self._resources: dict[str, Any] = {}
        self.resources: dict[str, Any] = {}
        log.debug(f"Ensuring resources for {self.name} displacement")
        self.ensure_resources()
        return None

    @abstractmethod
    def ensure_resources(self) -> None: ...

    @abstractmethod
    def load_resources(
        self, epoch: "time.Time", shared: dict[str, Any]
    ) -> dict[str, Any]: ...

    @abstractmethod
    def calculate(
        self, epoch: "time.Time", resources: dict[str, Any]
    ) -> np.ndarray: ...
