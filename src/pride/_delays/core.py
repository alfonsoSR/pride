from abc import abstractmethod
from typing import TYPE_CHECKING, Any
from ..logger import log

if TYPE_CHECKING:
    from ..experiment.core import Experiment
    from astropy import time


class Delay:
    """Base class for implementation of delay models"""

    name: str = NotImplemented
    key: str = NotImplemented
    etc: dict[str, Any] = NotImplemented
    requires_spice: bool = NotImplemented

    def __init__(self, exp: "Experiment") -> None:
        """Initialize delay model from experiment"""

        self.exp = exp
        self.config: dict[str, Any] = self.exp.setup.delays[self.name]
        self.resources: dict[str, Any] = {}

        # Ensure resources required to calculate delays
        self.ensure_resources()
        log.debug(f"Resources for {self.name} delay are present")

        return None

    @abstractmethod
    def ensure_resources(self) -> None:
        log.critical(f"Method ensure_resources not implemented for {self.name} delay")
        exit(1)

    def load_resources(self) -> dict[str, Any]:

        log.debug(f"Loading resources for {self.name} delay")
        return self._load_resources()

    @abstractmethod
    def _load_resources(self) -> dict[str, Any]:
        log.critical(f"Method _load_resources not implemented for {self.name} delay")
        exit(1)

    @abstractmethod
    def calculate(self, observation: "Observation", epoch: "time.Time"):
        log.critical(f"Method calculate not implemented for {self.name} delay")
        exit(1)
