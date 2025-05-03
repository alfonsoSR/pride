from abc import abstractmethod, ABCMeta
from typing import TYPE_CHECKING, Any
from ..logger import log

if TYPE_CHECKING:
    from ..experiment.experiment import Experiment
    from ..experiment.observation import Observation


class Delay(metaclass=ABCMeta):
    """Base class for implementation of delay models

    :param name: Unique name that identifies the delay model
    :param etc: Configuration parameters not meant to be adjusted by users
    :param requires_spice: Whether the delay model requires SPICE kernels
    :param station_specific: Whether the resources required to calculate the delay are station-specific
    :param exp: Experiment object
    :param config: Section of the configuration file associated with the delay
    :param resources: Private container to be used internally when loading resources
    """

    name: str = NotImplemented
    etc: dict[str, Any] = NotImplemented
    requires_spice: bool = NotImplemented
    station_specific: bool = NotImplemented

    def __init__(self, exp: "Experiment") -> None:
        """Initialize delay model from experiment"""

        self.exp = exp
        self.config: dict[str, Any] = self.exp.setup.delays[self.name]
        self.resources: dict[str, Any] = {}
        self.loaded_resources: dict[str, Any] = {}

        # Ensure resources required to calculate the delay
        log.debug(f"Ensuring resources for {self.name} delay")
        self.ensure_resources()

        # Load resources
        self.loaded_resources = self.load_resources()

        return None

    @abstractmethod
    def ensure_resources(self) -> None: ...

    @abstractmethod
    def load_resources(self) -> dict[str, Any]: ...

    @abstractmethod
    def calculate(self, obs: "Observation") -> Any: ...
