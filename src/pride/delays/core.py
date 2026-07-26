from abc import abstractmethod, ABCMeta
from typing import TYPE_CHECKING, Any
from ..logger import log

if TYPE_CHECKING:
    from ..experiment.experiment import Experiment
    from ..experiment.observation import Observation


class Delay(metaclass=ABCMeta):
    """Base class for implementation of delay models

    :param name: Unique name that identifies the delay model
    :param exp: Experiment object
    :param config: Section of the configuration file associated with the delay
    :param resources: Private container to be used internally when loading resources
    """

    def __init__(self, exp: "Experiment") -> None:
        """Initialize delay model from experiment"""

        # Get name from name of the class
        self.name = type(self).__name__

        self.exp = exp
        self.config: dict[str, Any] = self.exp.setup.delays[self.name]
        self.resources: dict[str, Any] = {}
        self.loaded_resources: dict[str, Any] = {}

        # Ensure resources required to calculate the delay
        log.debug(f"Ensuring resources for {self.name} delay")
        self.ensure_resources()

        # Load resources
        log.debug(f"Loading resources for {self.name} delay")
        self.loaded_resources = self.load_resources()

        return None

    @abstractmethod
    def ensure_resources(self) -> None: ...

    @abstractmethod
    def load_resources(self) -> dict[str, Any]: ...

    @abstractmethod
    def calculate(self, obs: "Observation") -> Any: ...

    def calculate_with_logging(self, obs: "Observation") -> Any:

        log.debug(
            f"Calculating {self.name} delay of {obs.source.name} from "
            f"{obs.station.name}"
        )
        return self.calculate(obs)
