from typing import TYPE_CHECKING, Any
import datetime
from astropy import time
from ..logger import log
import numpy as np

if TYPE_CHECKING:
    from .experiment import Experiment
    from ..source import Source
    from ..types import Band
    from .baseline import Baseline


class Observation:
    """Observation of a source

    The time stamps are stored as an astropy Time object including information about the coordinates of the station for which the observation was performed. The coordinates of the station are obtained by correcting the location at a reference epoch (loaded from data file) for tectonic motion.

    :param source: Source object representing the target
    :param band: Frequency band in which the source was detected
    :param tstamps: Collection of epochs in which the source was detected
    """

    __slots__ = (
        "source",
        "band",
        "obs_freq",
        "tstamps",
        "baseline",
        "station",
        "exp",
        "icrf2itrf",
        "seu2itrf",
        "dot_icrf2itrf",
        "source_az",
        "source_el",
        "source_ra",
        "source_dec",
        "tx_epochs",
        "delays",
        "delay_dict",
        "has_delays",
    )

    def __init__(
        self,
        baseline: "Baseline",
        source: "Source",
        band: "Band",
        tstamps: list[datetime.datetime],
    ) -> None:
        """Initialize observation

        :param Baseline: Baseline from which the observation was performed
        :param source: Source object representing the target
        :param band: Frequency band in which the source was detected
        :param tstamps: List of UTC epochs in which the source was detected
        :param exp: Experiment object to which the observation belongs
        """

        # Input-based properties
        self.source = source
        self.band = band
        _tstamps = time.Time(sorted(tstamps), scale="utc")
        self.tstamps = time.Time(
            _tstamps,
            scale="utc",
            location=baseline.station.tectonic_corrected_location(_tstamps),
        )
        self.baseline = baseline
        self.station = baseline.station

        # Optional properties
        self.obs_freq: "np.ndarray" = NotImplemented
        self.exp: "Experiment" = NotImplemented
        self.icrf2itrf: "np.ndarray" = NotImplemented
        self.seu2itrf: "np.ndarray" = NotImplemented
        self.dot_icrf2itrf: "np.ndarray" = NotImplemented
        self.source_az: "np.ndarray" = NotImplemented
        self.source_el: "np.ndarray" = NotImplemented
        self.source_ra: "np.ndarray" = NotImplemented
        self.source_dec: "np.ndarray" = NotImplemented
        self.tx_epochs: "time.Time" = NotImplemented
        self.delays: "np.ndarray" = NotImplemented
        self.delay_dict: "dict" = NotImplemented
        self.has_delays: bool = False

        return None

    @staticmethod
    def from_experiment(
        baseline: "Baseline",
        source: "Source",
        band: "Band",
        tstamps: list[datetime.datetime],
        experiment: "Experiment",
    ) -> "Observation":

        # Initialize normal observation
        observation = Observation(baseline, source, band, tstamps)

        # Add experiment as attribute
        observation.exp = experiment

        # # Add observation frequency
        # if isinstance(observation.source, FarFieldSource):

        #     freq = experiment.setup.general["reference_frequency"]
        #     observation.obs_freq = freq * np.ones_like(observation.tstamps.jd)  # type: ignore

        # elif isinstance(observation.source, NearFieldSource):

        #     with io.internal_file(
        #         f"ramp.{observation.source.spice_id}"
        #     ).open() as f:
        #         for line in f:
        #             print(line)

        #     exit(0)

        # else:
        #     raise NotImplementedError("Not possible")

        return observation

    def update_with_source_coordinates(self) -> None:
        """Update with source coordinates at timestamps"""

        (
            self.source_az,
            self.source_el,
            self.source_ra,
            self.source_dec,
            _tx_epochs,
        ) = self.source.spherical_coordinates(self)

        if _tx_epochs is not None:
            self.tx_epochs = _tx_epochs

        return None

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            log.error(f"Attribute {name} is not set for {self.baseline.id}")
            exit(1)
        return val

    def calculate_delays(self) -> None:

        if self.has_delays:
            log.error(
                "Attempted to calculate delays twice for observation of"
                f" {self.source.name} from {self.station.name}"
            )
            exit(1)

        log.debug(
            f"Calculating delays of {self.source.name} from {self.station.name}"
        )
        delay: np.ndarray = np.zeros_like(self.tstamps.jd)  # type: ignore
        self.delay_dict = {}
        for delay_model in self.exp.delay_models:
            val = delay_model.calculate(self)
            self.delay_dict[delay_model.name] = val
            delay += val
        self.delays = delay
        self.has_delays = True

        return None
