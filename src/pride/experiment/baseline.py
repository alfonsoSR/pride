from typing import TYPE_CHECKING, Any
from ..logger import log
from astropy import time
import numpy as np
from .. import coordinates as coord
from scipy import interpolate

if TYPE_CHECKING:
    from .station import Station
    from .observation import Observation
    from ..displacements import Displacement


class Baseline:
    """VLBI baseline

    A baseline is a pair of stations that have observations of different sources associated with them

    :param center: Station object representing the phase center
    :param station: Station object representing the other station in the baseline
    :param observations: List of observations associated with the baseline
    :param tstamps: Time stamps of the observations associated with the baseline
    :param id: Unique identifier for the baseline (center-station)
    """

    __slots__ = ("center", "station", "observations", "tstamps", "id")

    def __init__(
        self,
        center: "Station",
        station: "Station",
        observations: list["Observation"],
    ) -> None:

        # Merge timestamps of all the observations
        tstamps_no_location = time.Time(
            [observation.tstamps for observation in observations],
            scale="utc",
            location=None,
        ).sort()
        assert isinstance(tstamps_no_location, time.Time)
        tstamps = time.Time(
            tstamps_no_location,
            location=station.tectonic_corrected_location(tstamps_no_location),
        )

        # Attributes
        self.center = center
        self.station = station
        self.observations = observations
        self.tstamps = tstamps
        self.id = f"{center.name}-{station.name}"

        return None

    def update_station_with_geophysical_displacements(
        self,
        displacement_models: list["Displacement"],
        augmented_tstamps: time.Time,
        augmented_eops: np.ndarray,
        augmented_icrf2itrf: np.ndarray,
        augmented_seu2itrf: np.ndarray,
        augmented_lat: np.ndarray,
        augmented_lon: np.ndarray,
        icrf2itrf: np.ndarray,
        dot_icrf2itrf: np.ndarray,
    ) -> None:

        log.debug(
            f"Updating {self.station.name} station with geophysical "
            "displacements"
        )

        # Shared resources
        shared_resources = {
            "station_names": self.station.possible_names,
            "eops": augmented_eops,
            "icrf2itrf": augmented_icrf2itrf,
            "seu2itrf": augmented_seu2itrf,
            "lat": augmented_lat,
            "lon": augmented_lon,
            "xsta_itrf": self.station.location(augmented_tstamps),
        }
        log.debug("Right after shared resources")

        # Station position at reference epochs [Tectonic corrected]
        xsta_itrf = shared_resources["xsta_itrf"][1::3]
        xsta_icrf = (
            icrf2itrf.swapaxes(-1, -2) @ xsta_itrf[:, :, None]
        ).squeeze()

        # Station velocity at reference epochs
        vsta_itrf = np.zeros_like(xsta_itrf)
        vsta_icrf = (
            dot_icrf2itrf.swapaxes(-1, -2) @ xsta_itrf[:, :, None]
        ).squeeze()

        # Update positions and velocities with displacements
        for model in displacement_models:

            # Calculate augmented displacement in ICRF and ITRF
            resources = model.load_resources(
                augmented_tstamps, shared_resources
            )
            a_dx_itrf = model.calculate(augmented_tstamps, resources)
            a_dx_icrf = (
                augmented_icrf2itrf.swapaxes(-1, -2) @ a_dx_itrf[:, :, None]
            ).squeeze()

            # Update positions and velocities
            xsta_itrf += a_dx_itrf[1::3]
            xsta_icrf += a_dx_icrf[1::3]
            vsta_itrf += (a_dx_itrf[2::3] - a_dx_itrf[0::3]) * 0.5
            vsta_icrf += (a_dx_icrf[2::3] - a_dx_icrf[0::3]) * 0.5

        # Generate interpolation polynomials
        self.station._interp_xsta_itrf = interpolate.interp1d(
            self.tstamps.jd, xsta_itrf, kind="cubic", axis=0
        )
        self.station._interp_xsta_icrf = interpolate.interp1d(
            self.tstamps.jd, xsta_icrf, kind="cubic", axis=0
        )
        self.station._interp_vsta_itrf = interpolate.interp1d(
            self.tstamps.jd, vsta_itrf, kind="cubic", axis=0
        )
        self.station._interp_vsta_icrf = interpolate.interp1d(
            self.tstamps.jd, vsta_icrf, kind="cubic", axis=0
        )

        # Update geophysical corrections flag for station
        self.station.has_geophysical_corrections = True

        return None
