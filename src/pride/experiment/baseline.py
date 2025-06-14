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
    :param nobs: Number of observations associated with the baseline
    :param tstamps: Time stamps of the observations associated with the baseline
    :param exp: Experiment object to which the baseline belongs
    """

    # __slots__ = (
    #     "center",
    #     "station",
    #     "observations",
    #     "tstamps",
    #     "a_tstamps",
    #     "a_eops",
    #     "icrf2itrf",
    #     "a_icrf2itrf",
    #     "dot_icrf2itrf",
    #     "a_seu2itrf",
    #     "a_lat",
    #     "a_lon",
    # )

    def __init__(
        self,
        center: "Station",
        station: "Station",
        observations: list["Observation"],
        eops: "coord.EOP",
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

        # Augment time stamps with +/- 1 second around each epoch
        __augmented_tstamps: time.Time = (
            tstamps_no_location[:, None]
            + time.TimeDelta([-1, 0, 1], format="sec")[None, :]
        ).ravel()  # type: ignore
        a_tstamps = time.Time(
            __augmented_tstamps,
            scale="utc",
            location=station.tectonic_corrected_location(__augmented_tstamps),
        )
        assert a_tstamps.location is not None

        # Calculate geodetic coordinates of station
        a_geodetic = a_tstamps.location.to_geodetic("GRS80")
        a_lat = np.array(a_geodetic.lat.rad, dtype=float)
        a_lon = np.array(a_geodetic.lon.rad, dtype=float)

        # Calculate rotation matrices
        augmented_eops = eops.at_epoch(a_tstamps, unit="arcsec")
        # self.a_eops = eops.at_epoch(self.a_tstamps, unit="arcsec")
        a_icrf2itrf = coord.icrf2itrf(augmented_eops, a_tstamps)
        a_seu2itrf = coord.seu2itrf(a_lat, a_lon)

        # Calculate derivative of ICRF to ITRF rotation matrix
        dt_tdb: np.ndarray = (
            (a_tstamps.tdb[2::3] - a_tstamps.tdb[0::3])
            .to("s")  # type: ignore
            .value
        )
        diff_icrf2itrf = a_icrf2itrf[2::3] - a_icrf2itrf[0::3]
        dot_icrf2itrf = diff_icrf2itrf / dt_tdb[:, None, None]

        # Get attributes at observation epochs
        icrf2itrf = a_icrf2itrf[1::3]
        seu2itrf = a_seu2itrf[1::3]

        # Update observations with calculated data
        for observation in observations:

            # Get the index of the time stamps associated with the observation
            flags = np.sum(
                observation.tstamps[:, None] == tstamps[None, :],
                axis=0,
                dtype=bool,
            )

            observation.icrf2itrf = icrf2itrf[flags]
            observation.seu2itrf = seu2itrf[flags]
            observation.dot_icrf2itrf = dot_icrf2itrf[flags]

        # Attributes
        self.center = center
        self.station = station
        self.observations = observations
        self.tstamps = tstamps
        self.a_tstamps = a_tstamps
        self.a_eops = augmented_eops
        self.icrf2itrf = icrf2itrf
        self.a_icrf2itrf = a_icrf2itrf
        self.dot_icrf2itrf = dot_icrf2itrf
        self.a_seu2itrf = a_seu2itrf
        self.a_lat = a_lat
        self.a_lon = a_lon

        return None

    @property
    def id(self) -> str:
        return f"{self.center.name}-{self.station.name}"

    @property
    def nobs(self) -> int:
        return len(self.observations)

    def __str__(self) -> str:
        return self.id

    def update_station_with_geophysical_displacements(
        self, displacement_models: list["Displacement"]
    ) -> None:

        log.info(
            f"Updating {self.station.name} station with geophysical "
            "displacements"
        )

        # Shared resources
        shared_resources = {
            "station_names": self.station.possible_names,
            "eops": self.a_eops,
            "icrf2itrf": self.a_icrf2itrf,
            "seu2itrf": self.a_seu2itrf,
            "lat": self.a_lat,
            "lon": self.a_lon,
            "xsta_itrf": self.station.location(self.a_tstamps),
        }
        log.debug("Right after shared resources")

        # Station position at reference epochs [Tectonic corrected]
        xsta_itrf = shared_resources["xsta_itrf"][1::3]
        xsta_icrf = (
            self.icrf2itrf.swapaxes(-1, -2) @ xsta_itrf[:, :, None]
        ).squeeze()

        # Station velocity at reference epochs
        vsta_itrf = np.zeros_like(xsta_itrf)
        vsta_icrf = (
            self.dot_icrf2itrf.swapaxes(-1, -2) @ xsta_itrf[:, :, None]
        ).squeeze()

        # Update positions and velocities with displacements
        for model in displacement_models:

            # Calculate augmented displacement in ICRF and ITRF
            resources = model.load_resources(self.a_tstamps, shared_resources)
            a_dx_itrf = model.calculate(self.a_tstamps, resources)
            a_dx_icrf = (
                self.a_icrf2itrf.swapaxes(-1, -2) @ a_dx_itrf[:, :, None]
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
