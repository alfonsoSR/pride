raise DeprecationWarning("Only kept for reference during development")

from typing import TYPE_CHECKING, Any
from ..types import SourceType
from astropy import coordinates, time
from nastro import types as nt
from ..logger import log
import numpy as np
import spiceypy as spice
from ..constants import J2000


if TYPE_CHECKING:
    from .experiment import Experiment
    from ..io import VexContent
    from .station import Station


class Source:

    __slots__ = (
        "name",
        "type",
        "observed_ra",
        "observed_dec",
        "observed_ks",
        "spice_id",
        "exp",
    )

    def __init__(
        self,
        name: str,
        source_type: SourceType,
        ra: float,
        dec: float,
        spice_id: str | None,
    ) -> None:

        self.name = name
        self.type = source_type
        _coordinates = coordinates.SkyCoord(ra, dec, unit="rad", frame="icrs")
        self.observed_ra: np.ndarray = _coordinates.ra.rad  # type: ignore
        self.observed_dec: np.ndarray = _coordinates.dec.rad  # type: ignore
        self.observed_ks = np.array(
            [
                np.cos(self.observed_ra) * np.cos(self.observed_dec),
                np.sin(self.observed_ra) * np.cos(self.observed_dec),
                np.sin(self.observed_dec),
            ]
        )

        # Optional attributes
        self.spice_id: str = (
            spice_id if spice_id is not None else NotImplemented
        )
        self.exp: "Experiment" = NotImplemented

        # self.name = name
        # self.type = source_type
        # self.coordinates = coordinates.SkyCoord(
        #     ra, dec, unit="rad", frame="icrs"
        # )
        # setattr(self, "_spice_id", spice_id)
        # setattr(self, "_exp", None)
        # setattr(self, "__az", None)
        # setattr(self, "__el", None)

        return None

    def __getattribute__(self, name: str) -> Any:

        val = super().__getattribute__(name)
        if val is NotImplemented:
            raise AttributeError(f"Attribute {name} not set for {self.name}")
        return val

    # @property
    # def exp(self) -> "Experiment":

    #     if getattr(self, "_exp") is None:
    #         log.error(f"Experiment not set for {self.name} source")
    #         exit(1)
    #     return getattr(self, "_exp")

    # @property
    # def spice_id(self) -> str:

    #     if getattr(self, "_spice_id") is None:
    #         log.error(f"Spice ID not set for {self.name} source")
    #         exit(1)
    #     return getattr(self, "_spice_id")

    @staticmethod
    def from_experiment(name: str, experiment: "Experiment") -> "Source":

        # Read source information from VEX file
        if name not in experiment.vex["SOURCE"]:
            log.error(
                f"Failed to generate source: {name} not found in VEX file"
            )
            exit(1)
        source_info = experiment.vex["SOURCE"][name]

        # Read coordinates from VEX
        if source_info["ref_coord_frame"] != "J2000":
            raise NotImplementedError(
                f"Failed to generate source object for {name}: "
                f"Coordinate frame {source_info['ref_coord_frame']} not supported"
            )
        coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )

        # Identify source type and add spice ID if necessary
        source_type = Source.__find_type(source_info, name)
        spice_id: str | None = None
        if source_type == SourceType.NearField:
            spice_id = experiment.setup.general["target"]

        # Generate object and add experiment as attribute
        source = Source(
            name,
            source_type,
            coords.ra.rad,  # type: ignore
            coords.dec.rad,  # type: ignore
            spice_id,
        )
        source.exp = experiment

        return source

    # @property
    # def K_s(self) -> np.ndarray:
    #     """Unit vector in the direction of the source"""

    #     ra: np.ndarray = self.coordinates.ra  # type: ignore
    #     dec: np.ndarray = self.coordinates.dec  # type: ignore

    #     return np.array(
    #         [
    #             np.cos(ra) * np.cos(dec),
    #             np.sin(ra) * np.cos(dec),
    #             np.sin(dec),
    #         ]
    #     )

    @staticmethod
    def __find_type(source_info: "VexContent", _source_name: str) -> SourceType:

        if "source_type" not in source_info:
            raise ValueError(
                f"Failed to generate source object for {_source_name}: "
                "Missing type information in VEX file."
            )

        source_type = source_info["source_type"]

        if source_type == "calibrator":
            return SourceType.FarField
        elif source_type == "target":
            return SourceType.NearField
        else:
            raise TypeError(
                f"Failed to generate source object for {_source_name}: "
                f"Invalid type {source_type}"
            )

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"

    # @property
    # def az(self) -> np.ndarray:

    #     if getattr(self, "__az") is None:
    #         log.error(f"Azimuth not calculated for {self.name} source")
    #         exit(1)
    #     return getattr(self, "__az")

    # @property
    # def el(self) -> np.ndarray:

    #     if getattr(self, "__el") is None:
    #         log.error(f"Elevation not calculated for {self.name} source")
    #         exit(1)
    #     return getattr(self, "__el")

    def gcrf_position_at_tx(self, station: "Station", epochs: "time.Time"):
        """Non-aberrated GCRF position at TX epochs

        :param station: Station from which the source is observed
        :param epochs: Observation epochs
        """

        if self.type == SourceType.FarField:
            log.error(
                f"Failed to calculate position for {self.name}: "
                "Source is a calibrator"
            )
            exit(1)

        clight = spice.clight() * 1e3

        # Calculate station coordinates at observation epochs
        xsta_gcrf = station.location(epochs, frame="icrf")
        s_sta_gcrf = nt.CartesianState(*xsta_gcrf.T, *(0.0 * xsta_gcrf.T))

        # Intialize light-time between target and station
        et: np.ndarray = (epochs.tdb - J2000.tdb).to("s").value  # type: ignore
        xsc_gcrf = (
            np.array(
                spice.spkpos(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
            )
            * 1e3
        )
        lt_0 = np.linalg.norm(xsc_gcrf - xsta_gcrf, axis=1) / clight
        epoch_tx0 = epochs.tdb - time.TimeDelta(lt_0, format="sec")

        # Initialize light-time container
        lt_i = np.zeros_like(lt_0)
        ni = 0
        precision = 1e-12
        nmax = 3

        # Get GM of celestial bodies
        bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )

        while np.any(np.abs(lt_0 - lt_i) > precision) and (ni < nmax):

            # Update light-time
            lt_i = lt_0
            epoch_tx = epochs.tdb - time.TimeDelta(lt_i, format="sec")
            dt = (epoch_tx.tdb - epoch_tx0.tdb).to("s").value  # type: ignore

            # Calculate target's position wrt station at TX epoch
            et = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
            s_sc_earth_tx = nt.CartesianState(
                *np.array(
                    spice.spkezr(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
                ).T
                * 1e3
            )
            s_sta_sc_tx = s_sta_gcrf - s_sc_earth_tx
            r01 = s_sta_sc_tx.r_vec.T
            r01_mag = s_sta_sc_tx.r_mag
            r0 = s_sc_earth_tx.r_vec.T
            v0 = s_sc_earth_tx.v_vec.T

            # Get state of celestial bodies at TX epoch
            bodies_cstate = np.array(
                [
                    np.array(spice.spkezr(body, et, "J2000", "NONE", "SSB")[0])
                    * 1e3
                    for body in bodies
                ]
            )
            rb = bodies_cstate[:, :, :3]
            vb = bodies_cstate[:, :, 3:]

            # Calculate terms of relativistic correction
            r0b = r0[None, :, :] - (rb - dt[None, :, None] * vb)
            r0b_mag = np.linalg.norm(r0b, axis=-1)
            r1b = xsta_gcrf[None, :, :] - rb
            r1b_mag = np.linalg.norm(r1b, axis=-1)
            r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)

            # Calculate relativistic correction
            gmc = 2.0 * bodies_gm[:, None] / (clight * clight)
            rlt = np.sum(
                (gmc / clight)
                * np.log(
                    (r0b_mag + r1b_mag + r01b_mag + gmc)
                    / (r0b_mag + r1b_mag - r01b_mag + gmc)
                ),
                axis=0,
            )

            # Update light-time with relativistic correction
            denom = 1.0 - (np.sum((r01.T * v0.T), axis=0) / (clight * r01_mag))
            lt_0 -= (lt_0 - r01_mag / clight - rlt) / denom

            # Update TX epoch and counter
            epoch_tx = epochs.tdb - time.TimeDelta(lt_0, format="sec")
            ni += 1

        # Calculate position of target wrt station at TX epoch [Final]
        et: np.ndarray = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
        s_sc_earth_tx = nt.CartesianState(
            *np.array(
                spice.spkezr(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
            ).T
            * 1e3
        )

        return s_sc_earth_tx

    # def azel_from(self, station: "Station", epoch: "time.Time") -> Any:
    #     """Calculate azimuth and elevation as seen from a station

    #     :param station: Station from which the source is observed
    #     :param epoch: Observation epochs
    #     :return: Azimuth and elevation of the source as numpy arrays
    #     """

    #     if self.type == SourceType.NearField:
    #         self.__spacecraft_azel(station, epoch)
    #     else:
    #         print(f"{self.name} is a calibrator")

    #     return None

    # def __spacecraft_azel(self, station: "Station", epoch: "time.Time"):

    #     clight = spice.clight() * 1e3

    #     # Calculate station coordinates at observation epochs
    #     xsta_gcrf = station.location(epoch, frame="icrf")
    #     sta_r_earth = xsta_gcrf
    #     cstate_sta = nt.CartesianState(*xsta_gcrf.T, *(0.0 * xsta_gcrf.T))

    #     # Initialize light-time between target and station
    #     et: np.ndarray = (epoch.tdb - J2000.tdb).to("s").value  # type: ignore
    #     xsc_gcrf = (
    #         np.array(
    #             spice.spkpos(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
    #         )
    #         * 1e3
    #     )
    #     lt_0 = np.linalg.norm(xsc_gcrf - xsta_gcrf, axis=1) / clight
    #     epoch_tx0 = epoch.tdb - time.TimeDelta(lt_0, format="sec")

    #     # Initialize light-time container
    #     lt_i = np.zeros_like(lt_0)
    #     ni = 0
    #     precision = 1e-12
    #     nmax = 3

    #     # Get GM of celestial bodies
    #     bodies = (
    #         "sun",
    #         "mercury",
    #         "venus",
    #         "earth",
    #         "moon",
    #         "mars",
    #         "jupiter_barycenter",
    #         "saturn_barycenter",
    #         "uranus_barycenter",
    #         "neptune_barycenter",
    #     )
    #     bodies_gm = (
    #         np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
    #         * 1e9
    #     )

    #     while np.any(np.abs(lt_0 - lt_i) > precision) and (ni < nmax):

    #         # Update light-time
    #         lt_i = lt_0
    #         epoch_tx = epoch.tdb - time.TimeDelta(lt_i, format="sec")
    #         dt = (epoch_tx.tdb - epoch_tx0.tdb).to("s").value  # type: ignore

    #         # Calculate target's position wrt station at TX epoch
    #         et = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
    #         s_sc_earth_tx = nt.CartesianState(
    #             *np.array(
    #                 spice.spkezr(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
    #             ).T
    #             * 1e3
    #         )
    #         s_sta_sc_tx = cstate_sta - s_sc_earth_tx
    #         r01 = s_sta_sc_tx.r_vec.T
    #         r01_mag = s_sta_sc_tx.r_mag
    #         r0 = s_sc_earth_tx.r_vec.T
    #         v0 = s_sc_earth_tx.v_vec.T

    #         # Get state of celestial bodies at TX epoch
    #         bodies_cstate = np.array(
    #             [
    #                 np.array(spice.spkezr(body, et, "J2000", "NONE", "SSB")[0])
    #                 * 1e3
    #                 for body in bodies
    #             ]
    #         )
    #         rb = bodies_cstate[:, :, :3]
    #         vb = bodies_cstate[:, :, 3:]

    #         # Calculate terms of relativistic correction
    #         r0b = r0[None, :, :] - (rb - dt[None, :, None] * vb)
    #         r0b_mag = np.linalg.norm(r0b, axis=-1)
    #         r1b = sta_r_earth[None, :, :] - rb
    #         r1b_mag = np.linalg.norm(r1b, axis=-1)
    #         r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)

    #         # Calculate relativistic correction
    #         gmc = 2.0 * bodies_gm[:, None] / (clight * clight)
    #         rlt = np.sum(
    #             (gmc / clight)
    #             * np.log(
    #                 (r0b_mag + r1b_mag + r01b_mag + gmc)
    #                 / (r0b_mag + r1b_mag - r01b_mag + gmc)
    #             ),
    #             axis=0,
    #         )

    #         # Update light-time with relativistic correction
    #         denom = 1.0 - (np.sum((r01.T * v0.T), axis=0) / (clight * r01_mag))
    #         lt_0 -= (lt_0 - r01_mag / clight - rlt) / denom

    #         # Update TX epoch and counter
    #         epoch_tx = epoch.tdb - time.TimeDelta(lt_0, format="sec")
    #         ni += 1

    #     # Calculate position of target wrt station at TX epoch
    #     et = (epoch_tx.tdb - J2000.tdb).to("s").value  # type: ignore
    #     s_sc_earth_tx = nt.CartesianState(
    #         *np.array(
    #             spice.spkezr(self.spice_id, et, "J2000", "NONE", "EARTH")[0]
    #         ).T
    #         * 1e3
    #     )
    #     s_sc_sta_tx = s_sc_earth_tx - cstate_sta

    #     print((epoch_tx.tdb - epoch.tdb).to("s").value)

    #     exit(0)

    #     return 0
