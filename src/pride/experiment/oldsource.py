from typing import TYPE_CHECKING, Any
from astropy import coordinates, time
from ..logger import log
import numpy as np
import spiceypy as spice
from ..constants import J2000, L_C
from abc import abstractmethod, ABCMeta
from .. import io

if TYPE_CHECKING:
    from .experiment import Experiment
    from ..io import VexContent
    from .observation import Observation
    from .station import Station


class Source(metaclass=ABCMeta):
    """Radio source"""

    def __init__(self, name: str, ra: float, dec: float) -> None:

        # Properties from input
        self.name = name
        _coords = coordinates.SkyCoord(ra, dec, unit="rad", frame="icrs")
        self.observed_ra: float = float(_coords.ra.rad)  # type: ignore
        self.observed_dec: float = float(_coords.dec.rad)  # type: ignore

        # Derived properties
        self.observed_ks: np.ndarray = np.array(
            [
                np.cos(self.observed_ra) * np.cos(self.observed_dec),
                np.sin(self.observed_ra) * np.cos(self.observed_dec),
                np.sin(self.observed_dec),
            ]
        )

        # Optional properties
        self.exp: "Experiment" = NotImplemented
        self.spice_id: str = NotImplemented

        return None

    def __getattribute__(self, name: str) -> Any:
        val = super().__getattribute__(name)
        if val is NotImplemented:
            log.error(f"Attribute {name} is not set for source {self.name}")
            exit(1)
        return val

    def __repr__(self) -> str:
        return f"{self.name:10s}: {super().__repr__()}"

    @staticmethod
    def coordinates_from_experiment(
        name: str, experiment: "Experiment"
    ) -> coordinates.SkyCoord:

        # Read source data from VEX file
        if name not in experiment.vex["SOURCE"]:
            log.error(
                f"Failed to generate source: {name} not found in VEX file"
            )
            exit(1)
        source_info = experiment.vex["SOURCE"][name]

        # Read ICRF coordinates
        if source_info["ref_coord_frame"] != "J2000":
            raise NotImplementedError(
                f"Failed to generate {name} source: "
                f"Invalid reference frame {source_info['ref_coord_frame']}"
            )
        coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )

        return coords

    @abstractmethod
    def spherical_coordinates(self, obs: "Observation") -> tuple:
        pass


class NearFieldSource(Source):
    """Near-field source"""

    def __init__(self, name: str, ra: float, dec: float, spice_id: str) -> None:

        super().__init__(name, ra, dec)
        self.spice_id = spice_id

        # Link frequency data
        self.three_way_ramping: dict[
            tuple[time.Time, time.Time], tuple[float, float, str]
        ] = NotImplemented
        self.one_way_ramping: dict[
            tuple[time.Time, time.Time], tuple[float, float]
        ] = NotImplemented
        self.reference_downlink_freq: float = NotImplemented

        return None

    @staticmethod
    def from_experiment(
        name: str, experiment: "Experiment"
    ) -> "NearFieldSource":

        # Get source coordinates
        coords = Source.coordinates_from_experiment(name, experiment)

        # Initialize near-field source
        source = NearFieldSource(
            name,
            coords.ra.rad,  # type: ignore
            coords.dec.rad,  # type: ignore
            experiment.setup.general["target"],
        )
        source.exp = experiment

        # # Load link frequency data
        # target = experiment.setup.general["target"]
        # path_base: str = experiment.setup.catalogues["frequency_ramping"]
        # source.three_way_ramping = io.load_three_way_ramping_data(
        #     f"{path_base}3w.{target}"
        # )
        # source.one_way_ramping = io.load_one_way_ramping_data(
        #     f"{path_base}1w.{target}"
        # )
        # source.reference_downlink_freq = io.get_target_information(target)[
        #     "downlink_freq"
        # ]

        return source

    def tx_from_rx(self, rx: "time.Time", station: "Station") -> time.Time:
        """Calculate TX epoch from RX epoch at a station"""

        clight = spice.clight() * 1e3

        # Calculate GCRF coordinates of station at RX
        xsta_gcrf_rx = station.location(rx, frame="icrf")

        # Calculate BCRS position of source at RX
        et_rx: np.ndarray = (rx.tdb - J2000.tdb).to("s").value  # type: ignore
        xsrc_bcrf_rx = (
            np.array(
                spice.spkpos(self.spice_id, et_rx, "J2000", "NONE", "SSB")[0]
            )
            * 1e3
        )

        # Calculate GM and BCRS position of celestial bodies at RX
        bodies = self.exp.setup.internal["lt_correction_bodies"]
        bodies_gm = (
            np.array([spice.bodvrd(body, "GM", 1)[1][0] for body in bodies])
            * 1e9
        )
        xbodies_bcrf_rx = np.array(
            [
                np.array(spice.spkpos(body, et_rx, "J2000", "NONE", "SSB")[0])
                * 1e3
                for body in bodies
            ]
        )

        # Calculate Newtonian potential of all solar system bodies at geocenter
        _earth_idx = bodies.index("earth")
        searth_bcrf_rx = (
            np.array(spice.spkezr("earth", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        xearth_bcrf_rx = searth_bcrf_rx[:, :3]
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        xbodies_gcrf_rx = np.delete(
            xbodies_bcrf_rx - xearth_bcrf_rx, _earth_idx, axis=0
        )
        bodies_gm_noearth = np.delete(bodies_gm, _earth_idx, axis=0)
        U_earth = np.sum(
            bodies_gm_noearth[:, None]
            / np.linalg.norm(xbodies_gcrf_rx, axis=-1),
            axis=0,
        )

        # Calculate BCRS position of station at RX
        xsta_bcrf_rx = (
            xearth_bcrf_rx
            + (1.0 - L_C - (U_earth / (clight * clight)))[:, None]
            * xsta_gcrf_rx
            - (
                np.sum(vearth_bcrf_rx.T * xsta_gcrf_rx.T, axis=0)[:, None]
                * vearth_bcrf_rx
                / (2.0 * clight * clight)
            )
        )

        # Calculate relative position of station wrt to celestial bodies at RX
        r1b = xsta_bcrf_rx[None, :, :] - xbodies_bcrf_rx
        r1b_mag = np.linalg.norm(r1b, axis=-1)  # (M, N)

        # Initialize light travel time between source and station
        lt_0 = np.linalg.norm(xsta_bcrf_rx - xsrc_bcrf_rx, axis=-1) / clight
        tx_0: time.Time = rx.tdb - time.TimeDelta(lt_0, format="sec")  # type: ignore

        # Initialize variables for iterative estimation of TX
        lt_i = 0.0 * lt_0
        n_i = 0
        precision = float(self.exp.setup.internal["lt_precision"])
        n_max = self.exp.setup.internal["lt_max_iterations"]

        # Iterative correction of TX
        while np.any(np.abs(lt_0 - lt_i) > precision) and (n_i < n_max):

            # Update light travel time and TX
            lt_i = lt_0
            tx_i = rx.tdb - time.TimeDelta(lt_i, format="sec")

            # Convert TX to ephemeris time
            et_tx: np.ndarray = (
                (tx_i.tdb - J2000.tdb).to("s").value  # type: ignore
            )

            # Calculate BCRF coordinates of source at TX
            ssrc_bcrf_tx = (
                np.array(
                    spice.spkezr(self.spice_id, et_tx, "J2000", "NONE", "SSB")[
                        0
                    ]
                )
                * 1e3
            )
            xsrc_bcrf_tx = ssrc_bcrf_tx[:, :3]
            vsrc_bcrf_tx = ssrc_bcrf_tx[:, 3:]

            # Calculate BCRF coordinates of celestial bodies at TX
            xbodies_bcrf_tx = np.array(
                [
                    np.array(
                        spice.spkpos(body, et_tx, "J2000", "NONE", "SSB")[0]
                    )
                    * 1e3
                    for body in bodies
                ]
            )

            # Calculate relativistic correction
            r01 = xsta_bcrf_rx - xsrc_bcrf_tx  # (N, 3)
            r01_mag = np.linalg.norm(r01, axis=-1)  # (N,)
            r0b = xsrc_bcrf_tx[None, :, :] - xbodies_bcrf_tx  # (M, N, 3)
            r0b_mag = np.linalg.norm(r0b, axis=-1)  # (M, N)
            r01b_mag = np.linalg.norm(r1b - r0b, axis=-1)  # (M, N)
            gmc = 2.0 * bodies_gm[:, None] / (clight * clight)  # (M, 1)
            rlt_01 = np.sum(
                (gmc / clight)
                * np.log(
                    (r0b_mag + r1b_mag + r01b_mag + gmc)
                    / (r0b_mag + r1b_mag - r01b_mag + gmc)
                ),
                axis=0,
            )

            # Update light travel time and TX
            dot_p01_c = (
                np.sum((r01 / r01_mag[:, None]) * vsrc_bcrf_tx, axis=-1)
                / clight
            )  # (N,)
            lt_0 -= (lt_0 - (r01_mag / clight) - rlt_01) / (1.0 - dot_p01_c)
            tx_0 = rx.tdb - time.TimeDelta(lt_0, format="sec")  # type: ignore

            # Update iteration counter
            n_i += 1

        return tx_0

    def spherical_coordinates(self, obs: "Observation"):

        # Calculate TX epochs for observation
        tx = self.tx_from_rx(obs.tstamps, obs.station)

        # Convert TX and RX epochs to ephemeris time
        et_tx: np.ndarray = (tx.tdb - J2000.tdb).to("s").value  # type: ignore
        et_rx: np.ndarray = (
            (obs.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        )

        # Calculate aberrated position of source wrt station
        xsrc_gcrf_tx = np.array(
            spice.spkpos(self.spice_id, et_tx, "J2000", "NONE", "EARTH")[0]
            * 1e3
        )
        xsrc_sta_ab = xsrc_gcrf_tx - obs.station.location(obs.tstamps, "icrf")

        # Calculate aberrated position of source wrt Earth
        xsrc_bcrf_tx = np.array(
            spice.spkpos(self.spice_id, et_tx, "J2000", "NONE", "SSB")[0] * 1e3
        )
        xearth_bcrf_rx = np.array(
            spice.spkpos("earth", et_rx, "J2000", "NONE", "SSB")[0] * 1e3
        )
        xsrc_earth_ab = (xsrc_bcrf_tx - xearth_bcrf_rx).T

        # Calculate aberrated pointing vector in SEU [az, el]
        k_gcrf = xsrc_sta_ab / np.linalg.norm(xsrc_sta_ab, axis=-1)[:, None]
        k_itrf = obs.icrf2itrf @ k_gcrf[:, :, None]
        s, e, u = (obs.seu2itrf.swapaxes(-1, -2) @ k_itrf).squeeze().T

        # Calculate azimuth and elevation
        az = np.arctan2(e, -s)
        az += (az < 0.0) * 2.0 * np.pi
        el = np.arcsin(u)

        # Calculate right ascension and declination
        ra = np.arctan2(xsrc_earth_ab[1], xsrc_earth_ab[0])
        ra += (ra < 0.0) * 2.0 * np.pi
        dec = np.arctan2(
            xsrc_earth_ab[2],
            np.sqrt(xsrc_earth_ab[0] ** 2 + xsrc_earth_ab[1] ** 2),
        )

        return az, el, ra, dec


class FarFieldSource(Source):
    """Far-field source"""

    def __init__(self, name: str, ra: float, dec: float) -> None:
        super().__init__(name, ra, dec)
        return None

    @staticmethod
    def from_experiment(
        name: str, experiment: "Experiment"
    ) -> "FarFieldSource":

        coords = Source.coordinates_from_experiment(name, experiment)
        source = FarFieldSource(name, coords.ra.rad, coords.dec.rad)  # type: ignore
        source.exp = experiment
        return source

    def spherical_coordinates(self, obs: "Observation"):
        """Aberrated spherical coordinates of source at observation epochs

        Calculates azimuth and elevation of source corrected for diurnal (motion of station due to Earth's rotation) and annual (motion of station due to Earth's orbit around the Sun) aberration. Under the assumption that the source remains static during the period of observation, the right ascension and declination are taken directly from the VEX file.

        Source: Spherical Astrometry (сферическая астрометрия), Zharov (2006) - Eq 5.105 [Available online as of 01/2025]
        """

        clight = spice.clight() * 1e3

        # Convert RX to ephemeris time
        et_rx: np.ndarray = (
            (obs.tstamps.tdb - J2000.tdb).to("s").value  # type: ignore
        )

        # Calculate BCRF velocity of station at RX
        searth_bcrf_rx = (
            np.array(spice.spkezr("EARTH", et_rx, "J2000", "NONE", "SSB")[0])
            * 1e3
        )
        vearth_bcrf_rx = searth_bcrf_rx[:, 3:]
        vsta_bcrf_rx = vearth_bcrf_rx + obs.station.velocity(
            obs.tstamps, frame="icrf"
        )
        v_mag = np.linalg.norm(vsta_bcrf_rx, axis=-1)[:, None]
        v_unit = vsta_bcrf_rx / v_mag

        # Unit vector along non-aberrated pointing direction
        s0 = self.observed_ks[None, :]

        # Aberrated pointing direction [Equation 5.105 from Zharov (2006)]
        v_c = v_mag / clight
        gamma = 1.0 / np.sqrt(1.0 - v_c * v_c)
        s0_dot_n = np.sum(s0 * v_unit, axis=-1)[:, None]
        s_aber = (
            (s0 / gamma)
            + (v_c * v_unit)
            + ((gamma - 1.0) * s0_dot_n * v_unit / gamma)
        ) / (1.0 + v_c * s0_dot_n)
        s_aber_icrf = s_aber / np.linalg.norm(s_aber, axis=-1)[:, None]

        # Transform pointing direction from GCRF to SEU
        s_aber_itrf = obs.icrf2itrf @ s_aber_icrf[:, :, None]
        s_aber_seu = (obs.seu2itrf.swapaxes(-1, -2) @ s_aber_itrf).squeeze().T

        # Calculate azimuth and elevation
        el = np.arcsin(s_aber_seu[2])
        az = np.arctan2(s_aber_seu[1], -s_aber_seu[0])
        az += (az < 0.0) * 2.0 * np.pi

        # Calculate station-centric right ascension and declination
        ra = np.ones_like(az) * self.observed_ra
        dec = np.ones_like(el) * self.observed_dec

        return az, el, ra, dec
