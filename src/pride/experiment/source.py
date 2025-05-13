from typing import TYPE_CHECKING, Any
from astropy import coordinates, time
from ..logger import log
import numpy as np
import spiceypy as spice
from ..constants import J2000, L_C
from abc import abstractmethod, ABCMeta
from ..io import internal_file


if TYPE_CHECKING:
    from .experiment import Experiment
    from .observation import Observation
    from .station import Station


class Source(metaclass=ABCMeta):
    """Base class for radio sources"""

    __slots__ = (
        "name",
        "observed_ra",
        "observed_dec",
        "observed_ks",
        "exp",
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
        self.exp: "Experiment" = NotImplemented
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


class FarFieldSource(Source):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.is_farfield = True
        return None

    @staticmethod
    def from_experiment(exp: "Experiment", name: str) -> "Source":

        # Initialize source
        source = FarFieldSource(name)
        source.exp = exp

        # Read coordinates from VEX file
        source_info = exp._Experiment__vex._Vex__content["SOURCE"][name]
        _coords = coordinates.SkyCoord(
            source_info["ra"], source_info["dec"], frame="icrs"
        )
        _ra = float(_coords.ra.to("rad").value)  # type: ignore
        _dec = float(_coords.dec.to("rad").value)  # type: ignore
        source.observed_ra = _ra
        source.observed_dec = _dec

        # Calculate pointing vector
        source.observed_ks = np.array(
            [
                np.cos(_ra) * np.cos(_dec),
                np.sin(_ra) * np.cos(_dec),
                np.sin(_dec),
            ]
        )

        return source

    def spherical_coordinates(
        self, obs: "Observation"
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, time.Time | None
    ]:
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

        return az, el, ra, dec, None


class NearFieldSource(Source):

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.is_nearfield = True
        return None

    @staticmethod
    def from_experiment(exp: "Experiment", name: str = "") -> "Source":

        # Initialize source
        source = NearFieldSource(exp.target["short_name"])
        source.exp = exp
        source.spice_id = exp.target["short_name"]

        # Load ramping data for three-way link
        path_base: str = exp.setup.catalogues["frequency_ramping"]
        _3way_data = {"t0": [], "t1": [], "f0": [], "df": [], "uplink": []}
        _3way_source = internal_file(f"{path_base}3w.{source.spice_id}")

        if not _3way_source.exists():
            log.warning(
                f"Three-way ramping data not found for {source.spice_id}"
            )
        else:
            with _3way_source.open() as f:

                # Load data from internal file
                for line in f:

                    # Skip comments and empty lines
                    content = line.strip().split()
                    if "#" in line or len(content) == 0:
                        continue

                    # Filter out data after the end of the experiment
                    t0_str = "T".join(content[:2])
                    if time.Time(t0_str) > exp.final_epoch:
                        break

                    # Read data from line
                    _3way_data["t0"].append(t0_str)
                    _3way_data["t1"].append("T".join(content[2:4]))
                    _3way_data["f0"].append(float(content[4]))
                    _3way_data["df"].append(float(content[5]))
                    _3way_data["uplink"].append(content[6])

            # Update attribute
            if len(_3way_data["t0"]) > 0:
                source.three_way_ramping = {
                    "t0": time.Time(_3way_data["t0"]),
                    "t1": time.Time(_3way_data["t1"]),
                    "f0": np.array(_3way_data["f0"]),
                    "df": np.array(_3way_data["df"]),
                    "uplink": _3way_data["uplink"],
                }
                source.has_three_way_ramping = True
            else:
                log.warning(
                    f"Three-way ramping data not found for {source.spice_id}"
                )

        # Load ramping data for one-way link
        _1way_data = {"t0": [], "t1": [], "f0": [], "df": []}
        _1way_source = internal_file(f"{path_base}1w.{source.spice_id}")

        if not _1way_source.exists():
            log.warning(f"One-way ramping data not found for {source.spice_id}")
        else:
            with _1way_source.open() as f:

                # Load data from internal file
                for line in f:

                    # Skip comments and empty lines
                    content = line.strip().split()
                    if "#" in line or len(content) == 0:
                        continue

                    # Filter out data after the end of the experiment
                    t0_str = "T".join(content[:2])
                    if time.Time(t0_str) > exp.final_epoch:
                        break

                    # Read data from line
                    _1way_data["t0"].append(t0_str)
                    _1way_data["t1"].append("T".join(content[2:4]))
                    _1way_data["f0"].append(float(content[4]))
                    _1way_data["df"].append(float(content[5]))

            # Update attribute
            if len(_1way_data["f0"]) > 0:
                source.one_way_ramping = {
                    "t0": time.Time(_1way_data["t0"]),
                    "t1": time.Time(_1way_data["t1"]),
                    "f0": np.array(_1way_data["f0"]),
                    "df": np.array(_1way_data["df"]),
                }
                source.has_one_way_ramping = True
            else:
                log.warning(
                    f"One-way ramping data not found for {source.spice_id}"
                )

        # Set default downlink frequency
        source.default_frequency = exp.target["downlink_frequency"] * 1e6  # Hz

        # # Read coordinates from VEX file
        # _ra: list[str] = []
        # _dec: list[str] = []
        # for source_id, source_info in exp.vex["SOURCE"].items():

        #     # Ensure that type information is available
        #     if "source_type" not in source_info:
        #         log.error(
        #             "Failed to initialize near field source: "
        #             f"Source type not found for {source_id}"
        #         )
        #         exit(1)

        #     # Skip if source is not near field
        #     if source_info["source_type"] != "target":
        #         continue

        #     # Ensure that reference frame is valid
        #     if source_info["ref_coord_frame"] != "J2000":
        #         raise NotImplementedError(
        #             "Failed to generate near field source: "
        #             f"Invalid reference frame {source_info['ref_coord_frame']}"
        #         )

        #     # Append coordinates
        #     _ra.append(source_info["ra"])
        #     _dec.append(source_info["dec"])

        # # Combine coordinates and convert to radians
        # _coords = coordinates.SkyCoord(_ra, _dec, frame="icrs")
        # source.observed_ra = np.array(
        #     _coords.ra.to("rad").value,  # type: ignore
        #     dtype=float,
        # )
        # source.observed_dec = np.array(
        #     _coords.dec.to("rad").value,  # type: ignore
        #     dtype=float,
        # )

        # # Calculate pointing vector
        # source.observed_ks = np.array(
        #     [
        #         np.cos(source.observed_ra) * np.cos(source.observed_dec),
        #         np.sin(source.observed_ra) * np.cos(source.observed_dec),
        #         np.sin(source.observed_dec),
        #     ]
        # )

        return source

    def tx_from_rx(self, rx: "time.Time", station: "Station") -> time.Time:
        """Calculate TX epoch from RX epoch at a station"""

        # Sanity
        if station.is_phase_center:
            log.error(
                "Calculation of TX from RX not valid for station at geocenter"
            )
            exit(1)

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
        # Function: F(TX) = RX - TX - R_01/c - RLT_01
        # Derivative: dF/dTX = -1 + (R_01_vec * dR_0_vec/dTX) / (R_01 * c)
        # Newton-Raphson: TX_{i+1} = TX_i - F(TX_i) / dF/dTX
        # Equivalent: LT_{i+1} = LT_i + F(TX_i) / dF/dTX
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

            # Evaluate function and derivative for Newton-Raphson
            f = lt_i - (r01_mag / clight) - rlt_01
            dfdtx = (
                -1.0
                + np.sum((r01 / r01_mag[:, None]) * vsrc_bcrf_tx, axis=-1)
                / clight
            )

            # Update light travel time and TX
            lt_0 = lt_i + f / dfdtx
            tx_0 = rx.tdb - time.TimeDelta(lt_0, format="sec")  # type: ignore

            # # Update light travel time and TX
            # dot_p01_c = (
            #     np.sum((r01 / r01_mag[:, None]) * vsrc_bcrf_tx, axis=-1)
            #     / clight
            # )  # (N,)
            # lt_0 -= (lt_0 - (r01_mag / clight) - rlt_01) / (1.0 - dot_p01_c)
            # tx_0 = rx.tdb - time.TimeDelta(lt_0, format="sec")  # type: ignore

            # Update iteration counter
            n_i += 1

        return tx_0

    def spherical_coordinates(
        self, obs: "Observation"
    ) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, time.Time | None
    ]:

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

        return az, el, ra, dec, tx
