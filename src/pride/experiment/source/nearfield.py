from typing import TYPE_CHECKING
from astropy import time
from ...logger import log
import numpy as np
import spiceypy as spice
from ...constants import J2000, L_C
from ... import io
from .core import Source

if TYPE_CHECKING:
    from ..experiment import Experiment
    from ..observation import Observation
    from ..station import Station


class NearFieldSource(Source):

    def __init__(self, name: str) -> None:
        print(f"Constructor NearFieldSource: {name}")
        super().__init__(name)
        self.is_nearfield = True
        return None

    @staticmethod
    def from_experiment(exp: "Experiment", name: str = "") -> "Source":

        # Initialize source
        source = NearFieldSource(exp.target["short_name"])
        source.spice_id = exp.target["short_name"]

        # Load ramping data for three-way link
        path_base: str = exp.setup.catalogues["frequency_ramping"]
        _3way_source = io.internal_file(f"{path_base}3w.{source.spice_id}")
        _3way_data = io.load_ramping_data(
            _3way_source, "three-way", (exp.initial_epoch, exp.final_epoch)
        )
        if _3way_data is not None:
            source.three_way_ramping = _3way_data
            source.has_three_way_ramping = True
        else:
            log.warning(f"Three-way ramping data not found for {source.name}")

        # Load ramping data for one-way link
        _1way_source = io.internal_file(f"{path_base}1w.{source.spice_id}")
        _1way_data = io.load_ramping_data(
            _1way_source, "one-way", (exp.initial_epoch, exp.final_epoch)
        )
        if _1way_data is not None:
            source.one_way_ramping = _1way_data
            source.has_one_way_ramping = True
        else:
            log.warning(f"One-way ramping data not found for {source.name}")

        # Set default downlink frequency
        source.default_frequency = exp.target["downlink_frequency"] * 1e6  # Hz

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
        bodies = io.load_catalog("config.yaml")["Configuration"][
            "lt_correction_bodies"
        ]
        # bodies = self.exp.setup.internal["lt_correction_bodies"]
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
        precision = float(
            io.load_catalog("config.yaml")["Configuration"]["lt_precision"]
        )
        n_max = io.load_catalog("config.yaml")["Configuration"][
            "lt_max_iterations"
        ]
        # precision = float(self.exp.setup.internal["lt_precision"])
        # n_max = self.exp.setup.internal["lt_max_iterations"]

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
